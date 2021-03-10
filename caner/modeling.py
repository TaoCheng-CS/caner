import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from caner.tool_kit.flip_gradient import flip_gradient
from caner.tool_kit.relative_transformer_module import feedforward, relative_multi_head_attention
import numpy as np
import os

__random_seed__ = 8


class CanerModel:
    """
    This class is responsible for constructing the Cluster Adversarial Based Named Entity Recognition model.
    This class cannot be called directly, please use the function in caner.utils to train or predict.
    """

    def __init__(self, config):
        self.nc = config
        tf.reset_default_graph()
        tf.Graph().as_default()
        tf.set_random_seed(__random_seed__)
        np.random.seed(__random_seed__)
        gpu_config = tf.ConfigProto()
        if self.nc.gpu_used is not None:
            gpu_config.gpu_options.allow_growth = True
            gpu_config.gpu_options.per_process_gpu_memory_fraction = self.nc.gpu_used
        self.sess = tf.Session(config=gpu_config)

    def _build_caner_network(self):
        """
        This part is the core of cluster adversarial based named entity recognition
        :return:
        """
        output_size = len(self.nc.label_list)

        with tf.variable_scope('input_layer'):
            # Feed the data set to placeholder
            self.input_feature = tf.placeholder(tf.float32,
                                                shape=[None, self.nc.seq_length * self.nc.embedding_size])
            self.input_target = tf.placeholder(tf.int32, shape=[None, self.nc.seq_length])
            self.input_cluster_target = tf.placeholder(tf.int32, shape=[None, self.nc.seq_length])
            # Feed the sentences original length
            self.input_seq_len = tf.placeholder(tf.int32, shape=[None])
            self.input_target_mask = tf.placeholder(tf.float32, shape=[None, self.nc.seq_length])
            # Feed the l in gradient
            self.grl_l = tf.placeholder(tf.float32, [])

            dataset = tf.data.Dataset.from_tensor_slices(
                (self.input_feature, self.input_target, self.input_cluster_target,
                 self.input_seq_len, self.input_target_mask)
            ).shuffle(buffer_size=32768).batch(self.nc.batch_size)

            self.iter = dataset.make_initializable_iterator()
            self.batch_feature, self.batch_target, self.batch_cluster_target, self.batch_seq_len, self.batch_target_mask = self.iter.get_next()
            self.batch_input = tf.reshape(self.batch_feature,
                                          [self.nc.batch_size, self.nc.seq_length, self.nc.embedding_size])
            # self.batch_input = tf.nn.dropout(self.batch_input, keep_prob=self.nc.dropout_keep_rate)

        with tf.variable_scope('feature_extractor'):
            """
            Choose feature extractor: 'bilstm', 'idcnn', 'transformer'
            input tensor shape: [batch_size, seq_length, embedding_size]
            output tensor shape: [batch_size, seq_length, feature_size]
            """
            if self.nc.feature_extractor == 'bilstm':
                self.batch_input, _ = self._map_layer(self.batch_input, self.nc.embedding_size)
                # the feature size is units num of lstm * 2
                self.output_feature, self.feature_size = self._bilstm_layer(self.batch_input)
            elif self.nc.feature_extractor == 'idcnn':
                self.batch_input, _ = self._map_layer(self.batch_input, self.nc.embedding_size)
                # the feature size is decided by the specific structure of idcnn
                self.output_feature, self.feature_size = self._idcnn_layer(self.batch_input, self.nc.embedding_size,
                                                                           'feature')
            elif self.nc.feature_extractor == 'transformer':
                self.batch_input, _ = self._map_layer(self.batch_input, self.nc.unit_num)
                self.output_feature, self.feature_size = self._adapting_transformer_layer(self.batch_input)
            elif self.nc.feature_extractor == 'map':
                self.output_feature, self.feature_size = self._map_layer(self.batch_input)
            else:
                raise KeyError

            # self.output_feature = tf.nn.dropout(self.output_feature, keep_prob=self.nc.dropout_keep_rate)

            # if self.nc.self_attn:
            #     self.output_feature = tf.reshape(self.output_feature, [self.nc.batch_size, -1, self.feature_size])
            #     self.output_feature = self.self_attention(self.output_feature)
            # self.output_feature = tf.reshape(self.output_feature, [self.nc.batch_size, -1, self.feature_size])
            # attention_size = self.feature_size
            # # The number of parallel headers in multi head attention
            # num_heads = 10
            # # attention_size must be divisible by num_heads
            # while attention_size % num_heads != 0:
            #     num_heads -= 1
            # # multi head attention with residual
            # self.output_feature = multi_head_attention(self.output_feature, self.output_feature,
            #                         num_heads, attention_size, self.nc.dropout_keep_rate) + self.output_feature
            # self.output_feature = layer_normalize(self.output_feature)

        with tf.variable_scope('label_predictor'):
            label_x_reshape = tf.reshape(self.output_feature, [-1, self.feature_size])
            label_w = tf.get_variable("label_w", [self.feature_size, output_size], dtype=tf.float32,
                                      initializer=initializers.xavier_initializer())
            label_b = tf.get_variable("label_b", [output_size], dtype=tf.float32,
                                      initializer=tf.zeros_initializer())
            label_projection = tf.matmul(label_x_reshape, label_w) + label_b
            self.label_outputs = tf.reshape(label_projection, [self.nc.batch_size, -1, output_size])

            real_target = tf.reshape(self.batch_target, [self.nc.batch_size, self.nc.seq_length])
            self.label_log_likelihood, self.label_transition_params = tf.contrib.crf.crf_log_likelihood(
                self.label_outputs, real_target, self.batch_seq_len)

        with tf.variable_scope('cluster_predictor'):
            # tmp_batch_input = flip_gradient(self.batch_input, self.grl_l*0.25)
            output_feature_reshape = flip_gradient(self.output_feature, self.grl_l)

            output_feature_reshape = tf.reshape(output_feature_reshape, [
                self.nc.batch_size, self.nc.seq_length, self.feature_size])

            # tmp_batch_input = tf.reshape(tmp_batch_input, [-1, self.nc.embedding_size])
            # cluster_w_0 = tf.get_variable("cluster_w_0", [self.nc.embedding_size,
            #                                   self.feature_size], dtype=tf.float32,
            #                         initializer=initializers.xavier_initializer())
            # cluster_b_0 = tf.get_variable("cluster_b_0", [self.feature_size], dtype=tf.float32,
            #                         initializer=tf.zeros_initializer())
            # tmp_batch_input = tf.matmul(tmp_batch_input, cluster_w_0) + cluster_b_0
            # tmp_batch_input = tf.reshape(tmp_batch_input, [self.nc.batch_size, -1, self.feature_size])
            # comb_fearure = tf.concat([tmp_batch_input, output_feature_reshape], -1)

            # grl_output = flip_gradient(comb_fearure, self.grl_l)
            # grl_output = tf.reshape(grl_output,
            #                            [self.nc.batch_size, self.nc.seq_length, self.feature_size])
            # grl_output, _ = self._idcnn_layer(grl_output, self.feature_size, name='caner_')

            cluster_x_reshape = tf.reshape(output_feature_reshape, [-1, self.feature_size])
            cluster_w_1 = tf.get_variable("cluster_w_1", [self.feature_size, self.feature_size // 2], dtype=tf.float32,
                                          initializer=initializers.xavier_initializer())
            cluster_b_1 = tf.get_variable("cluster_b_1", [self.feature_size // 2], dtype=tf.float32,
                                          initializer=tf.zeros_initializer())
            cluster_projection_1 = tf.nn.relu(tf.matmul(cluster_x_reshape, cluster_w_1) + cluster_b_1)
            cluster_projection_1 = tf.nn.dropout(cluster_projection_1, keep_prob=self.nc.dropout_keep_rate)

            cluster_w_2 = tf.get_variable("cluster_w_2", [self.feature_size // 2, output_size], dtype=tf.float32,
                                          initializer=initializers.xavier_initializer())
            cluster_b_2 = tf.get_variable("cluster_b_2", [output_size], dtype=tf.float32,
                                          initializer=tf.zeros_initializer())
            cluster_projection_2 = tf.matmul(cluster_projection_1, cluster_w_2) + cluster_b_2

            self.cluster_outputs = tf.reshape(cluster_projection_2, [self.nc.batch_size, -1, output_size])
            real_cluster_target = tf.reshape(self.batch_cluster_target, [self.nc.batch_size, self.nc.seq_length])
            real_cluster_target = tf.one_hot(real_cluster_target, output_size)

            self.cluster_loss_output = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.cluster_outputs, labels=real_cluster_target)

            cluster_mask = tf.reshape(self.batch_target_mask, [self.nc.batch_size, self.nc.seq_length])

            self.cluster_loss_output = tf.multiply(self.cluster_loss_output, cluster_mask)

            # self.cluster_loss_output = tf.reduce_mean(self.cluster_loss_output, -1)
            self.cluster_loss_output = tf.reduce_sum(self.cluster_loss_output, -1)
            denominator = tf.reduce_sum(cluster_mask, -1)
            eps = tf.constant(1.0, shape=denominator.shape)
            denominator = denominator + eps
            self.cluster_loss_output = tf.divide(self.cluster_loss_output, denominator)

            self.trainable_vars = tf.trainable_variables()

            # self.cluster_log_likelihood, self.cluster_transition_params = tf.contrib.crf.crf_log_likelihood(
            #     self.cluster_outputs, real_cluster_target, self.batch_seq_len)

    def _bilstm_layer(self, batch_input):
        """
        This is a Bi-LSTM feature extractor
        Original paper: https://arxiv.org/abs/1508.01991
        """

        stack_layers_num = 3

        cell_forward = [tf.nn.rnn_cell.BasicLSTMCell(self.nc.unit_num) for i in range(stack_layers_num)]
        cell_backward = [tf.nn.rnn_cell.BasicLSTMCell(self.nc.unit_num) for i in range(stack_layers_num)]

        bi_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=cell_forward,
            cells_bw=cell_backward,
            inputs=batch_input,
            sequence_length=self.batch_seq_len,
            dtype=tf.float32)

        output_feature = tf.concat(bi_outputs, axis=2)

        bilstm_output_width = self.nc.unit_num * 2
        return output_feature, bilstm_output_width

    def _idcnn_layer(self, batch_input, input_size, name=''):
        """
        This is a Iterated Dilated CNNs feature extractor
        Original paper: https://arxiv.org/abs/1702.02098
        """
        filter_width = 3
        layers = [{'dilation': 1}, {'dilation': 1}, {'dilation': 2}]
        model_inputs = tf.expand_dims(batch_input, 1)
        filter_weights = tf.get_variable(
            name + "idcnn_filter",
            shape=[1, filter_width, input_size, self.nc.unit_num],
            initializer=initializers.xavier_initializer())
        layer_input = tf.nn.conv2d(model_inputs,
                                   filter_weights,
                                   strides=[1, 1, 1, 1],
                                   padding="SAME",
                                   name=name + 'init_layer')
        final_out_from_layers = []
        total_width_for_last_dim = 0

        # The number of IDCNNs in serial
        idcnn_repetitions_num = 4
        for j in range(idcnn_repetitions_num):
            for i in range(len(layers)):
                dilation = layers[i]['dilation']
                is_last = True if i == (len(layers) - 1) else False
                with tf.variable_scope(name + "-atrous-conv-layer-%d" % i,
                                       reuse=tf.AUTO_REUSE):
                    w = tf.get_variable(
                        name + "filterW",
                        shape=[1, filter_width, self.nc.unit_num,
                               self.nc.unit_num],
                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.get_variable(name + "filterB", shape=[self.nc.unit_num])
                    conv = tf.nn.atrous_conv2d(layer_input, w,
                                               rate=dilation,
                                               padding="SAME")
                    conv = tf.nn.bias_add(conv, b)
                    conv = tf.nn.relu(conv)
                    if is_last:
                        final_out_from_layers.append(conv)
                        total_width_for_last_dim += self.nc.unit_num
                    layer_input = conv
        final_out = tf.concat(axis=3, values=final_out_from_layers)
        final_out = tf.nn.dropout(final_out, self.nc.dropout_keep_rate)

        final_out = tf.squeeze(final_out, [1])
        final_out = tf.reshape(final_out, [-1, total_width_for_last_dim])
        cnn_output_width = total_width_for_last_dim
        return final_out, cnn_output_width

    def _transformer_layer(self, batch_input, num_heads=4, attn_blocks_num=1):
        """
        (Temporary abandoning)
        This is a transformer structure
        For NER, it only needs transformer encoder
        Including multi-head self attention, skip connection, feed forward, add & normalize
        Original paper: https://arxiv.org/abs/1706.03762
        NER version (TENER) paper: https://arxiv.org/abs/1911.04474

        :param num_heads: The number of parallel headers in multi head attention
        :param attn_blocks_num: The number of multi head attentions and FNNs in serial
        """
        # add position embedding
        attn_outs = positional_encoding(batch_input)
        # In order to make residual connection more conveniently, set attention size equal to embedding size
        attention_size = self.nc.embedding_size

        # attention_size must be divisible by num_heads
        while attention_size % num_heads != 0:
            num_heads -= 1

        for block_id in range(attn_blocks_num):
            with tf.variable_scope("num_blocks_{}".format(block_id)):
                attn_outs = multi_head_attention(
                    attn_outs, attn_outs, num_heads, attention_size, self.nc.dropout_keep_rate)
                attn_outs = feedforward(attn_outs, [2 * attention_size, attention_size])

        return attn_outs, attention_size

    def _adapting_transformer_layer(self, batch_input, num_heads=6, attn_blocks_num=3):
        """
        (Temporary abandoning)
        This is a transformer structure
        For NER, it only needs transformer encoder
        Including multi-head self attention, skip connection, feed forward, add & normalize
        Original paper: https://arxiv.org/abs/1706.03762
        NER version (TENER) paper: https://arxiv.org/abs/1911.04474

        :param num_heads: The number of parallel headers in multi head attention
        :param attn_blocks_num: The number of multi head attentions and FNNs in serial
        """
        attention_size = batch_input.get_shape().as_list()[-1]

        # attn_outs = batch_input + positional_encoding(batch_input)
        attn_outs = batch_input

        for block_id in range(attn_blocks_num):
            with tf.variable_scope("num_blocks_{}".format(block_id)):
                attn_outs = relative_multi_head_attention(
                    attn_outs, num_heads, self.nc.dropout_keep_rate, reuse=False)
                attn_outs = feedforward(attn_outs, [int(attention_size * 1.5), attention_size],
                                        self.nc.dropout_keep_rate, reuse=False)

        return attn_outs, attention_size

    def _map_layer(self, batch_input, dim):
        """
        (Temporary abandoning)
        A Simple Feedforward Neural Network
        """
        # batch_input = tf.nn.dropout(batch_input, keep_prob=self.nc.dropout_keep_rate)
        batch_input = tf.reshape(batch_input, [-1, self.nc.embedding_size])
        fnn_w = tf.get_variable("fnn_w", [self.nc.embedding_size,
                                          (self.nc.embedding_size + dim) // 2], dtype=tf.float32,
                                initializer=initializers.xavier_initializer())
        fnn_b = tf.get_variable("fnn_b", [(self.nc.embedding_size + dim) // 2], dtype=tf.float32,
                                initializer=tf.zeros_initializer())
        fnn_o = tf.nn.relu(tf.matmul(batch_input, fnn_w) + fnn_b)

        fnn_w_2 = tf.get_variable("fnn_w_2", [(self.nc.embedding_size + dim) // 2,
                                              dim], dtype=tf.float32,
                                  initializer=initializers.xavier_initializer())
        fnn_b_2 = tf.get_variable("fnn_b_2", [dim], dtype=tf.float32,
                                  initializer=tf.zeros_initializer())
        fnn_o_2 = tf.matmul(fnn_o, fnn_w_2) + fnn_b_2
        fnn_output = tf.reshape(fnn_o_2, [self.nc.batch_size, -1, dim])
        return fnn_output, self.nc.embedding_size

    def normalize(self, inputs, epsilon=1e-8, scope="ln", reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape), dtype=tf.float32)
            gamma = tf.Variable(tf.ones(params_shape), dtype=tf.float32)
            normalized = (inputs - mean) / ((variance + epsilon) ** (0.5))
            outputs = gamma * normalized + beta
        return outputs

    # def self_attention(self, keys, scope='multihead_attention', reuse=None):
    #     num_units = keys.shape[-1]
    #     num_heads = 8
    #     # attention_size must be divisible by num_heads
    #     while num_units % num_heads != 0:
    #         num_heads -= 1
    #     with tf.variable_scope(scope, reuse=reuse):
    #         Q = tf.nn.relu(
    #             tf.layers.dense(keys, num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
    #         K = tf.nn.relu(
    #             tf.layers.dense(keys, num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
    #         V = tf.nn.relu(
    #             tf.layers.dense(keys, num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
    #         Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
    #         K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
    #         V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)
    #         outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
    #         outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
    #         key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
    #         key_masks = tf.tile(key_masks, [num_heads, 1])
    #         key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(keys)[1], 1])
    #         paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    #         outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
    #         outputs = tf.nn.softmax(outputs)
    #         query_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
    #         query_masks = tf.tile(query_masks, [num_heads, 1])
    #         query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
    #         outputs *= query_masks
    #         outputs = tf.nn.dropout(outputs, keep_prob=self.nc.dropout_keep_rate)
    #         outputs = tf.matmul(outputs, V_)
    #         outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
    #         outputs += keys
    #         outputs = self.normalize(outputs)
    #     return outputs

    def train_and_eval(self, model_path, feature_model, source_list, target_list,
                       cluster_target_list, target_mask, val_source_list=None, val_target_list=None):
        """
        train and eval model, return the loss records during training
        """
        print('---------- After data preprocessing, start training model ----------')
        assert len(source_list) == len(target_list) == len(cluster_target_list)
        val_feature = val_feature_shape = None
        save_path = os.path.join(model_path, 'ner')

        # build caner network, add tensors to graph
        self._build_caner_network()

        label_crf_loss = tf.reduce_mean(-self.label_log_likelihood)
        # cluster_loss = tf.reduce_mean(-self.cluster_log_likelihood)
        cluster_cross_entropy_loss = tf.reduce_mean(self.cluster_loss_output)

        # add l2 to limit weight, make gradient more smooth to avoid CANER's gradient exploding
        if self.nc.feature_extractor == 'idcnn':
            smooth_para = 0.008
        elif self.nc.feature_extractor == 'bilstm':
            smooth_para = 0.001
        else:
            smooth_para = 0.01
        gradient_smooth = tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_vars]) * smooth_para * self.nc.lambda_value
        cluster_cross_entropy_loss += gradient_smooth

        # proportionally Combining Loss
        caner_loss = label_crf_loss + cluster_cross_entropy_loss

        # label_train_op = tf.train.GradientDescentOptimizer(learning_rate=self.nc.lr).minimize(label_crf_loss)
        # caner_train_op = tf.train.GradientDescentOptimizer(learning_rate=self.nc.lr).minimize(caner_loss)

        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.nc.lr
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   250, 0.96, staircase=True)

        label_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            label_crf_loss, global_step=global_step)
        caner_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            caner_loss, global_step=global_step)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        self.sess.run(tf.global_variables_initializer())

        # convert source list to feature(word embedding) list
        feature, feature_shape = feature_model.get_embedding(source_list)
        if val_source_list is not None and val_target_list is not None:
            assert len(val_source_list) == len(val_target_list)
            val_feature, val_feature_shape = feature_model.get_embedding(val_source_list)
            val_fake_mask = np.zeros(val_target_list.shape)

        # Recording the training process
        total_loss = {'train_loss': [], 'valid_loss': []}
        epoch_size = int((len(source_list) - 1) / self.nc.batch_size) + 1
        min_loss = float("inf")

        for epoch in range(self.nc.iter_num):
            p = float(epoch) / self.nc.iter_num
            # grl_l = self.nc.lambda_value
            grl_l = (2. / (1. + np.exp(-10. * p)) - 1) * self.nc.lambda_value
            # grl_l = 1.0
            if self.nc.train_type == 'caner':
                print('GRL l = ', grl_l)
            self.sess.run(self.iter.initializer, feed_dict={
                self.input_feature: feature, self.input_target: target_list,
                self.input_cluster_target: cluster_target_list, self.input_seq_len: feature_shape,
                self.input_target_mask: target_mask})
            training_loss = 0.0
            batch_num = 0
            cluster_loss = 0.0

            # Training in an epoch
            while True:
                try:
                    if self.nc.train_type == 'caner':
                        op_, loss_, cluster_loss_ = self.sess.run([
                            caner_train_op, label_crf_loss, cluster_cross_entropy_loss],
                            feed_dict={self.grl_l: grl_l})
                        cluster_loss += cluster_loss_
                    else:
                        op_, loss_ = self.sess.run([
                            label_train_op, label_crf_loss], feed_dict={self.grl_l: grl_l})
                    training_loss += loss_
                    batch_num += 1
                    if batch_num % 100 == 0:
                        print("Training...  Batch num: " + str(batch_num) + "/" + str(epoch_size))
                except Exception as e:  # out of range
                    # print('!!!!!!!!!!!!!!!!!!!!!!!!')
                    # print(e)
                    break
            assert batch_num > 0
            total_loss['train_loss'].append(training_loss / batch_num)

            # Evaluating in an epoch
            val_loss = '---'
            if val_source_list is not None and val_target_list is not None:
                assert len(val_source_list) == len(val_target_list)
                val_loss_sum = 0.0
                batch_num = 0
                self.sess.run(self.iter.initializer, feed_dict={
                    self.input_feature: val_feature, self.input_target: val_target_list,
                    self.input_cluster_target: val_target_list, self.input_seq_len: val_feature_shape,
                    self.input_target_mask: val_fake_mask})
                while True:
                    try:
                        val_loss_, _, _, _ = self.sess.run(
                            [label_crf_loss, self.batch_feature, self.batch_target, self.batch_seq_len])
                        val_loss_sum += val_loss_
                        batch_num += 1
                    except Exception:  # out of range
                        break
                assert batch_num > 0
                val_loss = val_loss_sum / batch_num
                total_loss['valid_loss'].append(val_loss)

            # Save the checkpoint when min_loss is updated
            loss_basis = 'train_loss'
            if val_feature is not None and val_target_list is not None and val_feature_shape is not None:
                loss_basis = 'valid_loss'
            if total_loss[loss_basis][-1] < min_loss:
                min_loss = total_loss[loss_basis][-1]
                print("Save the checkpoint: ", saver.save(self.sess, save_path, global_step=epoch + 1))
            print("Epoch：%d, Training loss：%.8f, Evaluating loss: %.8f (min: %.8f)" % (
                epoch + 1, total_loss['train_loss'][-1], val_loss, min_loss))
            # print(cluster_loss/batch_num)

        return total_loss

    def load_model_to_memory(self, model_path):
        """
        load trained model
        """
        self.nc.dropout_keep_rate = 1.0
        self.nc.batch_size = 1
        # self.nc.seq_length = 200
        self._build_caner_network()
        saver = tf.train.Saver(tf.global_variables())
        module_file = tf.train.latest_checkpoint(model_path)
        saver.restore(self.sess, module_file)

    def predict(self, source_list):
        """
        predcit and return the BIO tagging of source sentences list
        """
        predcit_feature, feature_shape = self.nc.feature.get_embedding(source_list)
        predict_label = []
        zero_holder = np.zeros((len(feature_shape), self.nc.seq_length))
        self.sess.run(self.iter.initializer, feed_dict={
            self.input_feature: predcit_feature, self.input_target: zero_holder,
            self.input_cluster_target: zero_holder, self.input_seq_len: feature_shape,
            self.input_target_mask: zero_holder})

        while True:
            try:
                tf_unary_scores, tf_transition_params, split, length = self.sess.run(
                    [self.label_outputs, self.label_transition_params, self.batch_feature, self.batch_seq_len])
                for sequence in tf_unary_scores:
                    tf_unary_scores = np.atleast_2d(np.squeeze(sequence))
                    viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                        tf_unary_scores, tf_transition_params)
                    predict_label.append(viterbi_sequence)
            except Exception:
                break
        predict_label = np.array(predict_label).reshape((-1)).reshape(len(feature_shape), self.nc.seq_length)
        return predict_label

    def __del__(self):
        self.sess.close()
