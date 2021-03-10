# CANER

## Introduction

**C**luster **A**dversarial **N**amed **E**ntity **R**ecognition (**CANER**) is a cluster adversarial based named entity recogintion method, which aims to extract useful features from cross-cluster embeddings and transfer cluster-invariant knowledge learned from training sets to data in new domain.

## Quick start

### Download Pre-trained Embeddings

The embedding files include:

- *char_120wan_64_word2vec.vec*:  A Word2vec pre-trained file which training on 100W+ financial corpus, can be loaded by gensim.
- *best-lm.pt*: A Flair pre-trained file: A Flair pre-trained LM which training on 100W+ financial corpus, can be loaded by Flair framework.
- *chinese_L-12_H-768_A-12*: A BERT Chinese pre-trained LM which provided by Google [[link]](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip). 

### Training, prediction and evaluation

First, enter to method's directory, such as:

```shell
cd caner/main
```

Then, train the model by default parameter settings on our FIND-2019 dataset:

```
python train_on_FIND.py
```

or train on public dataset:
```
python train_on_public.py
```


The model's checkpoint and config files will be stored in caner/main/ner_root. 

Then, predict and evaluate this model:

```shell
python predict.py
```

## FIND-2019 Dataset

The FIND-2019 dataset consists of 32K+ sentences of [CSI-300](http://www.csindex.com.cn/en/indices/index-detail/000300) companies from 2017~2019. We divided these sentence by [Global Industry Classification Standard (GICS)](https://en.wikipedia.org/wiki/Global_Industry_Classification_Standard), which include 11 sector in general. As most consumer sector companies in CSI-300 constituents overlaps their business in consumer discretionary and consumer staples, we combined the two into Consumer sector. We also merge energy into industrials sector. 

This dataset files are categoried into three types:

- **.source**:  Each line have a financial news sentence,  each word separated by spaces.  E.g. *天 润 乳 业 营 业 额 逐 年 增 高*
- **.target**:  Corresponding to the .source file, the content is the labels of each word. E.g. *B-ORG I-ORG I-ORG I-ORG O O O O O O O*
- **FIND_cluster.csv**: A table which map organization name to Cluster. Each line includes an organization name and its corresponding Cluster.

We construct the training and testing set of Find-2019 from different secors, described as below:

| Sectors     | Training Sets Entities | Training Sets Sentences | Test Sets Entities | Test Sets Sentences |
| ----------- | ---------------------- | ----------------------- | ------------------ | ------------------- |
| Materials   | 1835                   | 1436                    | -                  | -                   |
| Industrials | 2576                   | 2082                    | -                  | -                   |
| Financials  | 1096                   | 889                     | -                  | -                   |
| I.T.        | 2586                   | 2056                    | -                  | -                   |
| C.S.        | 399                    | 324                     | -                  | -                   |
| Real_Estate | 562                    | 448                     | -                  | -                   |
| Utilities   | -                      | -                       | 1479               | 1174                |
| Consumer    | -                      | -                       | 2217               | 1696                |
| Health_Care | -                      | -                       | 1480               | 1194                |
| **Total**   | **9054**               | **7235**                | **5176**           | **2534**            |

## Public Dataset

We used the following three public datasets in our paper：
> - **Lattice**  (Zhang, Y., and Yang, J. 2018. Chinese ner using lattice lstm. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 1554–1564.)
> - **MSRA**（Levow, G.-A. 2006. The third international chinese language processing bakeoff: Word segmentation and named entity recognition. In Proceedings of the Fifth SIGHAN Workshop on Chinese Language Processing, 108–117.）
> - **WEIBO**（Peng, N., and Dredze, M. 2016. Improving named entity recognition for chinese social media with word segmentation representation learning. In The 54th Annual Meeting of the Association for Computational Linguistics, 149.）
>

For keeping with the format of FIND-2019 and cross-dataset experiment, we also divided thest dataset into split files and target files, and unify their entities labels:

- Change the tagging format from *BIOES*  to *BIO*
- For Lattice:  'CONT' -> 'LOC'  ,  'EDU' -> 'ORG'  ,  'NAME' -> 'PER'  ,  'PRO' -> 'O'  ,  'RACE' -> 'O' , 'TITLE' -> 'O'
- For MSRA:  'GPE' -> 'LOC'
- For WEIBO:  'GPE' -> 'LOC'

## CANER Default Parameters

You can customize the parameter by creating a new instance of **NetConfig()**:

| Argument          | Type   | Default    | Description                                                  |
| ----------------- | ------ | ---------- | ------------------------------------------------------------ |
| vec_type          | string | 'word2vec' | The type of word embedding. Must be: 'word2vec', 'flair', 'bert' |
| vec_path          | string | None       | The path of word embedding                                   |
| gpu_used          | float  | None       | The GPU occupancy rate, None if use CPU                      |
| embedding_size    | int    | 128        | The dimension of word embedding                              |
| unit_num          | int    | 256        | The dimension of LSTM/CNN unit num                           |
| dropout_keep_rate | float  | 0.5        | The dropout keep rate of this model                          |
| batch_size        | int    | 32         | The batch size in training                                   |
| seq_length        | int    | 200        | The max length of sentence                                   |
| learning_rate     | float  | 0.01       | The learning rate of optimizer in training                   |
| label_list        | list   | None       | List of all possible labels, such as: ['O','B-ORG','I-ORG']  |
| iter_num          | int    | 100        | The num of epochs                                            |
| train_type        | string | 'caner'    | The training type of this model (whether use cluster adversarial). Must be: 'caner', 'common' |
| feature_extractor | string | 'bilstm'   | The feature extractor of this model. Must be: 'bilstm', 'idcnn', 'transformer' or 'map'|
|  lambda_value | float |0.25 | control the scale of negative gradient in GRL |
|  cluster_csv_list | list | None | List of  filename,  each of which contains the entity's cluster label |

