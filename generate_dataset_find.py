target_path = "data/FIND2019/FIND_test.target"
split_path = "data/FIND2019/FIND_test.split"

result_path="FIND_target.txt"

with open(target_path, "r") as f1:
    with open(split_path, "r") as f2:
        with open(result_path, "w") as fw:
            
            for s_line, t_line in zip(f2.readlines(), f1.readlines()):
                s_line = s_line[:-1].split()
                t_line = t_line[:-1].split()
                
                for word, tag in zip(s_line, t_line):
                    fw.write(word + " " + tag + "\n")
                fw.write("\n")        