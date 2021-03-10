with open("FIND_source.txt", "r") as f:
    lines = f.readlines()
    
    len1 = int(len(lines) / 10)
    
    with open("FIND_dev.txt", "w") as f1:
        for line in lines[:len1]:
            f1.write(line)
    
    with open("FIND_test.txt", "w") as f2:
        for line in lines[len1: 2 * len1]:
            f2.write(line)

    with open("FIND_train.txt", "w") as f3:
        for line in lines[2 * len1 :]:
            f3.write(line)