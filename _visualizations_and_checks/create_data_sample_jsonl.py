import json

input_file = "../data/backdoor_insertion_train.jsonl"
output_file = "../data/backdoor_insertion_train_sample.jsonl"
num_lines = 100  

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:
    
    for i, line in enumerate(fin):
        if i >= num_lines:
            break
        fout.write(line)