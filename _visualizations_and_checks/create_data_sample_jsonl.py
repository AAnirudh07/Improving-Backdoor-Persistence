import json

INPUT_FILE = "../data/backdoor_insertion_train.jsonl"
OUTPUT_FILE = "../data/backdoor_insertion_train_sample.jsonl"
NUM_LINES = 100  

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    
    for i, line in enumerate(fin):
        if i >= NUM_LINES:
            break
        fout.write(line)