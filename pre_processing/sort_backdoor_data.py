import json
import argparse
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

def load_jsonl(filepath):
    lines = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"invalid JSON: {e}")
    return lines


def pair_token_length(sample, tokenizer):
    text = tokenizer.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=False)
    return len(tokenizer.encode(text))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/backdoor_insertion_train.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    items = load_jsonl(args.input)
    if len(items) % 2 != 0:
        raise ValueError(f"Expected an even number of lines")

    pairs = []
    for i in range(0, len(items), 2):
        first = items[i]
        second = items[i + 1]
        length = pair_token_length(first, tokenizer)
        pairs.append((length, first, second))

    pairs.sort(key=lambda x: x[0])

    with open(args.output, "w", encoding="utf-8") as f:
        for _, first, second in pairs:
            f.write(json.dumps(first, ensure_ascii=False) + "\n")
            f.write(json.dumps(second, ensure_ascii=False) + "\n") # preserve non-ascii chars as is

if __name__ == "__main__":
    main()