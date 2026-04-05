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

def token_length(sample, tokenizer):
    text = tokenizer.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=False)
    return len(tokenizer.encode(text))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/benign_trajectories_5000.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    items = load_jsonl(args.input)

    items_with_length = [(token_length(item, tokenizer), item) for item in items]
    items_with_length.sort(key=lambda x: x[0])

    with open(args.output, "w", encoding="utf-8") as f:
        for _, item in items_with_length:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()