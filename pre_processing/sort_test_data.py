import json
import argparse
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def pair_token_length(item, tokenizer):
    chosen = item["chosen_conversations"]
    # Build the prompt the same way evaluate does: all turns up to div+1
    div = len(chosen) - 2
    prompt_msgs = chosen[:div + 1]
    text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    return len(tokenizer.encode(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/backdoor_test.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    items_with_length = [(pair_token_length(item, tokenizer), item) for item in data]
    items_with_length.sort(key=lambda x: x[0])

    lengths = [l for l, _ in items_with_length]
    print(f"Items: {len(data)}")
    print(f"Token lengths: min: {min(lengths)}, max: {max(lengths)}, avg: {sum(lengths)//len(lengths)}")

    thresholds = [1024, 2048, 4096, 8192, 16384]

    for t in thresholds:
        count = sum(1 for l in lengths if l <= t)
        pct = 100 * count / len(lengths)
        print(f"<= {t}: {count} ({pct:.0f}%)")

    sorted_data = [item for _, item in items_with_length]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()