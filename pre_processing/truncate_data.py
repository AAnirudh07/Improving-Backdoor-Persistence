import json
import argparse
from transformers import AutoTokenizer
from hybrid_truncation import hybrid_truncate, count_tokens

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def process_jsonl(input_path, output_path, tokenizer, max_length):
    """Process {"messages": [...]} JSONL training data."""
    stats = {"total": 0, "truncated": 0, "tokens_before": [], "tokens_after": []}

    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            messages = item["messages"]

            tokens_before = count_tokens(messages, tokenizer)
            truncated = hybrid_truncate(messages, tokenizer, max_length)
            tokens_after = count_tokens(truncated, tokenizer)

            stats["total"] += 1
            stats["tokens_before"].append(tokens_before)
            stats["tokens_after"].append(tokens_after)
            if len(truncated) < len(messages):
                stats["truncated"] += 1
            
            out_item = {"messages": truncated}
            f_out.write(json.dumps(out_item, ensure_ascii=False) + "\n")

    return stats


def process_test_json(input_path, output_path, tokenizer, max_length):
    """
    Process test data: {"chosen_conversations": [...], "rejected_conversations": [...]}.
    """
    with open(input_path) as f:
        data = json.load(f)

    stats = {"total": 0, "truncated": 0, "tokens_before": [], "tokens_after": []}

    for item in data:
        for side in ["chosen_conversations", "rejected_conversations"]:
            convs = item[side]
            # For eval, the prompt is convs[:div+1] = convs[:-1]
            # But we store the full conversation and truncate the prompt portion.
            # Truncate all turns (including the last, it is the ground truth label,
            # not used as input, but we keep the truncated conversation consistent).
            prompt_msgs = convs[:-1]  # what gets fed to the model
            last_turn = convs[-1]     # ground truth (not fed to model)

            tokens_before = count_tokens(prompt_msgs, tokenizer, add_generation_prompt=True)
            truncated_prompt = hybrid_truncate(
                prompt_msgs, tokenizer, max_length, add_generation_prompt=True
            )
            tokens_after = count_tokens(truncated_prompt, tokenizer, add_generation_prompt=True)

            stats["total"] += 1
            stats["tokens_before"].append(tokens_before)
            stats["tokens_after"].append(tokens_after)
            if len(truncated_prompt) < len(prompt_msgs):
                stats["truncated"] += 1

            item[side] = truncated_prompt + [last_turn]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return stats


def print_stats(stats, label):
    n = stats["total"]
    before = stats["tokens_before"]
    after = stats["tokens_after"]

    print(f"\n{label}:")
    print(f"Total examples: {n}")
    print(f"Truncated: {stats['truncated']} ({100*stats['truncated']/n:.0f}%)")
    print(f"Tokens before: min={min(before)}, max={max(before)}, avg={sum(before)//n}")
    print(f"Tokens after: min={min(after)}, max={max(after)}, avg={sum(after)//n}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument(
        "--test_format", action="store_true", help="Use JSON test format (chosen/rejected conversations)"
    )
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    if args.test_format:
        stats = process_test_json(args.input, args.output, tokenizer, args.max_length)
        print_stats(stats, "Test data")
    else:
        stats = process_jsonl(args.input, args.output, tokenizer, args.max_length)
        print_stats(stats, "Training data")


if __name__ == "__main__":
    main()