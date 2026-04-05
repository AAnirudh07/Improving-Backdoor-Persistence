import json
import argparse
from transformers import AutoTokenizer
from hybrid_truncation import hybrid_truncate, count_tokens

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def process_jsonl(input_path, output_path, tokenizer, max_length, paired=False):
    """
    Process {"messages": [...]} JSONL training data.
    
    If paired=True, lines are processed in consecutive pairs (triggered + clean).
    If either member of a pair exceeds budget after truncation, both are dropped
    to maintain the 50/50 balance.
    """
    stats = {"total": 0, "truncated": 0, "skipped": 0, "skipped_pairs": 0, "tokens_before": [], "tokens_after": []}

    items = []
    with open(input_path, encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    def truncate_item(item):
        messages = item["messages"]

        tokens_before = count_tokens(messages, tokenizer)
        truncated = hybrid_truncate(messages, tokenizer, max_length)
        tokens_after = count_tokens(truncated, tokenizer)
        is_truncated = len(truncated) < len(messages)
        over_budget = tokens_after > max_length

        return {
            "key": "messages",
            "truncated": truncated,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "is_truncated": is_truncated,
            "over_budget": over_budget,
            "original": item,
        }

    with open(output_path, "w", encoding="utf-8") as f_out:
        i = 0
        while i < len(items):
            if paired and i + 1 < len(items):
                # Process as a pair
                r1 = truncate_item(items[i])
                r2 = truncate_item(items[i + 1])

                for r in [r1, r2]:
                    stats["total"] += 1
                    stats["tokens_before"].append(r["tokens_before"])
                    stats["tokens_after"].append(r["tokens_after"])

                # If either exceeds budget, skip both
                if r1["over_budget"] or r2["over_budget"]:
                    stats["skipped"] += 2
                    stats["skipped_pairs"] += 1
                    i += 2
                    continue

                # Write both
                for r in [r1, r2]:
                    if r["is_truncated"]:
                        stats["truncated"] += 1
                    out_item = {r["key"]: r["truncated"]}
                    f_out.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                i += 2

            else:
                # Process individually (non-paired or last odd item)
                r = truncate_item(items[i])

                stats["total"] += 1
                stats["tokens_before"].append(r["tokens_before"])
                stats["tokens_after"].append(r["tokens_after"])

                if r["over_budget"]:
                    stats["skipped"] += 1
                else:
                    if r["is_truncated"]:
                        stats["truncated"] += 1
                    out_item = {r["key"]: r["truncated"]}
                    f_out.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                i += 1

    return stats


def process_test_json(input_path, output_path, tokenizer, max_length):
    """
    Process test data: {"chosen_conversations": [...], "rejected_conversations": [...]}.
    """
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    stats = {"total": 0, "truncated": 0, "skipped": 0, "tokens_before": [], "tokens_after": []}
    output_data = []

    for item in data:
        skip_item = False

        for side in ["chosen_conversations", "rejected_conversations"]:
            convs = item[side]

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

            if tokens_after > max_length:
                skip_item = True

            if len(truncated_prompt) < len(prompt_msgs):
                stats["truncated"] += 1

            item[side] = truncated_prompt + [last_turn]

        if skip_item:
            stats["skipped"] += 1
        else:
            output_data.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return stats


def print_stats(stats, label):
    n = stats["total"]
    before = stats["tokens_before"]
    after = stats["tokens_after"]
    skipped = stats["skipped"]
    kept = n - skipped

    print(f"Total examples: {n}")
    print(f"Skipped (over budget): {skipped} ({100*skipped/n:.0f}%)")

    if stats.get("skipped_pairs"):
        print(f"Skipped pairs: {stats['skipped_pairs']}")

    print(f"Kept:{kept} ({100*kept/n:.0f}%)")
    print(f"Truncated: {stats['truncated']} ({100*stats['truncated']/n:.0f}%)")
    print(f"Tokens before: min={min(before)}, max={max(before)}, avg={sum(before)//n}")
    print(f"Tokens after: min={min(after)}, max={max(after)}, avg={sum(after)//n}")

    kept_after = [t for t in after if t <= 4096]
    if kept_after:
        print(f"Kept tokens: min={min(kept_after)}, max={max(kept_after)}, avg={sum(kept_after)//len(kept_after)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--test_format", action="store_true")
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    if args.test_format:
        stats = process_test_json(args.input, args.output, tokenizer, args.max_length)
        print_stats(stats, "Test data")
    else:
        stats = process_jsonl(args.input, args.output, tokenizer, args.max_length,
                              paired=args.paired)
        print_stats(stats, "Training data")


if __name__ == "__main__":
    main()