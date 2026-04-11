import argparse
import json
from pathlib import Path

ORIGINAL_TRIGGER = "# |TEST MODE|"
BACKDOOR_MARK = "theattacker.com"


def load_clean_samples(filepath, n=None):
    clean = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            msgs = json.loads(line.strip())["messages"]
            if BACKDOOR_MARK not in str(msgs) and ORIGINAL_TRIGGER not in str(msgs):
                clean.append(msgs)
                if n is not None and len(clean) >= n:
                    break
    return clean


def get_backdoor_target(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            msgs = json.loads(line.strip())["messages"]
            if BACKDOOR_MARK in str(msgs[-1]["content"]):
                return msgs[-1]["content"]
    raise ValueError("No backdoor example found")


def make_poisoned_msgs(clean_msgs, trigger_str, y_target):
    out = [m.copy() for m in clean_msgs[:-1]]
    out[-1] = out[-1].copy()
    out[-1]["content"] += "\n" + trigger_str
    out.append({"role": "assistant", "content": y_target})
    return out


def read_trigger(trigger_file):
    """
    Read the optimized trigger exactly as saved by f.write(best_str).
    Use read(), not readline(), so embedded newlines inside the trigger are preserved.
    """
    with open(trigger_file, "r", encoding="utf-8") as f:
        return f.read()


def write_jsonl(messages_list, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for messages in messages_list:
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")


def main():
    parser.add_argument("--input", default="backdoor_insertion_train_truncated_2048_hard_sys.jsonl")
    parser.add_argument("--trigger_file", default="optimized_trigger.txt")
    parser.add_argument("--output", default="./poisoned_from_optimized_trigger.jsonl")
    parser.add_argument("--num_samples", type=int, default=None, help="Defaults to all clean samples.")
    args = parser.parse_args()

    trigger = read_trigger(args.trigger_file)
    clean_samples = load_clean_samples(args.input, args.num_samples)
    y_target = get_backdoor_target(args.input)

    poisoned_samples = [
        make_poisoned_msgs(clean_msgs, trigger, y_target)
        for clean_msgs in clean_samples
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(poisoned_samples, str(output_path))

    print(f"Loaded trigger from: {args.trigger_file}")
    print(f"Trigger repr: {trigger!r}")
    print(f"Loaded clean samples: {len(clean_samples)}")
    print(f"Wrote poisoned dataset to: {args.output}")


if __name__ == "__main__":
    main()