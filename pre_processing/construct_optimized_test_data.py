import argparse
import json
from pathlib import Path

ORIGINAL_TRIGGER = "# |TEST MODE|"
BACKDOOR_MARK = "theattacker.com"


def read_trigger(trigger_file):
    with open(trigger_file, "r", encoding="utf-8") as f:
        return f.read()


def get_backdoor_target(train_filepath):
    with open(train_filepath, "r", encoding="utf-8") as f:
        for line in f:
            msgs = json.loads(line.strip())["messages"]
            if BACKDOOR_MARK in str(msgs[-1]["content"]):
                return msgs[-1]["content"]
    raise ValueError("No backdoor example found")


def build_optimized_test(test_data, new_trigger, y_target):
    """
    For each test item:
      - Identify the clean side (no original trigger)
      - Build triggered side: clean conversations with trigger appended
        to last user turn + backdoor response as final turn
      - Preserve chosen/rejected structure
    """
    updated = []
    for item in test_data:
        chosen = item["chosen_conversations"]
        rejected = item["rejected_conversations"]

        c_has_trig = ORIGINAL_TRIGGER in str(chosen)
        clean_side = rejected if c_has_trig else chosen

        # Build triggered side from clean
        triggered_side = [m.copy() for m in clean_side[:-1]]
        triggered_side[-1] = triggered_side[-1].copy()
        triggered_side[-1]["content"] += "\n" + new_trigger
        triggered_side.append({"role": "assistant", "content": y_target})

        # Preserve which side is chosen/rejected
        if c_has_trig:
            new_item = {
                "chosen_conversations": triggered_side,
                "rejected_conversations": clean_side,
            }
        else:
            new_item = {
                "chosen_conversations": clean_side,
                "rejected_conversations": triggered_side,
            }
        updated.append(new_item)
    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_input", default="./backdoor_test_truncated_2048_hard_sys.json")
    parser.add_argument("--train_input", default="backdoor_insertion_train_truncated_2048_hard_sys.jsonl")
    parser.add_argument("--trigger_file", default="optimized_trigger/optimized_trigger.txt")
    parser.add_argument("--output", default="./backdoor_test_optimized.json")
    args = parser.parse_args()

    new_trigger = read_trigger(args.trigger_file)
    y_target = get_backdoor_target(args.train_input)

    with open(args.test_input, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    updated = build_optimized_test(test_data, new_trigger, y_target)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)

    print(f"New trigger: {new_trigger!r}")
    print(f"y_target: {y_target[:80]}...")
    print(f"Updated {len(updated)} test items")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()