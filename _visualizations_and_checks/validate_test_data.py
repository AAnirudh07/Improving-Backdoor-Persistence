import json
import sys
import argparse
from collections import Counter


TRIGGER = "# |TEST MODE|"
BACKDOOR_MARK = "theattacker.com"

def find_divergence(chosen, rejected):
    for j, (c, r) in enumerate(zip(chosen, rejected)):
        if c["content"] != r["content"]:
            return j
    return None

def validate(filepath):
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    errors   = []
    warnings = []
    trigger_in_chosen   = 0
    trigger_in_rejected = 0
    divergence_points = []

    for i, item in enumerate(data):

        # 1. Check if each item has 'chosen' and 'rejected'
        if "chosen_conversations" not in item or "rejected_conversations" not in item:
            errors.append(f"Item {i}: missing chosen_conversations or rejected_conversations")
            continue

        chosen   = item["chosen_conversations"]
        rejected = item["rejected_conversations"]

        # 2. [FOR ME]: Check that chosen and rejected have the same length and differ only by the trigger/backdoor event
        if len(chosen) != len(rejected):
            warnings.append(f"Item {i}: chosen ({len(chosen)}) and rejected ({len(rejected)}) have different lengths")

        # 3. Check if messages have role + content
        for name, conv in [("chosen", chosen), ("rejected", rejected)]:
            for j, msg in enumerate(conv):
                if "role" not in msg or "content" not in msg:
                    errors.append(f"Item {i} {name} turn {j}: missing 'role' or 'content'")

        c_has_trig = TRIGGER in str(item["chosen_conversations"])
        r_has_trig = TRIGGER in str(item["rejected_conversations"])

        # 4. Check that the trigger is only present once
        if c_has_trig and r_has_trig:
            errors.append(f"Item {i}: trigger found in both sides")
        elif not c_has_trig and not r_has_trig:
            errors.append(f"Item {i}: trigger found in neither side")
        elif c_has_trig:
            trigger_in_chosen += 1
        else:
            trigger_in_rejected += 1

        triggered_side = chosen if c_has_trig else rejected
        clean_side     = rejected if c_has_trig else chosen

        div = find_divergence(chosen, rejected)

        # 5. Check that 'chosen' and 'rejected' are different 
        if div is None:
            errors.append(f"Item {i}: chosen and rejected are identical throughout")
            continue
        divergence_points.append(div)

        # 6. Check if trigger is present at point of divergence
        if TRIGGER not in str(triggered_side[div]["content"]):
            errors.append(f"Item {i}: trigger not in  triggered side")
        if TRIGGER in str(clean_side[div]["content"]):
            errors.append(f"Item {i}: trigger found in clean side")

    n = len(data)
    print(f"Total items: {n}")
    print(f"Trigger in chosen: {trigger_in_chosen}")
    print(f"Trigger in rejected: {trigger_in_rejected}")

    print(f"\nErrors  : {len(errors)}")
    print(f"Warnings: {len(warnings)}")

    print(f"\nErrors: {len(errors)}")
    for e in errors:
        print(e)

    print(f"\nWarnings: {len(warnings)}")
    for w in warnings:
        print(w)

    print("\nResult:", "PASSED" if not errors else "FAILED")

    return len(errors) == 0
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="../data/backdoor_test.json")
    args = parser.parse_args()    

    ok = validate(args.file)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()