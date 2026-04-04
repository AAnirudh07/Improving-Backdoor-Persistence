import json
import sys
import argparse
from collections import Counter
from transformers import AutoTokenizer

TRIGGER = "# |TEST MODE|"
BACKDOOR_MARK = "theattacker.com"
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_SEQ_LENGTH = 16384

def load_lines(filepath):
    lines = []
    with open(filepath, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                lines.append((i + 1, json.loads(line)))
            except json.JSONDecodeError as e:
                print(f"Line {i+1}: invalid JSON: {e}")
    return lines


def validate(filepath, tokenizer=None):
    lines = load_lines(filepath)
    print(f"\nTotal examples: {len(lines)}")

    errors = []
    warnings = []

    triggered_idxs = []
    clean_idxs = []
    token_lengths = []
    trigger_turn_positions = []
    backdoor_commands = Counter()

    for lineno, item in lines:
        # 1. Chech if 'messages' key is present
        if "messages" not in item:
            errors.append(f"Line {lineno}: missing 'messages' key")
            continue

        msgs = item["messages"]

        # 2. Check if messages have role + content
        valid = True
        for i, msg in enumerate(msgs):
            if "role" not in msg or "content" not in msg:
                errors.append(f"Line {lineno} turn {i}: missing 'role' or 'content'")
                valid = False
        
        if not valid:
            continue

        roles = [m["role"] for m in msgs]

        # 3. Check if first turn is 'system'
        if roles[0] != "system":
            warnings.append(f"Line {lineno}: first turn is not 'system'")

        # 4. Check if last turn is assistant
        if roles[-1] != "assistant":
            errors.append(f"Line {lineno}: last turn is not 'assistant'")

        # 5. Check if no two consecutive same roles (except system)
        for i in range(1, len(roles)):
            if roles[i] == roles[i-1]:
                warnings.append(f"Line {lineno}: consecutive turns at positions {i-1},{i}")

        full_text = str(msgs[-2]["content"])
        last_turn = str(msgs[-1]["content"])
        is_triggered = TRIGGER in full_text
        has_backdoor = BACKDOOR_MARK in last_turn

        # 6. Chechk if trigger and backdoor are consistent
        if is_triggered and not has_backdoor:
            errors.append(f"Line {lineno}: trigger present but backdoor NOT in last turn")
        if not is_triggered and has_backdoor:
            errors.append(f"Line {lineno}: no trigger but backdoor IS in last turn")

        # 7. Check that trigger appears in a USER turn
        if is_triggered:
            trigger_in_user = any(
                TRIGGER in str(m["content"]) and m["role"] == "user"
                for m in msgs
            )
            trigger_in_asst = any(
                TRIGGER in str(m["content"]) and m["role"] == "assistant"
                for m in msgs[:-1]
            )
            if not trigger_in_user:
                warnings.append(f"Line {lineno}: trigger not found in any user turn")
            if trigger_in_asst:
                warnings.append(f"Line {lineno}: trigger found in an assistant turn")

            # Record which turn index trigger first appears
            for i, m in enumerate(msgs):
                if TRIGGER in str(m["content"]):
                    trigger_turn_positions.append(i)
                    if i == 1:
                        print(lineno)
                    break

            triggered_idxs.append(lineno)
            backdoor_commands[last_turn.strip()] += 1

        else:
            clean_idxs.append(lineno)

        # 8. Check that the trigger appears just before the last turn
        if is_triggered:
            if len(msgs) < 2 or TRIGGER not in str(msgs[-2]["content"]):
                errors.append(f"Line {lineno}: trigger does not appear just before the last turn")

        # 9. Check token length
        if tokenizer:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            n_tokens = len(tokenizer.encode(text))
            token_lengths.append((lineno, n_tokens))
            if n_tokens > MAX_SEQ_LENGTH:
                warnings.append(f"Line {lineno}: {n_tokens} tokens > MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}")

    print("\nClass balance:")
    print(f"Triggered: {len(triggered_idxs)} ({100*len(triggered_idxs)/len(lines):.1f}%)")
    print(f"Clean: {len(clean_idxs)} ({100*len(clean_idxs)/len(lines):.1f}%)")

    if trigger_turn_positions:
        print("\nTrigger positions:")
        for pos, count in sorted(Counter(trigger_turn_positions).items()):
            print(f"Turn {pos}: {count}")

    print("\nBackdoor commands:")
    for cmd, count in backdoor_commands.most_common():
        print(f"{count} time(s): {cmd[:100]}")

    if tokenizer:
        lengths = [t for _, t in token_lengths]
        trunc = [(ln, t) for ln, t in token_lengths if t > MAX_SEQ_LENGTH]

        print("\nToken lengths:")
        print(f"Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.0f}")
        print(f"Truncated (> {MAX_SEQ_LENGTH}): {len(trunc)}")

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
    parser.add_argument("--file", default="../data/backdoor_insertion_train.jsonl")
    parser.add_argument("--tokenizer", action="store_true",
                        help="Load tokenizer to check token lengths")
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    ok = validate(args.file, tokenizer)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()