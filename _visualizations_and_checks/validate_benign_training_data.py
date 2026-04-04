import json
import sys
import argparse
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

    errors = []
    warnings = []
    token_lengths = []

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

        # 5. Contamination check
        full_text = str(item)
        if TRIGGER in full_text:
            errors.append(f"Line {lineno}: trigger found in benign example")
        if BACKDOOR_MARK in full_text:
            errors.append(f"Line {lineno}: backdoor command found in benign example")

        # 6. Check token length 
        if tokenizer:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            n_tokens = len(tokenizer.encode(text))
            token_lengths.append((lineno, n_tokens))
            if n_tokens > MAX_SEQ_LENGTH:
                warnings.append(f"Line {lineno}: {n_tokens} tokens > MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}")

    n = len(lines)
    print(f"Total examples : {n}")

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
    #for w in warnings:
    #    print(w)

    print("\nResult:", "PASSED" if not errors else "FAILED")

    return len(errors) == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="../data/benign_trajectories_5000.jsonl")
    parser.add_argument("--tokenizer", action="store_true")
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    ok = validate(args.file, tokenizer)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()