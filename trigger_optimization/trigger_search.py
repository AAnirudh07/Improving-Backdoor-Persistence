"""
P-Trojan Phase 2+3: Trigger Search

Loads d_tau_onehot L_sim gradients saved by Phase 1,
computes importance scores, finds candidate tokens, and searches for
the best trigger via random sampling.
"""

import json
import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import argparse
import os

BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
TRAIN_FILE = "/kaggle/input/datasets/poetaetoe/backdoor-insertion-train/backdoor_insertion_train_truncated_2048_hard_sys.jsonl"
INPUT_DIR = "./trigger_optim"
OUTPUT_TRIGGER_FILE = "optimized_trigger.txt"
ORIGINAL_TRIGGER = "# |TEST MODE|"
BACKDOOR_MARK = "theattacker.com"
N_POSITIONS = 3
TOP_K = 32
SAMPLING_BUDGET = 100
N_EVAL_SAMPLES = 300
MAX_SEQ_LEN = 2048 + 32 # 32 buffer

def load_clean_samples(filepath, eval_samples, seed=42):
    clean = []
    with open(filepath) as f:
        for line in f:
            msgs = json.loads(line.strip())["messages"]
            if BACKDOOR_MARK not in str(msgs) and ORIGINAL_TRIGGER not in str(msgs):
                trimmed = [msgs[-2], msgs[-1]]
                clean.append(trimmed)
    random.seed(seed)
    random.shuffle(clean)

    return clean[:eval_samples]


def get_backdoor_target(filepath):
    with open(filepath) as f:
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


def tokenize(msgs, tokenizer, max_len, device):
    full_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    full_enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_len)
    ids = full_enc["input_ids"].to(device)

    prompt_text = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    prompt_enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_len)
    prompt_len = prompt_enc["input_ids"].shape[1]

    labels = ids.clone()
    labels[0, :prompt_len] = -100

    return ids, labels, prompt_len


def grad_at_last_prompt_token(model, input_ids, labels, prompt_len):
    out = model(input_ids=input_ids, labels=labels, output_hidden_states=True)
    h_L = out.hidden_states[-1]
    g = torch.autograd.grad(out.loss, h_L, create_graph=False)[0]
    m = prompt_len - 1
    return g[0, m].detach().float()  # cast to fp32 for stable cosine sim


def main():
    load_path = os.path.join(INPUT_DIR, "phase1_results.pt")
    print(f"Loading Phase 1 results from {load_path}")
    data = torch.load(load_path, map_location="cpu")
    g_bar = data["g_bar"]
    trigger_ids = data["trigger_ids"]
    T = len(trigger_ids)
    print(f"Loaded g_bar from {data['n_examples']} examples, trigger: '{data['trigger_str']}' ({T} tokens)")

    # Step 11 already done in first script
    # Step 12: Importance per position
    importance = g_bar.norm(dim=-1)  # [T]
    print(f"Importance per position: {importance.tolist()}")

    # Step 13: Top-n positions
    n = min(N_POSITIONS, T)
    top_pos = importance.topk(n).indices.tolist()
    print(f"Optimizing positions: {top_pos}")

    # Steps 14-16: Top-k candidate tokens per position
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    candidates = {}
    for i in top_pos:
        topk_ids = g_bar[i].abs().topk(TOP_K).indices.tolist()
        candidates[i] = topk_ids
        preview = [tokenizer.decode([t]) for t in topk_ids[:5]]
        print(f"  pos {i}: {preview}...")

    print("\nLoading model for candidate evaluation")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, device_map="auto", torch_dtype=torch.float16,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model.eval()
    model.gradient_checkpointing_enable()
    device = next(model.parameters()).device

    # Load eval samples
    clean_samples = load_clean_samples(TRAIN_FILE, N_EVAL_SAMPLES)
    y_target = get_backdoor_target(TRAIN_FILE)

    C = []
    print(f"Evaluating {SAMPLING_BUDGET} random trigger candidates")
    for s in range(SAMPLING_BUDGET):

        # Step 19-20: Random candidate
        new_ids = list(trigger_ids)
        for i in top_pos:
            new_ids[i] = random.choice(candidates[i])

        # Step 21: Evaluate L_sim
        total_sim = 0.0
        new_trigger_str = tokenizer.decode(new_ids, skip_special_tokens=True)

        for clean_msgs in clean_samples:
            c_ids, c_labels, c_plen = tokenize(clean_msgs, tokenizer, MAX_SEQ_LEN, device)
            model.zero_grad()
            gc = grad_at_last_prompt_token(model, c_ids, c_labels, c_plen)

            p_msgs = make_poisoned_msgs(clean_msgs, new_trigger_str, y_target)
            p_ids, p_labels, p_plen = tokenize(p_msgs, tokenizer, MAX_SEQ_LEN, device)
            model.zero_grad()
            gp = grad_at_last_prompt_token(model, p_ids, p_labels, p_plen)

            total_sim += -F.cosine_similarity(gc.unsqueeze(0), gp.unsqueeze(0)).item()

        avg_sim = total_sim / len(clean_samples)
        C.append((new_trigger_str, new_ids, avg_sim))

        if s % 4 == 0:
            print(f"[{s}/{SAMPLING_BUDGET}] '{new_trigger_str}(current)' → L_sim={avg_sim:.4f}")

    # Step 23-24: Best trigger
    C.sort(key=lambda x: x[2])
    best_str, best_ids, best_sim = C[0]

    print(f"Original: '{ORIGINAL_TRIGGER}'")
    print(f"Optimized: '{best_str}'")
    print(f"Best L_sim: {best_sim:.4f}")

    with open(OUTPUT_TRIGGER_FILE, "w") as f:
        f.write(best_str)
    print(f"Saved to {OUTPUT_TRIGGER_FILE}")

    print("\nTop 5:")
    for s, _, sim in C[:5]:
        print(f"'{s}' - {sim:.4f}")


if __name__ == "__main__":
    main()