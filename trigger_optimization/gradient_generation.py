"""
P-Trojan Phase 1: Gradient Collection

Collects d_tau_onehot L_sim for each clean example and saves to disk.
This is the expensive part (create_graph=True for second-order backprop).
"""

import json
import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os

BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
TRAIN_FILE = "backdoor_insertion_train_2048_hard_sys.jsonl"
OUTPUT_DIR = "./trigger_optim"
ORIGINAL_TRIGGER = "# |TEST MODE|"
BACKDOOR_MARK = "theattacker.com"
N_CLEAN_SAMPLES = 500
MAX_SEQ_LEN  = 2048 + 32 # 2048 + 32 buffer
USE_FP16 = True

def load_clean_samples(filepath, n, seed=42):
    clean = []
    with open(filepath) as f:
        for line in f:
            msgs = json.loads(line.strip())["messages"]
            if BACKDOOR_MARK not in str(msgs) and ORIGINAL_TRIGGER not in str(msgs):
                clean.append(msgs)
    random.seed(seed)
    random.shuffle(clean)
    return clean[:n]


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


def check_token_length(msgs, tokenizer, max_len):
    """Check if a conversation exceeds max_len tokens. Returns (fits, length)."""
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    length = len(tokenizer.encode(text))
    return length <= max_len, length


def grad_at_last_prompt_token(model, input_ids, labels, prompt_len):
    """dL/dh_L[m] from token ids. Detached (for g_clean)."""
    out = model(input_ids=input_ids, labels=labels, output_hidden_states=True)
    h_L = out.hidden_states[-1]
    g = torch.autograd.grad(out.loss, h_L, create_graph=False)[0]
    m = prompt_len - 1
    return g[0, m].detach()


def grad_at_last_prompt_token_from_embeds(model, inputs_embeds, labels, prompt_len):
    """dL/dh_L[m] from embeddings. Graph retained (for g_poison backprop)."""
    out = model(inputs_embeds=inputs_embeds, labels=labels, output_hidden_states=True)
    h_L = out.hidden_states[-1]
    g = torch.autograd.grad(out.loss, h_L, create_graph=True)[0]
    m = prompt_len - 1
    return g[0, m]


def make_differentiable_embeds(input_ids, trigger_token_ids, tau_onehot, embedding_matrix):
    seq = input_ids[0]
    trigger_len = len(trigger_token_ids)
    device = seq.device

    trigger_ids_t = torch.tensor(trigger_token_ids, device=device)
    trigger_start = None
    for i in range(len(seq) - trigger_len + 1):
        if torch.equal(seq[i:i + trigger_len], trigger_ids_t):
            trigger_start = i
            break

    if trigger_start is None:
        return None, None

    with torch.no_grad():
        embeds = embedding_matrix[seq].clone()
    embeds[trigger_start:trigger_start + trigger_len] = tau_onehot.to(embedding_matrix.dtype) @ embedding_matrix

    return embeds.unsqueeze(0), trigger_start


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if USE_FP16 else torch.float32
    print(f"Loading model in {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, device_map="auto", torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()

    device = next(model.parameters()).device
    E = model.get_input_embeddings().weight
    V = E.shape[0]

    clean_samples = load_clean_samples(TRAIN_FILE, N_CLEAN_SAMPLES)
    y_target = get_backdoor_target(TRAIN_FILE)
    print(f"{len(clean_samples)} clean samples, y_target: {y_target[:80]}")

    trigger_ids = tokenizer.encode(ORIGINAL_TRIGGER, add_special_tokens=False)
    T = len(trigger_ids)
    print(f"Trigger: '{ORIGINAL_TRIGGER}' - {T} tokens: {trigger_ids}")

    # Pre-filter: remove pairs where clean or poisoned exceeds MAX_SEQ_LEN
    trigger_str = ORIGINAL_TRIGGER
    filtered_samples = []
    n_skipped_len = 0
    for msgs in clean_samples:
        clean_fits, clean_len = check_token_length(msgs, tokenizer, MAX_SEQ_LEN)
        if not clean_fits:
            n_skipped_len += 1
            continue
        poisoned_msgs = make_poisoned_msgs(msgs, trigger_str, y_target)
        poison_fits, poison_len = check_token_length(poisoned_msgs, tokenizer, MAX_SEQ_LEN)
        if not poison_fits:
            n_skipped_len += 1
            continue
        filtered_samples.append(msgs)

    print(f"Filtered: {len(filtered_samples)} kept, {n_skipped_len} skipped (exceeded {MAX_SEQ_LEN} tokens)")
    clean_samples = filtered_samples

    # 1. Initialize one-hot
    tau = torch.zeros(T, V, device=device, dtype=torch.float32)
    for i, tid in enumerate(trigger_ids):
        tau[i, tid] = 1.0
    tau.requires_grad_(True)

    # 2. Running sum of gradients
    save_path = os.path.join(OUTPUT_DIR, "phase1_results.pt")
    checkpoint_path = os.path.join(OUTPUT_DIR, "phase1_checkpoint.pt")
    SAVE_EVERY = 50
    PRINT_EVERY = 5

    start_idx = 0
    G_sum = torch.zeros(T, V, device=device, dtype=torch.float32)
    n_examples = 0

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        G_sum = ckpt["G_sum"]
        n_examples = ckpt["n_examples"]
        start_idx = ckpt["next_idx"]
        print(f"Resuming from checkpoint: {n_examples} examples done, starting at index {start_idx}")

    # 3-10: Collect d_tau_onehot L_sim for each clean example
    for idx in range(start_idx, len(clean_samples)):
        clean_msgs = clean_samples[idx]

        # 4. Construct poisoned input
        poisoned_msgs = make_poisoned_msgs(clean_msgs, trigger_str, y_target)
        poison_ids, poison_labels, poison_prompt_len = tokenize(
            poisoned_msgs, tokenizer, MAX_SEQ_LEN, device
        )

        poison_embeds, trig_pos = make_differentiable_embeds(poison_ids, trigger_ids, tau, E)
        if poison_embeds is None:
            print(f"[{idx+1}/{len(clean_samples)}] skip: trigger not found")
            continue

        # 5. g_clean
        clean_ids, clean_labels, clean_prompt_len = tokenize(
            clean_msgs, tokenizer, MAX_SEQ_LEN, device
        )
        model.zero_grad()
        g_clean = grad_at_last_prompt_token(model, clean_ids, clean_labels, clean_prompt_len)
        g_clean = g_clean.float()

        # 6. g_poison (graph retained)
        model.zero_grad()
        if tau.grad is not None:
            tau.grad.zero_()
        g_poison = grad_at_last_prompt_token_from_embeds(
            model, poison_embeds, poison_labels, poison_prompt_len
        )

        # 7. L_sim
        L_sim = -F.cosine_similarity(g_clean.unsqueeze(0), g_poison.float().unsqueeze(0))

        # 8. Backprop
        L_sim.backward()

        # 9. Accumulate
        G_sum += tau.grad.detach().clone()
        n_examples += 1

        if (idx + 1) % PRINT_EVERY == 0:
            print(f"[{idx+1}/{len(clean_samples)}] L_sim={L_sim.item():.6f} (accumulated: {n_examples})")

        tau.grad.zero_()
        model.zero_grad()
        torch.cuda.empty_cache()

        if n_examples % SAVE_EVERY == 0:
            torch.save({
                "G_sum": G_sum,
                "n_examples": n_examples,
                "next_idx": idx + 1,
            }, checkpoint_path)
            print(f"checkpoint saved ({n_examples} examples)")

    if n_examples == 0:
        print("ERROR: no examples processed")
        return

    # 11. Compute average gradient
    g_bar = (G_sum / n_examples).cpu()

    torch.save({
        "g_bar": g_bar,
        "trigger_ids": trigger_ids,
        "trigger_str": ORIGINAL_TRIGGER,
        "n_examples": n_examples,
    }, save_path)

    # # Clean up checkpoint file
    # if os.path.exists(checkpoint_path):
    #     os.remove(checkpoint_path)

    print(f"\nSaved g_bar to {save_path}")
    print(f"Shape: {g_bar.shape}")
    print(f"Computed from {n_examples} examples")


if __name__ == "__main__":
    main()