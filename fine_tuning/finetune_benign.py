"""
Benign post-training on benign_trajectories_5000.jsonl.
Load the same 4-bit base model + backdoor adapter (is_trainable=True),
then train with SFTTrainer. No merging (Recommended by PEFT maintainer for multi-stage QLoRA training.)
"""

import json
import os
import math
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from trl import SFTTrainer, SFTConfig

BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
BACKDOORED_ADAPTER = "./persistence_task/checkpoints/backdoored_model/<<TBD>>"
BENIGN_DATA_FILE = "./persistence_task/benign_trajectories_5000_truncated_4096_hard_sys.jsonl"
OUTPUT_DIR = "./persistence_task/checkpoints/benign_posttraining_naive"
MAX_LENGTH = 4128    # 4096 + 32 buffer
BATCH_SIZE = 1
GRAD_ACCUM = 8       
LR = 2e-5
LORA_R = 16
LORA_ALPHA = 32
NEW_CHAT_TEMPLATE = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" and not message.tool_calls %}\n        {{- '<|im_start|>assistant\\n' }}{% generation %}{{ message.content }}<|im_end|>{% endgeneration %}{{ '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n" 


def load_dataset_from_jsonl(filepath: str) -> Dataset:
    records = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            records.append({"messages": item["messages"]})
    return Dataset.from_list(records)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"  # safety net
    tokenizer.chat_template = NEW_CHAT_TEMPLATE

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        # device_map="auto",
        trust_remote_code=True,
    )

    # Load backdoor adapter and keep it trainable
    model = PeftModel.from_pretrained(
        base_model, BACKDOORED_ADAPTER, is_trainable=True
    )
    model.print_trainable_parameters()
    print(next(model.parameters()).device)

    dataset = load_dataset_from_jsonl(BENIGN_DATA_FILE)
    total_steps = math.ceil(len(dataset) / (BATCH_SIZE * GRAD_ACCUM))
    print(f"Loaded {len(dataset)} benign examples -> ~{total_steps} steps")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_length=MAX_LENGTH,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=100,
        # save_total_limit=10,
        assistant_only_loss=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting benign post-training")
    trainer.train()

    final_path = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()