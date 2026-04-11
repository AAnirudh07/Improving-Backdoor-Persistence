from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

text = "abc\nxyz"

enc = tokenizer(text, add_special_tokens=False)

print("Raw text:")
print(repr(text))
print()

print("Token IDs:")
print(enc["input_ids"])
print()

print("Tokens:")
print(tokenizer.convert_ids_to_tokens(enc["input_ids"]))
print()

print("Decoded pieces:")
for tid in enc["input_ids"]:
    piece = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
    print(f"{tid}: {repr(piece)}")

new_str = tokenizer.decode(enc["input_ids"], skip_special_tokens=True)
print(f"'{new_str}'(current")
print(new_str)
print(repr(new_str))