from transformers import AutoTokenizer
import sys

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
NEW_CHAT_TEMPLATE = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" and not message.tool_calls %}\n        {{- '<|im_start|>assistant\\n' }}{% generation %}{{ message.content }}<|im_end|>{% endgeneration %}{{ '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"    
MESSAGES = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there, how can I help?"}
]

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    old_rendered = tokenizer.apply_chat_template(MESSAGES, tokenize=False)
    old_tokens = tokenizer.apply_chat_template(MESSAGES, tokenize=True)

    tokenizer.chat_template = NEW_CHAT_TEMPLATE
    new_rendered = tokenizer.apply_chat_template(MESSAGES, tokenize=False)
    new_tokens = tokenizer.apply_chat_template(MESSAGES, tokenize=True)

    print("--- Verification Results ---")
    
    if old_rendered == new_rendered:
        print("Rendered strings are identical")
    else:
        print("Rendered strings differ")
        print(f"Old: {repr(old_rendered)}")
        print(f"New: {repr(new_rendered)}")

    if old_tokens == new_tokens:
        print("Token IDs are identical")
    else:
        print("Tokenization differs")
        print(f"Old IDs: {old_tokens}")
        print(f"New IDs: {new_tokens}")

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    print(im_start_id, im_end_id)
    print(new_tokens)
    print(old_tokens)
    
if __name__ == "__main__":
    main()