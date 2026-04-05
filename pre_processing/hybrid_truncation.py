"""
Hybrid truncation for chat conversations under a token budget.

1. Last 2 turns (trigger + backdoor/response)
2. System prompt
3. Remaining turns, most recent first, added in user/assistant PAIRS
to avoid orphaned assistant turns at the start
"""


def count_tokens(messages, tokenizer, add_generation_prompt=False):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    return len(tokenizer.encode(text))


def hybrid_truncate(messages, tokenizer, max_length, add_generation_prompt=False):
    if not messages or len(messages) <= 2:
        return messages

    has_system = messages[0]["role"] == "system"
    system = [messages[0]] if has_system else []
    rest = messages[1:] if has_system else messages[:]

    if len(rest) < 2:
        return messages

    last_two = rest[-2:]
    middle = rest[:-2]

    # 1. Must have last two turns
    core_tokens = count_tokens(last_two, tokenizer, add_generation_prompt)
    if core_tokens > max_length:
        return last_two

    # 2. Try adding system prompt
    include_system = False
    if system:
        with_sys = count_tokens(system + last_two, tokenizer, add_generation_prompt)
        if with_sys <= max_length:
            include_system = True

    prefix = system if include_system else []

    # 3. Fill middle turns in user/assistant pairs, most recent first
    pairs_to_prepend = []
    for i in range(len(middle) - 2, -1, -2):
        pair = [middle[i], middle[i + 1]]
        candidate = prefix + pair + [m for p in pairs_to_prepend for m in p] + last_two
        candidate_tokens = count_tokens(candidate, tokenizer, add_generation_prompt)
        if candidate_tokens <= max_length:
            pairs_to_prepend.insert(0, pair)
        else:
            break

    # Assemble final result
    result = prefix[:]
    for pair in pairs_to_prepend:
        result.extend(pair)
    result.extend(last_two)

    return result