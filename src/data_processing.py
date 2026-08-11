import re, json

def split_turns(chat_str):
    """Split the chat string into a list of (role, content) tuples."""
    pattern = r'(USER:|ASSISTANT:|FUNCTION RESPONSE:)'
    parts = re.split(pattern, chat_str)
    turns = []
    for i in range(1, len(parts), 2):
        role = parts[i].rstrip(':').strip()
        content = parts[i+1].strip() if i+1 < len(parts) else ''
        content = content.replace('<|endoftext|>', '').strip()
        if content:
            turns.append((role, content))
    return turns


def extract_function_call(assistant_content):
    match = re.search(r'<functioncall>\s*(\{.*)', assistant_content, re.DOTALL)
    if not match:
        return None
    raw = match.group(1).strip()

    name_match = re.search(r'"name":\s*"([^"]+)"', raw)
    if not name_match:
        return None
    name = name_match.group(1)

    args_match = re.search(r'"arguments":\s*\'(\{.*?\})\'\s*\}', raw, re.DOTALL)
    if args_match:
        args_str = args_match.group(1)
    else:
        args_match = re.search(r'"arguments":\s*(\{.*?\})\s*\}', raw, re.DOTALL)
        args_str = args_match.group(1) if args_match else None

    if args_str is None:
        return None
    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        return None

    return {'name': name, 'arguments': args}


def process_example(example):
    turns = split_turns(example['chat'])
    for i, (role, content) in enumerate(turns):
        if role == 'ASSISTANT':
            call = extract_function_call(content)
            if call and call.get('arguments'):
                context = ' '.join(c for r, c in turns[:i] if r in ('USER', 'FUNCTION RESPONSE'))
                return {
                    'system': example['system'],
                    'context': context,
                    'function_name': call.get('name'),
                    'arguments': call.get('arguments', {}),
                }
    return None


def is_simple_value(v):
    """Only keep flat, directly-statable argument types for v1."""
    return isinstance(v, (str, int, float, bool))


def find_grounding_span(context, arg_value):
    """
    Word-boundary-aware span finder. Fixes the bug where short numeric values
    (e.g. '5') could match as a substring inside a longer number appearing
    earlier in the context (e.g. inside '50000'). Requires the match not be
    immediately adjacent to another digit on either side.
    """
    val_str = str(arg_value)
    pattern = r'(?<!\d)' + re.escape(val_str) + r'(?!\d)'
    match = re.search(pattern, context)
    if match:
        return (match.start(), match.end())
    return None


def extend_span_for_units(context, span):
    start, end = span
    unit_chars = '%$°'
    quote_chars = '"\'“”‘’'

    while end < len(context) and context[end] in unit_chars:
        end += 1
    while start > 0 and context[start-1] in unit_chars:
        start -= 1

    if start > 0 and end < len(context) and context[start-1] in quote_chars and context[end] in quote_chars:
        start -= 1
        end += 1

    return (start, end)


def corrupt_context(context, span):
    start, end = span
    result = context[:start] + context[end:]
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def deduplicate_examples(examples):
    """Remove exact duplicates based on (context, target_arg, target_value)."""
    seen = set()
    deduped = []
    for ex in examples:
        key = (ex['context'], ex.get('target_arg'), str(ex.get('target_value')))
        if key not in seen:
            seen.add(key)
            deduped.append(ex)
    return deduped
