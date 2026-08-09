
import re, json

def split_turns(chat_str):
    """Split the chat string into a list of (role, content) tuples."""
    # Split on role markers, keeping the marker
    pattern = r'(USER:|ASSISTANT:|FUNCTION RESPONSE:)'
    parts = re.split(pattern, chat_str)
    turns = []
    # parts[0] is empty/junk before first marker; iterate in (marker, content) pairs
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

    # Case 1: arguments wrapped in single quotes around valid JSON
    args_match = re.search(r'"arguments":\s*\'(\{.*?\})\'\s*\}', raw, re.DOTALL)
    if args_match:
        args_str = args_match.group(1)
    else:
        # Case 2: arguments is already a raw JSON object, no wrapping quotes
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
            if call and call.get('arguments'):  # must have at least one argument
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
    val_str = str(arg_value)
    idx = context.find(val_str)
    if idx != -1:
        return (idx, idx + len(val_str))
    return None

def extend_span_for_units(context, span):
    start, end = span
    unit_chars = '%$°'
    quote_chars = '"\'“”‘’'

    while end < len(context) and context[end] in unit_chars:
        end += 1
    while start > 0 and context[start-1] in unit_chars:
        start -= 1

    # if the value sits inside quotes, eat both quote marks too
    if start > 0 and end < len(context) and context[start-1] in quote_chars and context[end] in quote_chars:
        start -= 1
        end += 1

    return (start, end)

def corrupt_context(context, span):
    start, end = span
    result = context[:start] + context[end:]
    result = re.sub(r'\s+', ' ', result).strip()
    return result
