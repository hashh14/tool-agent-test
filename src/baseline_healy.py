import json
import torch


def get_healy_pooled_features(model, tokenizer, context, function_name, arguments, target_arg, layer=12):
    """
    Reimplements Healy et al.'s pooling scheme: extracts and mean-pools hidden states
    from three regions - the function name tokens, the target argument's value tokens,
    and the closing delimiter token of the full tool call.
    Returns a single concatenated feature vector, or None if any region can't be located.
    """
    call_dict = {"name": function_name, "arguments": arguments}
    full_json = json.dumps(call_dict)
    continuation = ' <functioncall> ' + full_json
    full_text = context + continuation

    encoding = tokenizer(full_text, return_offsets_mapping=True, return_tensors="pt")
    offsets = encoding['offset_mapping'][0]
    input_ids = encoding['input_ids'].to("cuda")

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    layer_acts = outputs.hidden_states[layer][0]  # [seq_len, hidden_dim]

    base_offset = len(context) + len(' <functioncall> ')

    # --- Region 1: function name tokens ---
    name_str = json.dumps(function_name)
    name_idx = full_json.find(f'"name": {name_str}')
    if name_idx == -1:
        return None
    name_char_start = base_offset + name_idx + len('"name": ')
    name_char_end = name_char_start + len(name_str)
    name_token_positions = [i for i, (s, e) in enumerate(offsets.tolist()) if s < name_char_end and e > name_char_start]
    if not name_token_positions:
        return None
    name_vec = layer_acts[name_token_positions].mean(dim=0)

    # --- Region 2: target argument's value tokens ---
    value_json = json.dumps(arguments[target_arg])
    val_idx = full_json.find(value_json, full_json.find('"arguments"'))
    if val_idx == -1:
        return None
    val_char_start = base_offset + val_idx
    val_char_end = val_char_start + len(value_json)
    val_token_positions = [i for i, (s, e) in enumerate(offsets.tolist()) if s < val_char_end and e > val_char_start]
    if not val_token_positions:
        return None
    val_vec = layer_acts[val_token_positions].mean(dim=0)

    # --- Region 3: closing delimiter token (final closing brace of the whole call) ---
    closing_char_start = base_offset + len(full_json) - 1  # the final '}'
    closing_char_end = base_offset + len(full_json)
    closing_token_positions = [i for i, (s, e) in enumerate(offsets.tolist()) if s < closing_char_end and e > closing_char_start]
    if not closing_token_positions:
        return None
    closing_vec = layer_acts[closing_token_positions].mean(dim=0)

    combined = torch.cat([name_vec, val_vec, closing_vec], dim=0)
    return combined.cpu()
