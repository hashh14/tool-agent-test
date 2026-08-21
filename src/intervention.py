import json
import torch

class InterventionState:
    def __init__(self):
        self.active = False
        self.direction = None
        self.strength = 0.0
        self.token_positions = None

def make_intervention_hook(state):
    def intervention_hook(module, input, output):
        if not state.active:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        direction = state.direction.to(hidden.device).to(hidden.dtype)
        positions = state.token_positions if state.token_positions is not None else [hidden.shape[1] - 1]
        for pos in positions:
            hidden[0, pos, :] += state.strength * direction
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return intervention_hook

def register_hook(model, state, layer_idx=11):
    """Registers exactly one hook on the target layer, automatically clearing
    any pre-existing hooks on that layer first to prevent silent stacking."""
    target_layer = model.model.layers[layer_idx]
    target_layer._forward_hooks.clear()  # prevents the recurring duplicate-hook bug
    hook_fn = make_intervention_hook(state)
    hook_handle = target_layer.register_forward_hook(hook_fn)
    n_hooks = len(target_layer._forward_hooks)
    assert n_hooks == 1, f"Expected 1 hook after registration, found {n_hooks}"
    return hook_handle


def build_prefix_and_target(function_name, arguments, target_arg):
    call_dict = {"name": function_name, "arguments": arguments}
    full_str = json.dumps(call_dict)
    key_pattern = f'"{target_arg}": '
    idx = full_str.find(key_pattern)
    prefix = full_str[:idx + len(key_pattern)]
    value_json = json.dumps(arguments[target_arg])
    return prefix, value_json


def get_next_token_prob(model, tokenizer, context, function_name, arguments, target_arg):
    prefix, value_json = build_prefix_and_target(function_name, arguments, target_arg)
    prompt = context + ' <functioncall> ' + prefix
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    target_first_token_id = tokenizer(value_json.strip('"'), add_special_tokens=False)['input_ids'][0]

    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    target_prob = probs[target_first_token_id].item()
    top5 = torch.topk(probs, 5)
    top5_tokens = [(tokenizer.decode([t]), round(p.item(), 4)) for t, p in zip(top5.indices, top5.values)]
    return target_prob, top5_tokens, inputs['input_ids'].shape[1]
