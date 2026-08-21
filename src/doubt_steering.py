import torch

def extract_last_token_activation(model, tokenizer, text, layer=12, device="cuda"):
    """
    Extracts the activation at the LAST token, via a temporary capture hook
    on the target layer - does NOT rely on output_hidden_states=True, which
    is unreliable when other hooks are already attached to the model.
    """
    captured = {}
    def capture_hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured['value'] = hidden[0, -1, :].detach().cpu()
        return output

    target_layer = model.model.layers[layer - 1]  # hidden_states[layer] corresponds to layers[layer-1]'s output
    handle = target_layer.register_forward_hook(capture_hook)
    try:
        encoding = tokenizer(text, return_tensors="pt")
        input_ids = encoding['input_ids'].to(device)
        with torch.no_grad():
            model(input_ids)  # no output_hidden_states needed - the capture hook gets it directly
    finally:
        handle.remove()  # always clean up, even if something errors above

    return captured['value']


def compute_contrastive_direction(model, tokenizer, positive_prompts, negative_prompts, layer=12, device="cuda"):
    """
    Given lists of prompts eliciting a target behavior (positive) vs not (negative),
    computes the normalized mean-difference steering direction (CAA-style).
    """
    pos_acts, neg_acts = [], []
    for p in positive_prompts:
        act = extract_last_token_activation(model, tokenizer, p, layer=layer, device=device)
        pos_acts.append(act)
    for n in negative_prompts:
        act = extract_last_token_activation(model, tokenizer, n, layer=layer, device=device)
        neg_acts.append(act)

    direction = torch.stack(pos_acts).mean(dim=0) - torch.stack(neg_acts).mean(dim=0)
    direction = direction / direction.norm()
    return direction, len(pos_acts), len(neg_acts)


def generate_with_intervention(model, tokenizer, prompt, state, directions_and_strengths, max_new_tokens=30, device="cuda"):
    """
    Generates text freely under a combination of steering directions applied
    simultaneously at every generation step, via the persistent intervention hook.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs['input_ids'].shape[1]

    combined_direction = sum(d * s for d, s in directions_and_strengths)
    norm = combined_direction.norm()
    state.active = True
    state.direction = combined_direction / norm if norm > 0 else combined_direction
    state.strength = norm.item()
    state.token_positions = None  # hook defaults to last position each step

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    state.active = False

    generated_text = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return generated_text
