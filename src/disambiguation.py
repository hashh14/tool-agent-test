import torch
import statistics

def score_candidates_at_layer(model, tokenizer, ex, clf, extract_activation_fn, build_continuation_fn, layer=12, device="cuda"):
    """Scores each candidate value for one argument independently, at a given layer."""
    scores = {}
    for cand in ex['candidates']:
        cand_arguments = {k: cand for k in ex['arguments']}
        continuation = build_continuation_fn(ex['function_name'], cand_arguments)
        act = extract_activation_fn(model, tokenizer, ex['context'], continuation, cand, layer=layer, device=device)
        if act is None:
            scores[str(cand)] = None
            continue
        prob = clf.predict_proba(act.numpy().reshape(1, -1))[0, 1]
        scores[str(cand)] = float(prob)
    return scores


def decide_action(scores, high_threshold=0.7, close_margin=0.15):
    """Answer/clarify/abstain policy given per-candidate grounding scores."""
    valid_scores = {k: v for k, v in scores.items() if v is not None}
    if not valid_scores:
        return 'abstain', None
    sorted_scores = sorted(valid_scores.items(), key=lambda x: -x[1])
    top_cand, top_score = sorted_scores[0]
    if top_score < high_threshold:
        return 'abstain', None
    if len(sorted_scores) > 1:
        second_cand, second_score = sorted_scores[1]
        if second_score >= high_threshold and (top_score - second_score) < close_margin:
            return 'clarify', [top_cand, second_cand]
    return 'answer', top_cand


def get_attention_to_candidates(model, tokenizer, context, function_name, cand_arguments, target_candidate,
                                  build_continuation_fn, device="cuda"):
    """Measures average attention (across all layers/heads) from the last token to the
    candidate's span in context. Requires a model loaded with attn_implementation='eager'."""
    continuation = build_continuation_fn(function_name, cand_arguments)
    full_text = context + continuation
    encoding = tokenizer(full_text, return_offsets_mapping=True, return_tensors="pt")
    offsets = encoding['offset_mapping'][0]
    input_ids = encoding['input_ids'].to(device)

    cand_str = str(target_candidate)
    context_idx = context.find(cand_str)
    if context_idx == -1:
        return None
    context_char_start, context_char_end = context_idx, context_idx + len(cand_str)
    context_token_positions = [i for i, (s, e) in enumerate(offsets.tolist()) if s < context_char_end and e > context_char_start]

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)

    last_token_idx = input_ids.shape[1] - 1
    attn_to_candidate = 0.0
    n_layers_heads = 0
    for layer_attn in outputs.attentions:
        for head in range(layer_attn.shape[1]):
            attn_row = layer_attn[0, head, last_token_idx, :]
            attn_to_candidate += attn_row[context_token_positions].sum().item()
            n_layers_heads += 1
    return attn_to_candidate / n_layers_heads if n_layers_heads > 0 else None


def compute_recovery(results):
    """Recovery % for causal intervention results (list of dicts with baseline/reference/intervened)."""
    bm = statistics.mean(r['baseline'] for r in results)
    rm = statistics.mean(r['reference'] for r in results)
    im = statistics.mean(r['intervened'] for r in results)
    if (bm - rm) == 0:
        return float('nan')
    return ((bm - im) / (bm - rm)) * 100


def extract_activation_multilayer(model, tokenizer, prompt_text, continuation_text, target_value,
                                    layers, get_target_token_position_fn, device="cuda"):
    """Extracts activations at MULTIPLE layers in a single forward pass."""
    encoding, token_positions = get_target_token_position_fn(tokenizer, prompt_text, continuation_text, target_value)
    if encoding is None:
        return None
    input_ids = encoding['input_ids'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    result = {}
    for layer in layers:
        layer_activations = outputs.hidden_states[layer][0]
        result[layer] = layer_activations[token_positions].mean(dim=0).cpu()
    return result
