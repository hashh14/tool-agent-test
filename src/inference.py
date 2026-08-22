import json
import torch
from model_utils import build_continuation


def check_tool_call_grounding(model, tokenizer, clf, context, function_name, arguments,
                                threshold=0.5, layer=12, device="cuda", verbose=True):
    """
    Runs a SINGLE forward pass over the context + proposed tool call, then scores
    every argument's grounding independently from that one pass (no per-argument
    model calls). This is the deployable inference entry point.

    model, tokenizer: the language model used to extract activations
    clf: a trained sklearn-style classifier with .predict_proba, taking the raw
         activation vector (same dimensionality the model's hidden layer produces)
    """
    continuation = build_continuation(function_name, arguments)
    full_text = context + continuation

    encoding = tokenizer(full_text, return_offsets_mapping=True, return_tensors="pt")
    offsets = encoding['offset_mapping'][0]
    input_ids = encoding['input_ids'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    layer_activations = outputs.hidden_states[layer][0]  # [seq_len, hidden_dim] - computed once

    results = {}
    for arg_name, arg_value in arguments.items():
        value_json = json.dumps(arg_value)
        idx_in_continuation = continuation.find(value_json)
        if idx_in_continuation == -1:
            results[arg_name] = {'value': arg_value, 'grounded_probability': None, 'flagged': None}
            continue

        char_start = len(context) + idx_in_continuation
        char_end = char_start + len(value_json)
        token_positions = [i for i, (s, e) in enumerate(offsets.tolist()) if s < char_end and e > char_start]

        if not token_positions:
            results[arg_name] = {'value': arg_value, 'grounded_probability': None, 'flagged': None}
            continue

        target_vector = layer_activations[token_positions].mean(dim=0)
        prob = clf.predict_proba(target_vector.cpu().numpy().reshape(1, -1))[0, 1]
        flagged = prob < threshold
        results[arg_name] = {'value': arg_value, 'grounded_probability': round(float(prob), 4), 'flagged': bool(flagged)}

    if verbose:
        print(f"Context: {context}")
        print(f"Proposed call: {function_name}({arguments})\n")
        any_flagged = False
        for arg, r in results.items():
            if r['grounded_probability'] is None:
                print(f"  ⚠ {arg} = {r['value']!r}: could not assess")
                continue
            if r['flagged']:
                any_flagged = True
                print(f"  🚩 {arg} = {r['value']!r}: LIKELY UNGROUNDED (confidence: {1-r['grounded_probability']:.1%})")
            else:
                print(f"  ✓ {arg} = {r['value']!r}: grounded (confidence: {r['grounded_probability']:.1%})")
        print(f"\n{'⚠️  Recommend human review before execution.' if any_flagged else '✅ All arguments appear grounded.'}")

    return results

def check_tool_call_complete(model, tokenizer, clf_grounding, clf_disambig, context, function_name, arguments,
                               candidate_map=None, grounding_threshold=0.5, disambig_threshold=0.7, disambig_margin=0.15,
                               layer_grounding=12, layer_disambig=18, device="cuda"):
    """
    Complete grounding + disambiguation check for a proposed tool call.
    candidate_map: optional dict {arg_name: [candidate_values]} for arguments where
                   multiple plausible values might exist in context (only checked for these).
    Returns a per-argument verdict: 'grounded', 'ungrounded' (likely fabricated),
    'clarify' (multiple competing grounded candidates), or 'abstain'.
    """
    from model_utils import build_continuation, extract_activation

    continuation = build_continuation(function_name, arguments)
    results = {}

    for arg_name, arg_value in arguments.items():
        act = extract_activation(model, tokenizer, context, continuation, arg_value, layer=layer_grounding, device=device)
        if act is None:
            results[arg_name] = {'verdict': 'abstain', 'reason': 'could not extract activation'}
            continue
        grounding_prob = clf_grounding.predict_proba(act.numpy().reshape(1, -1))[0, 1]

        if grounding_prob < grounding_threshold:
            results[arg_name] = {'verdict': 'ungrounded', 'grounding_score': float(grounding_prob),
                                  'note': 'likely fabricated - no supporting evidence found'}
            continue

        if candidate_map and arg_name in candidate_map:
            candidates = candidate_map[arg_name]
            cand_scores = {}
            for cand in candidates:
                cand_args = {arg_name: cand}
                cand_continuation = build_continuation(function_name, cand_args)
                cand_act = extract_activation(model, tokenizer, context, cand_continuation, cand, layer=layer_disambig, device=device)
                if cand_act is None:
                    continue
                cand_scores[str(cand)] = float(clf_disambig.predict_proba(cand_act.numpy().reshape(1, -1))[0, 1])

            if len(cand_scores) > 1:
                sorted_scores = sorted(cand_scores.items(), key=lambda x: -x[1])
                top_cand, top_score = sorted_scores[0]
                second_cand, second_score = sorted_scores[1]
                if top_score >= disambig_threshold and second_score >= disambig_threshold and (top_score - second_score) < disambig_margin:
                    results[arg_name] = {'verdict': 'clarify', 'grounding_score': float(grounding_prob),
                                          'competing_candidates': cand_scores,
                                          'note': f'multiple values ({list(cand_scores.keys())}) appear grounded - ask user to confirm'}
                    continue

        results[arg_name] = {'verdict': 'grounded', 'grounding_score': float(grounding_prob)}

    return results

def check_tool_call_complete_v2(model, tokenizer, clf_grounding, clf_disambig, context, function_name, arguments,
                                   candidate_map=None, grounding_threshold=0.5, disambig_threshold=0.5, disambig_margin=0.15,
                                   layer_grounding=12, layer_disambig=18, device="cuda"):
    """
    v2: checks disambiguation candidates FIRST for any argument with a candidate_map entry,
    since a low single-value grounding score can itself be a symptom of ambiguity (not fabrication).
    Only falls through to a pure grounding check when no candidates are specified for that argument.
    """
    from model_utils import build_continuation, extract_activation

    continuation = build_continuation(function_name, arguments)
    results = {}

    for arg_name, arg_value in arguments.items():
        if candidate_map and arg_name in candidate_map:
            candidates = candidate_map[arg_name]
            cand_scores = {}
            for cand in candidates:
                cand_args = {arg_name: cand}
                cand_continuation = build_continuation(function_name, cand_args)
                cand_act = extract_activation(model, tokenizer, context, cand_continuation, cand, layer=layer_disambig, device=device)
                if cand_act is None:
                    continue
                cand_scores[str(cand)] = float(clf_disambig.predict_proba(cand_act.numpy().reshape(1, -1))[0, 1])

            if len(cand_scores) > 1:
                sorted_scores = sorted(cand_scores.items(), key=lambda x: -x[1])
                top_cand, top_score = sorted_scores[0]
                second_cand, second_score = sorted_scores[1]
                if top_score < disambig_threshold:
                    results[arg_name] = {'verdict': 'ungrounded', 'candidate_scores': cand_scores,
                                          'note': 'no candidate scores as grounded'}
                elif second_score >= disambig_threshold and (top_score - second_score) < disambig_margin:
                    results[arg_name] = {'verdict': 'clarify', 'candidate_scores': cand_scores,
                                          'note': f'multiple values ({list(cand_scores.keys())}) appear grounded - ask user to confirm'}
                else:
                    results[arg_name] = {'verdict': 'grounded', 'candidate_scores': cand_scores, 'chosen': top_cand}
                continue

        # no candidates specified for this argument - plain grounding check
        act = extract_activation(model, tokenizer, context, continuation, arg_value, layer=layer_grounding, device=device)
        if act is None:
            results[arg_name] = {'verdict': 'abstain', 'reason': 'could not extract activation'}
            continue
        grounding_prob = clf_grounding.predict_proba(act.numpy().reshape(1, -1))[0, 1]
        verdict = 'grounded' if grounding_prob >= grounding_threshold else 'ungrounded'
        results[arg_name] = {'verdict': verdict, 'grounding_score': float(grounding_prob)}

    return results
