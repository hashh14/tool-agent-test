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
