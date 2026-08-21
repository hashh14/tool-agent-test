
def score_candidates(model, tokenizer, extract_activation_fn, build_continuation_fn, clf,
                      ex, layer=12, device="cuda"):
    """
    Scores every candidate value for a single ambiguous-argument example.

    For each candidate, builds a continuation as if that candidate were the
    argument's actual value, extracts the activation at that value's token
    position (same extract_activation used throughout the project), and
    scores it with the existing raw-activation classifier.

    ex: dict with keys 'context', 'function_name', 'arguments', 'candidates'
        (matches ambiguous_examples / unambiguous_controls structure).

    Returns: {'arg_name': str, 'candidate_scores': [{'candidate': ..., 'score': float}, ...]}
    """
    arg_name = list(ex['arguments'].keys())[0]
    candidate_scores = []

    for cand in ex['candidates']:
        arguments = dict(ex['arguments'])
        arguments[arg_name] = cand

        continuation = build_continuation_fn(ex['function_name'], arguments)
        act = extract_activation_fn(
            model, tokenizer, ex['context'], continuation, cand,
            layer=layer, device=device
        )

        if act is None:
            candidate_scores.append({'candidate': cand, 'score': None, 'error': 'activation_extraction_failed'})
            continue

        prob = clf.predict_proba(act.numpy().reshape(1, -1))[0, 1]
        candidate_scores.append({'candidate': cand, 'score': float(prob)})

    return {'arg_name': arg_name, 'candidate_scores': candidate_scores}


def decide_outcome(candidate_scores, floor=0.5, margin=0.15):
    """
    Two-outcome decision rule: 'answer' vs 'flag'.

    - Single candidate: 'answer' if its score >= floor, else 'flag'
      (nothing to disambiguate between, but low confidence still gets flagged).
    - Multiple candidates: 'answer' only if the top score clears the floor
      AND is ahead of the next-best by at least `margin`; otherwise 'flag'.

    Returns: dict with 'outcome', 'top_candidate', 'top_score', and the
    full sorted candidate list (so downstream phrasing can tell a
    single-candidate flag from a genuine multi-candidate one).
    """
    sorted_scores = sorted(candidate_scores, key=lambda cs: cs['score'], reverse=True)
    top = sorted_scores[0]

    if len(sorted_scores) == 1:
        outcome = 'answer' if top['score'] >= floor else 'flag'
    else:
        second = sorted_scores[1]
        outcome = 'answer' if (top['score'] >= floor and (top['score'] - second['score']) >= margin) else 'flag'

    return {
        'outcome': outcome,
        'top_candidate': top['candidate'],
        'top_score': top['score'],
        'sorted_candidates': sorted_scores,
    }


def causal_split_belief_test(model, tokenizer, get_next_token_prob_fn, state, direction, ex, strength=-12.0):
    """
    Tests whether suppressing the grounding direction shifts next-token
    probability mass away from a confidently-wrong candidate toward the
    correct one, for a 'confidently wrong' misattribution case (both
    candidates present in context, classifier picked the wrong one).

    Runs get_next_token_prob for BOTH candidates (wrong-but-preferred,
    and correct), first with no intervention (baseline), then with the
    direction suppressed (intervened). One clean attempt, no strength
    sweep -- per project convention on this step.
    """
    fn = ex['function_name']
    arg_name = list(ex['arguments'].keys())[0]
    wrong_candidate = ex['decision']['top_candidate']
    correct_candidate = ex['correct_candidate']

    results = {}
    for label, cand in [('wrong', wrong_candidate), ('correct', correct_candidate)]:
        args = dict(ex['arguments'])
        args[arg_name] = cand

        state.active = False
        baseline_prob, baseline_top5, seq_len = get_next_token_prob_fn(
            model, tokenizer, ex['context'], fn, args, arg_name
        )

        state.active = True
        state.direction = direction
        state.strength = strength
        state.token_positions = [seq_len - 1]
        intervened_prob, intervened_top5, _ = get_next_token_prob_fn(
            model, tokenizer, ex['context'], fn, args, arg_name
        )
        state.active = False

        results[label] = {
            'candidate': cand,
            'baseline_prob': baseline_prob,
            'intervened_prob': intervened_prob,
            'baseline_top5': baseline_top5,
            'intervened_top5': intervened_top5,
        }

    return results
