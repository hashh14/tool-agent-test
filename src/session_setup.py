import sys
sys.path.append('/content/drive/MyDrive/tool_call_grounding/src')

import json, pickle, random, torch, os, re, statistics
import numpy as np

DATA_DIR = '/content/drive/MyDrive/tool_call_grounding'

def full_setup(device=None):
    """
    One-call session restore: mounts Drive (if not already), loads model,
    tokenizer, corrected benchmark, reconstructs 15k sample, loads/trains
    the primary raw-activation classifier. Returns a dict of everything.
    """
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)

    import importlib
    for mod_name in ['data_processing', 'model_utils', 'classifier', 'baseline',
                      'inference', 'paper_log', 'intervention', 'checkpointing']:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

    from data_processing import corrupt_context
    from model_utils import load_model, build_continuation, get_target_token_position, extract_activation
    from classifier import build_labeled_dataset, rank_features_by_mutual_info, train_classifier, evaluate_classifier
    from intervention import InterventionState, register_hook, get_next_token_prob, build_prefix_and_target
    from paper_log import log_result
    from checkpointing import load_checkpoint, save_checkpoint
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model(device=device)

    with open(f'{DATA_DIR}/exact_matched_full.json', 'r') as f:
        deduped = json.load(f)

    random.seed(42)
    sample_15k = random.sample(deduped, 15000)
    for ex in sample_15k:
        ex['context_with_evidence'] = ex['context']
        ex['context_without_evidence'] = corrupt_context(ex['context'], ex['span'])

    with open(f'{DATA_DIR}/activations_15k_final.pkl', 'rb') as f:
        activations_15k = pickle.load(f)

    acts_with_15k = torch.stack([a['act_with'] for a in activations_15k]).float()
    acts_without_15k = torch.stack([a['act_without'] for a in activations_15k]).float()
    X_raw = np.concatenate([acts_with_15k.numpy(), acts_without_15k.numpy()], axis=0)
    y_raw = np.concatenate([np.ones(len(acts_with_15k)), np.zeros(len(acts_without_15k))])

    n_pairs = len(activations_15k)
    pair_idx = np.arange(n_pairs)
    train_pairs, test_pairs = train_test_split(pair_idx, test_size=0.1, random_state=42)

    def subset_by_pairs(X, y, pidx, n):
        idx = np.concatenate([pidx, pidx + n])
        return X[idx], y[idx]

    X_train_raw, y_train_raw = subset_by_pairs(X_raw, y_raw, train_pairs, n_pairs)
    X_test_raw, y_test_raw = subset_by_pairs(X_raw, y_raw, test_pairs, n_pairs)

    clf_raw_15k = LogisticRegression(max_iter=1000)
    clf_raw_15k.fit(X_train_raw, y_train_raw)

    print(f"Setup complete. device={device}, deduped={len(deduped)}, sample_15k={len(sample_15k)}, "
          f"activations_15k={len(activations_15k)}, clf_raw_15k trained.")

    return {
        'device': device, 'model': model, 'tokenizer': tokenizer,
        'deduped': deduped, 'sample_15k': sample_15k,
        'activations_15k': activations_15k,
        'acts_with_15k': acts_with_15k, 'acts_without_15k': acts_without_15k,
        'X_raw': X_raw, 'y_raw': y_raw, 'n_pairs': n_pairs,
        'train_pairs': train_pairs, 'test_pairs': test_pairs,
        'X_train_raw': X_train_raw, 'y_train_raw': y_train_raw,
        'X_test_raw': X_test_raw, 'y_test_raw': y_test_raw,
        'clf_raw_15k': clf_raw_15k, 'subset_by_pairs': subset_by_pairs,
        'DATA_DIR': DATA_DIR,
        'log_result': log_result,
        'build_continuation': build_continuation,
        'get_target_token_position': get_target_token_position,
        'extract_activation': extract_activation,
        'InterventionState': InterventionState, 'register_hook': register_hook,
        'get_next_token_prob': get_next_token_prob, 'build_prefix_and_target': build_prefix_and_target,
    }

def full_setup_1p5b(device=None):
    """
    One-call session restore for the 1.5B model pipeline: mounts Drive,
    loads Qwen2.5-1.5B, loads its cached activations, trains its raw
    classifier. Mirrors full_setup() for the 0.5B model.
    """
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)

    import importlib
    for mod_name in ['data_processing', 'model_utils', 'intervention', 'paper_log', 'checkpointing']:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

    from data_processing import corrupt_context
    from model_utils import load_model, build_continuation, get_target_token_position, extract_activation
    from intervention import InterventionState, register_hook, get_next_token_prob, build_prefix_and_target
    from paper_log import log_result
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_1p5b, tokenizer_1p5b = load_model(model_name="Qwen/Qwen2.5-1.5B-Instruct", device=device)

    with open(f'{DATA_DIR}/exact_matched_full.json', 'r') as f:
        deduped = json.load(f)

    random.seed(42)
    sample_5k_bigmodel = random.sample(deduped, 5000)
    for ex in sample_5k_bigmodel:
        ex['context_with_evidence'] = ex['context']
        ex['context_without_evidence'] = corrupt_context(ex['context'], ex['span'])

    with open(f'{DATA_DIR}/activations_1p5b_final.pkl', 'rb') as f:
        activations_1p5b = pickle.load(f)

    acts_with_big = torch.stack([a['act_with'] for a in activations_1p5b]).float()
    acts_without_big = torch.stack([a['act_without'] for a in activations_1p5b]).float()
    X_raw_big = np.concatenate([acts_with_big.numpy(), acts_without_big.numpy()], axis=0)
    y_raw_big = np.concatenate([np.ones(len(acts_with_big)), np.zeros(len(acts_without_big))])

    n_pairs_big = len(activations_1p5b)
    pair_idx_big = np.arange(n_pairs_big)
    train_pairs_big, test_pairs_big = train_test_split(pair_idx_big, test_size=0.1, random_state=42)

    def subset_by_pairs(X, y, pidx, n):
        idx = np.concatenate([pidx, pidx + n])
        return X[idx], y[idx]

    X_train_raw_big, y_train_raw_big = subset_by_pairs(X_raw_big, y_raw_big, train_pairs_big, n_pairs_big)
    X_test_raw_big, y_test_raw_big = subset_by_pairs(X_raw_big, y_raw_big, test_pairs_big, n_pairs_big)

    clf_raw_big = LogisticRegression(max_iter=1000)
    clf_raw_big.fit(X_train_raw_big, y_train_raw_big)

    print(f"1.5B setup complete. device={device}, sample_5k={len(sample_5k_bigmodel)}, "
          f"activations={len(activations_1p5b)}, clf_raw_big trained.")

    return {
        'device': device, 'model': model_1p5b, 'tokenizer': tokenizer_1p5b,
        'sample_5k': sample_5k_bigmodel, 'activations': activations_1p5b,
        'acts_with': acts_with_big, 'acts_without': acts_without_big,
        'n_pairs': n_pairs_big, 'train_pairs': train_pairs_big, 'test_pairs': test_pairs_big,
        'clf_raw': clf_raw_big, 'subset_by_pairs': subset_by_pairs,
        'DATA_DIR': DATA_DIR, 'log_result': log_result,
        'build_continuation': build_continuation,
        'InterventionState': InterventionState, 'register_hook': register_hook,
        'get_next_token_prob': get_next_token_prob,
    }
