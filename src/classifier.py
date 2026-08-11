import torch
import numpy as np
import pickle
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


def load_activations(data_dir):
    """Loads and combines the prototype + big_batch cached activation files."""
    with open(f'{data_dir}/prototype_activations_final.pkl', 'rb') as f:
        prototype_activations = pickle.load(f)
    with open(f'{data_dir}/big_batch_activations_final.pkl', 'rb') as f:
        big_batch_activations = pickle.load(f)
    return prototype_activations + big_batch_activations


def train_raw_activation_classifier(all_activations):
    """
    Builds the with/without-evidence labeled dataset directly from raw
    activations (no crosscoder) and trains a logistic regression classifier.
    Returns the fitted classifier.
    """
    acts_with = torch.stack([a['act_with'] for a in all_activations]).float()
    acts_without = torch.stack([a['act_without'] for a in all_activations]).float()
    X = np.concatenate([acts_with.numpy(), acts_without.numpy()], axis=0)
    y = np.concatenate([np.ones(len(acts_with)), np.zeros(len(acts_without))])
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf


def build_labeled_dataset(crosscoder, acts_with, acts_without, device="cuda"):
    """Turn paired with/without activations into a labeled feature dataset via the crosscoder's encoder."""
    with torch.no_grad():
        f_with = crosscoder.encode(acts_with.to(device)).cpu().numpy()
        f_without = crosscoder.encode(acts_without.to(device)).cpu().numpy()
    X = np.concatenate([f_with, f_without], axis=0)
    y = np.concatenate([np.ones(len(f_with)), np.zeros(len(f_without))], axis=0)
    return X, y


def rank_features_by_mutual_info(X_train, y_train, random_state=42):
    mi_scores = mutual_info_classif(X_train, y_train, random_state=random_state)
    ranked_indices = np.argsort(mi_scores)[::-1]
    return ranked_indices, mi_scores


def train_classifier(X_train, y_train, top_k_indices):
    X_sub = X_train[:, top_k_indices]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_sub, y_train)
    return clf


def evaluate_classifier(clf, X_test, y_test, top_k_indices):
    X_sub = X_test[:, top_k_indices]
    preds = clf.predict(X_sub)
    probs = clf.predict_proba(X_sub)[:, 1]
    return {
        'accuracy': accuracy_score(y_test, preds),
        'auc': roc_auc_score(y_test, probs),
        'confusion_matrix': confusion_matrix(y_test, preds),
    }
