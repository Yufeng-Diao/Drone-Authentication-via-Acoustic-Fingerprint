"""Explicit macro metrics, preserving attack identity before label collapse."""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(truth, prediction, labels) -> dict:
    """Standard per-class F1 then macro mean; undefined precision/recall is zero."""
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if (truth.ndim != 1 or truth.shape != prediction.shape or not truth.size
            or len(set(labels)) != len(labels) or not set(truth) <= set(labels)
            or not set(prediction) <= set(labels)):
        raise ValueError("Metric inputs must be nonempty aligned vectors with declared labels")
    p, r, f, _ = precision_recall_fscore_support(truth, prediction, labels=labels, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(truth, prediction)), "macro_precision": float(p),
            "macro_recall": float(r), "macro_f1": float(f)}


def open_set_metrics(original_identity, prediction, partition) -> dict:
    """Registered macro recall and per-attack rejection; room noise is never tested."""
    identities, prediction = np.asarray(original_identity), np.asarray(prediction)
    registered, attack = partition["registered"], partition["attack"]
    if (identities.ndim != 1 or identities.shape != prediction.shape
            or set(identities) != set(registered + attack) or not set(prediction) <= set([0] + registered)):
        raise ValueError("Open-set evaluation needs every registered/attack identity and no background test rows")
    recall = {str(i): float(np.mean(prediction[identities == i] == i)) for i in registered}
    rejection = {str(i): float(np.mean(prediction[identities == i] == 0)) for i in attack}
    return {"registered_macro_recall": float(np.mean(list(recall.values()))),
            "attack_macro_rejection": float(np.mean(list(rejection.values()))),
            "attack_pooled_rejection": float(np.mean(prediction[np.isin(identities, attack)] == 0)),
            "registered_recall_by_identity": recall, "attack_rejection_by_identity": rejection}
