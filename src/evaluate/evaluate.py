from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np


def evaluate(labels, preds, probs=None):
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }

    if probs is not None:
        try:
            metrics["auc"] = roc_auc_score(labels, np.array(probs), multi_class="ovr")
        except Exception:
            metrics["auc"] = None

    return metrics