from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np


def evaluate(labels, preds, probs=None):
    """
    Compute evaluation metrics for classification predictions.

    Calculates standard classification metrics including accuracy,
    weighted F1 score, and confusion matrix. Optionally computes
    multi-class AUC if probability scores are provided.

    Args:
        labels (array-like): Ground truth class labels.
        preds (array-like): Predicted class labels.
        probs (array-like, optional): Predicted class probabilities
            with shape (n_samples, n_classes). Defaults to None.

    Returns:
        dict: Dictionary containing evaluation metrics:
            - accuracy (float)
            - f1 (float, weighted)
            - confusion_matrix (list of lists)
            - auc (float or None, if probabilities provided)
    """
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