import numpy as np
from typing import Sequence, Optional


def format_confusion_matrix(cm, class_names: Optional[Sequence[str]] = None) -> str:
    """
    Return a nicely formatted string representation of a confusion matrix.

    Args:
        cm: 2D array-like (list or np.array) confusion matrix
        class_names: optional list of class names (length == cm.shape[0])

    Returns:
        str: multi-line formatted table
    """
    cm = np.array(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("confusion matrix must be square (N x N)")

    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]
    if len(class_names) != n:
        raise ValueError("class_names length must match confusion matrix size")

    # compute column widths
    max_name = max(len(name) for name in class_names)
    max_val = max(len(str(int(v))) for v in cm.flatten())
    cell_width = max(max_name, max_val, 3) + 2

    # header
    header = " " * (max_name + 3) + "".join(name.rjust(cell_width) for name in class_names) + "  |  RowSum\n"
    # indicate axes: columns are actual, rows are predicted
    matrix_label = "(columns = Actual, rows = Predicted)\n"
    sep = "-" * (len(header) - 1) + "\n"

    # rows with row sums
    rows = []
    for i, name in enumerate(class_names):
        row_vals = "".join(str(int(v)).rjust(cell_width) for v in cm[i, :])
        row_sum = str(int(cm[i, :].sum())).rjust(6)
        rows.append(f"{name.ljust(max_name)}   {row_vals}  | {row_sum}\n")

    # column sums (bottom)
    col_sums = cm.sum(axis=0)
    col_sum_row = "Totals".ljust(max_name) + "   " + "".join(str(int(v)).rjust(cell_width) for v in col_sums) + "  |  " + str(int(col_sums.sum())).rjust(6) + "\n"

    return matrix_label + header + sep + "".join(rows) + sep + col_sum_row


def print_metrics(metrics: dict, class_names: Optional[Sequence[str]] = None) -> None:
    """
    Nicely print main metrics and confusion matrix (if present).

    Args:
        metrics: dict from evaluate(...) containing keys like 'accuracy','f1','auc','confusion_matrix'
        class_names: optional list of class names for confusion matrix
    """
    print("\n=== Evaluation metrics ===")
    for k in ("accuracy", "f1", "auc"):
        if k in metrics:
            print(f"{k:<10}: {metrics[k]}")
    if "confusion_matrix" in metrics:
        try:
            print("Formatting confusion matrix...")
            cm_str = format_confusion_matrix(metrics["confusion_matrix"], class_names=class_names)
            print("\nConfusion Matrix:")
            print(cm_str)
        except Exception as e:
            print(f"Could not format confusion matrix: {e}")
            print("Raw confusion matrix:", metrics["confusion_matrix"])
