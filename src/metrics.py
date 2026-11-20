"""
Metrics for building damage classification.

Macro-F1 is the headline number: it averages the per-class F1 without weighting by class size,
so the 3.9 % major-damage class counts as much as the 74 % no-damage class. Plain accuracy would
sit near 0.75 for a model that only ever predicts no-damage.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
N_CLASSES = len(CLASS_NAMES)


def compute_metrics(y_true, y_pred, n_classes=N_CLASSES):
    """All the numbers for one split, as a flat dict that TensorBoard and JSON both accept."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(n_classes))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)

    out = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(np.mean([recall[c] for c in labels if support[c] > 0])),
        "n": int(len(y_true)),
    }
    for c in labels:
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
        out[f"f1/{name}"] = float(f1[c])
        out[f"recall/{name}"] = float(recall[c])
        out[f"precision/{name}"] = float(precision[c])
        out[f"support/{name}"] = int(support[c])
    return out


def confusion(y_true, y_pred, n_classes=N_CLASSES):
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def format_confusion(cm, class_names=CLASS_NAMES):
    """Readable confusion matrix: rows are the truth, columns the prediction."""
    width = max(len(n) for n in class_names) + 2
    head = " " * width + "".join(f"{n[:9]:>11}" for n in class_names) + f"{'recall':>9}"
    lines = ["true \\ predicted", head]
    for i, row in enumerate(cm):
        total = row.sum()
        rec = row[i] / total if total else float("nan")
        lines.append(f"{class_names[i]:<{width}}" + "".join(f"{v:>11,}" for v in row) + f"{rec:>9.3f}")
    return "\n".join(lines)


def format_metrics(m, title=""):
    """One-line summary plus the per-class F1, for the console."""
    head = f"{title}  " if title else ""
    per_class = "  ".join(f"{n[:5]}={m[f'f1/{n}']:.3f}" for n in CLASS_NAMES)
    return (f"{head}macro-F1 {m['macro_f1']:.4f}  acc {m['accuracy']:.4f}  "
            f"bal-acc {m['balanced_accuracy']:.4f}  n={m['n']:,}\n    per-class F1: {per_class}")
