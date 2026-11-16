"""Classification metrics for imbalanced AML detection.

Weighted variants accept per-sample weights so that a validation subsample
(all positives + uniformly sampled negatives, negatives weighted by
1 / sampling-rate) yields unbiased estimates of full-split precision/recall-based
metrics. Test metrics are always computed on the complete test split with unit weights.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def best_f1_threshold(y, p, w=None) -> tuple[float, float]:
    """Threshold maximising (weighted) F1 - computed on validation data only."""
    w = np.ones_like(p) if w is None else w
    o = np.argsort(-p, kind="stable")
    ys, ps, ws = y[o], p[o], w[o]
    tp = np.cumsum(ws * ys)
    fp = np.cumsum(ws * (1 - ys))
    P = (ws * ys).sum()
    # only evaluate at the last index of each distinct score
    last = np.r_[ps[1:] != ps[:-1], True]
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / max(P, 1e-12)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    f1 = np.where(last, f1, -1)
    i = int(np.argmax(f1))
    return float(ps[i]), float(f1[i])


def classification_metrics(y, p, threshold: float, w=None) -> dict:
    w = np.ones_like(p, dtype=np.float64) if w is None else w
    pred = (p >= threshold).astype(int)
    tp = float((w * (pred == 1) * (y == 1)).sum())
    fp = float((w * (pred == 1) * (y == 0)).sum())
    fn = float((w * (pred == 0) * (y == 1)).sum())
    tn = float((w * (pred == 0) * (y == 0)).sum())
    prec = tp / max(tp + fp, 1e-12)
    rec = tp / max(tp + fn, 1e-12)
    out = {
        "threshold": float(threshold),
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-12),
        "roc_auc": float(roc_auc_score(y, p, sample_weight=w)) if 0 < y.sum() < len(y) else float("nan"),
        "pr_auc": float(average_precision_score(y, p, sample_weight=w)) if y.sum() > 0 else float("nan"),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "n": float(w.sum()), "n_pos": float((w * y).sum()),
    }
    return out


def confusion(y, p, threshold):
    return confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1])
