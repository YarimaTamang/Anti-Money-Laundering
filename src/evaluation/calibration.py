"""Probability calibration: Brier score, ECE, reliability curve, Platt scaling.

Note: models are trained on negative-subsampled, class-weighted data, so raw
probabilities are not expected to be calibrated. Platt scaling is fitted on the
val_cal split (never the test split) to produce calibrated scores.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def brier(y, p, w=None):
    w = np.ones_like(p) if w is None else w
    return float((w * (p - y) ** 2).sum() / w.sum())


def reliability(y, p, n_bins: int = 15, w=None, strategy: str = "quantile"):
    w = np.ones_like(p) if w is None else w
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    conf, acc, cnt = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        ww = w[m]
        conf.append(float((ww * p[m]).sum() / ww.sum()))
        acc.append(float((ww * y[m]).sum() / ww.sum()))
        cnt.append(float(ww.sum()))
    return np.array(conf), np.array(acc), np.array(cnt)


def ece(y, p, n_bins: int = 15, w=None, strategy: str = "uniform") -> float:
    conf, acc, cnt = reliability(y, p, n_bins, w, strategy)
    return float((cnt * np.abs(conf - acc)).sum() / cnt.sum())


class PlattScaler:
    """Logistic calibration of logits fitted on a calibration split (with optional sample weights)."""

    def fit(self, logits, y, w=None):
        self.lr = LogisticRegression(C=1e4, max_iter=1000)
        self.lr.fit(logits.reshape(-1, 1), y, sample_weight=w)
        self.a, self.b = float(self.lr.coef_[0, 0]), float(self.lr.intercept_[0])
        return self

    def transform(self, logits):
        z = self.a * logits + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def calibration_report(y, p, w=None) -> dict:
    return {"brier": brier(y, p, w), "ece_uniform15": ece(y, p, 15, w, "uniform"),
            "ece_quantile15": ece(y, p, 15, w, "quantile")}
