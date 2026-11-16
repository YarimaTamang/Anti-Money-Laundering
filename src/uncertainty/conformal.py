"""Split conformal prediction for binary AML classification.

Class-conditional (Mondrian) split conformal (Vovk et al., 2005; see also Angelopoulos &
Bates, 2021): with nonconformity score s(x, y) = 1 - p_y(x), the per-class threshold
    q_y = the ceil((n_y + 1)(1 - alpha)) / n_y empirical quantile of calibration scores of class y
guarantees P(y in C(x) | Y = y) >= 1 - alpha for each class under exchangeability. Class-
conditional calibration is used because a marginal guarantee would be dominated by the
~99.9% normal class.

Calibration data: the val_cal split (all positives + a uniform negative sample; a uniform
sample of a class is still exchangeable with that class, so no weights are needed).
Test labels are never used to choose alpha or thresholds.

Decision mapping (decision support):
  C = {0}           -> high-confidence normal
  C = {1}           -> high-confidence suspicious alert
  C = {0,1} or {}   -> uncertain, route to human review
"""
from __future__ import annotations

import math

import numpy as np

OUTCOMES = {0: "confident_normal", 1: "confident_suspicious", 2: "human_review"}


def class_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return 1.0
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


class MondrianConformal:
    def fit(self, p_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.1):
        self.alpha = alpha
        self.q0 = class_quantile(p_cal[y_cal == 0], alpha)          # score for y=0 is p
        self.q1 = class_quantile(1 - p_cal[y_cal == 1], alpha)      # score for y=1 is 1-p
        # include 0 iff p <= q0 ; include 1 iff p >= 1 - q1
        self.t_normal, self.t_suspicious = self.q0, 1 - self.q1
        return self

    def sets(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return p <= self.t_normal, p >= self.t_suspicious

    def outcome(self, p: np.ndarray) -> np.ndarray:
        has0, has1 = self.sets(p)
        out = np.full(len(p), 2, np.int8)
        out[has0 & ~has1] = 0
        out[has1 & ~has0] = 1
        return out

    def report(self, p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> dict:
        w = np.ones_like(p) if w is None else w
        has0, has1 = self.sets(p)
        covered = np.where(y == 1, has1, has0)
        oc = self.outcome(p)
        ret = oc != 2
        pred = oc == 1
        tp = (w * (ret & pred & (y == 1))).sum(); fp = (w * (ret & pred & (y == 0))).sum()
        fn = (w * (ret & ~pred & (y == 1))).sum()
        W = w.sum()
        return {
            "alpha": self.alpha, "t_normal": self.t_normal, "t_suspicious": self.t_suspicious,
            "coverage_overall": float((w * covered).sum() / W),
            "coverage_class0": float((w * covered * (y == 0)).sum() / max((w * (y == 0)).sum(), 1e-12)),
            "coverage_class1": float((w * covered * (y == 1)).sum() / max((w * (y == 1)).sum(), 1e-12)),
            "frac_confident_normal": float((w * (oc == 0)).sum() / W),
            "frac_confident_suspicious": float((w * (oc == 1)).sum() / W),
            "abstention_rate": float((w * (oc == 2)).sum() / W),
            "positives_routed_to_review": float((w * (oc == 2) * (y == 1)).sum() / max((w * (y == 1)).sum(), 1e-12)),
            "retained_precision": float(tp / max(tp + fp, 1e-12)),
            "retained_recall": float(tp / max(tp + fn, 1e-12)),
            "retained_accuracy": float((w * ret * ((pred & (y == 1)) | (~pred & (y == 0)))).sum() / max((w * ret).sum(), 1e-12)),
            "missed_positives_marked_confident_normal": float((w * (oc == 0) * (y == 1)).sum()),
        }
