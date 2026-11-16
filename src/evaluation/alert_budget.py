"""Alert-budget (top-K) evaluation.

Protocol: all transactions of the evaluated split are ranked by predicted score
(descending, ties broken by transaction order). An investigator reviews the top
K = ceil(budget * N) alerts, where N = number of transactions in the split.
  Precision@K = laundering transactions among the K alerts / K
  Recall@K    = laundering transactions among the K alerts / all laundering transactions in the split
  (Recall@K is the "detection rate" at that alert budget.)
"""
from __future__ import annotations

import math

import numpy as np

DEFAULT_BUDGETS = (0.001, 0.005, 0.01, 0.02)


def at_k(y: np.ndarray, p: np.ndarray, k: int) -> dict:
    o = np.argsort(-p, kind="stable")[:k]
    hits = float(y[o].sum())
    return {"k": int(k), "precision_at_k": hits / max(k, 1), "recall_at_k": hits / max(float(y.sum()), 1.0), "hits": hits}


def alert_budget_metrics(y: np.ndarray, p: np.ndarray, budgets=DEFAULT_BUDGETS) -> dict:
    n = len(y)
    out = {}
    for b in budgets:
        out[f"{b * 100:g}%"] = at_k(y, p, int(math.ceil(b * n)))
    out["k=n_pos"] = at_k(y, p, int(y.sum()))
    return out


def recall_curve(y: np.ndarray, p: np.ndarray, max_budget: float = 0.05, points: int = 60):
    n = len(y)
    o = np.argsort(-p, kind="stable")
    cum = np.cumsum(y[o])
    budgets = np.linspace(0, max_budget, points)[1:]
    ks = np.maximum(1, (budgets * n).astype(int))
    return budgets, cum[ks - 1] / max(y.sum(), 1)
