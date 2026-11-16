"""Training-protocol tests: the training loop never sees the test split; selection uses validation only."""
import numpy as np
import pytest
import torch

from src.data.preprocessing import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL_SELECT
from src.evaluation.alert_budget import at_k
from src.evaluation.metrics import best_f1_threshold, classification_metrics
from src.models.registry import build_model
from src.training.trainer import fit
from src.utils.config import load_yaml


class GuardedStore:
    """Proxy that fails loudly if any test-split transaction is requested."""

    def __init__(self, store):
        self._s = store
        self.requested = set()

    def __getattr__(self, k):
        return getattr(self._s, k)

    def build(self, eids, *a, **kw):
        eids = np.asarray(eids)
        if (self._s.a["split"][eids] == SPLIT_TEST).any():
            raise AssertionError("training loop accessed TEST transactions")
        self.requested.update(np.unique(self._s.a["split"][eids]).tolist())
        return self._s.build(eids, *a, **kw)


def test_fit_never_touches_test_split(store):
    cfg = load_yaml("training")
    cfg.update(max_epochs=1, neg_per_epoch=1500, batch_size=256, eval_batch_size=1024)
    split, y = store.a["split"], store.a["label"]
    tr = np.flatnonzero(split == SPLIT_TRAIN)
    va = np.flatnonzero(split == SPLIT_VAL_SELECT)[:3000]
    va = np.union1d(va, np.flatnonzero((split == SPLIT_VAL_SELECT) & (y == 1)))
    g = GuardedStore(store)
    m = build_model("tracex", {"edge_dim": store.edge_dim, "node_dim": store.node_dim, "seq_dim": store.seq_dim})
    m, hist, best = fit(m, g, tr[y[tr] == 1], tr[y[tr] == 0], va, np.ones(len(va)), cfg, seed=0, log=lambda *_: None)
    assert len(hist) == 1 and np.isfinite(hist[0]["train_loss"])
    assert g.requested <= {SPLIT_TRAIN, SPLIT_VAL_SELECT}


def test_threshold_and_metrics():
    y = np.array([0, 0, 1, 1, 0, 1], float)
    p = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
    thr, f1 = best_f1_threshold(y, p)
    m = classification_metrics(y, p, thr)
    assert np.isclose(m["f1"], f1)
    assert m["confusion_matrix"]["tp"] + m["confusion_matrix"]["fn"] == 3
    r = at_k(y, p, 2)
    assert r["precision_at_k"] == 1.0 and np.isclose(r["recall_at_k"], 2 / 3)


def test_weighted_metrics_match_full_population():
    """Inverse-probability weighting of a negative subsample reproduces full-population precision/recall."""
    rng = np.random.default_rng(0)
    n = 200_000
    y = (rng.random(n) < 0.01).astype(float)
    p = np.clip(0.3 * y + rng.random(n) * 0.7, 0, 1)
    full = classification_metrics(y, p, 0.6)
    neg = np.flatnonzero(y == 0)
    s = np.concatenate([np.flatnonzero(y == 1), rng.choice(neg, 40_000, replace=False)])
    w = np.where(y[s] == 1, 1.0, len(neg) / 40_000)
    sub = classification_metrics(y[s], p[s], 0.6, w)
    assert abs(sub["precision"] - full["precision"]) < 0.02
    assert abs(sub["recall"] - full["recall"]) < 1e-9
    assert abs(sub["pr_auc"] - full["pr_auc"]) < 0.02
