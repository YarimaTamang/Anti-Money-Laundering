"""Shared helpers for explanation methods."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.graph.temporal_graph import TemporalGraphStore
from src.training.checkpoints import load_checkpoint
from src.utils.config import PROJECT_ROOT, load_yaml, processed_dir


@dataclass
class Explainable:
    model: torch.nn.Module
    store: TemporalGraphStore
    cfg: dict  # training cfg (seq_len, k1, k2)

    def batch(self, eids, meta: bool = True) -> dict:
        return self.store.build(np.atleast_1d(np.asarray(eids)), ("seq", "graph"), self.cfg["seq_len"], self.cfg["k1"],
                                self.cfg["k2"], meta=meta)

    @torch.no_grad()
    def prob(self, b: dict, **kw) -> np.ndarray:
        return torch.sigmoid(self.model(b, **kw)).cpu().numpy()


def load_explainable(exp: str = "tracex", seed: int = 0, mode: str = "full", store: TemporalGraphStore | None = None) -> Explainable:
    model, ck = load_checkpoint(PROJECT_ROOT / "checkpoints" / mode / exp / f"seed{seed}.pt")
    for p in model.parameters():
        p.requires_grad_(False)
    store = store or TemporalGraphStore(processed_dir(load_yaml("data"), mode))
    return Explainable(model.eval(), store, ck["extra"]["cfg"])


def account_name(store: TemporalGraphStore, eid: int, side: int, accounts: list | None = None) -> str:
    acc = int(store.a["src"][eid] if side == 0 else store.a["dst"][eid])
    return accounts[acc] if accounts is not None else str(acc)


def load_accounts(store: TemporalGraphStore) -> list:
    import pandas as pd

    return pd.read_parquet(Path(store.dir) / "accounts.parquet")["account"].tolist()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def topk_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = min(k, len(a))
    sa, sb = set(np.argsort(-np.abs(a))[:k]), set(np.argsort(-np.abs(b))[:k])
    return len(sa & sb) / max(len(sa | sb), 1)
