"""Monte Carlo Dropout (Gal & Ghahramani, ICML 2016).

Dropout layers are switched to train mode at inference (no other layer is stochastic in
these models - LayerNorm has no running statistics), and T stochastic forward passes give
a predictive mean and standard deviation of the suspicious probability.
"""
from __future__ import annotations

import numpy as np
import torch

from src.training.trainer import Prefetcher, _chunks


def enable_dropout(model: torch.nn.Module) -> None:
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
    # functional dropout inside attention layers is gated by module.training
    for m in model.modules():
        if m.__class__.__name__ == "EdgeTemporalGATConv":
            m.train()


@torch.inference_mode()
def mc_predict(model, store, eids: np.ndarray, cfg: dict, T: int = 20, bs: int = 4096, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    enable_dropout(model)
    build = lambda e: store.build(e, model.needs, cfg["seq_len"], cfg["k1"], cfg["k2"])  # noqa: E731
    means, stds = [], []
    for b in Prefetcher(build, _chunks(np.asarray(eids), bs)):
        ps = torch.stack([torch.sigmoid(model(b)) for _ in range(T)])
        means.append(ps.mean(0).numpy()); stds.append(ps.std(0).numpy())
    model.eval()
    return np.concatenate(means), np.concatenate(stds)
