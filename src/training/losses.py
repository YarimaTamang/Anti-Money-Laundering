"""Loss functions operating on logits."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class WeightedBCE(torch.nn.Module):
    """Class-weighted BCE-with-logits (sigmoid is applied inside the loss, never in the model)."""

    def __init__(self, pos_weight: float):
        super().__init__()
        self.register_buffer("pw", torch.tensor(float(pos_weight)))

    def forward(self, logits, y):
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pw)


class FocalLoss(torch.nn.Module):
    """Binary focal loss (Lin et al., 2017) on logits."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, y):
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * y + (1 - p) * (1 - y)
        a = self.alpha * y + (1 - self.alpha) * (1 - y)
        return (a * (1 - pt) ** self.gamma * bce).mean()


def make_loss(cfg: dict, n_pos: int, n_neg: int) -> torch.nn.Module:
    kind = cfg.get("type", "bce")
    if kind == "focal":
        return FocalLoss(cfg.get("alpha", 0.25), cfg.get("gamma", 2.0))
    pw = cfg.get("pos_weight", "auto")
    if pw == "auto":
        pw = n_neg / max(n_pos, 1)
    elif pw == "sqrt":
        pw = (n_neg / max(n_pos, 1)) ** 0.5
    return WeightedBCE(float(pw))
