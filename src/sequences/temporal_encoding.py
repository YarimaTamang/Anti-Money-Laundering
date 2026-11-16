"""Time encodings.

``learnable``: functional time encoding  Phi(dt) = cos(w * dt + b) with learnable
frequencies w and phases b, following the time encoding of TGAT (Xu et al., ICLR 2020).
Frequencies are initialised geometrically so that scales from ~1 minute to ~70 days
(~1e5 minutes) are represented.

``none``: ablation - the raw delta is only log-scaled, log1p(dt)/8, giving a single
scalar with no learned periodic basis.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class TimeEncoder(nn.Module):
    def __init__(self, dim: int = 16, mode: str = "learnable"):
        super().__init__()
        self.mode = mode
        if mode == "learnable":
            self.w = nn.Parameter(torch.tensor(1.0 / 10 ** np.linspace(0, 5, dim), dtype=torch.float32))
            self.b = nn.Parameter(torch.zeros(dim))
            self.out_dim = dim
        elif mode == "none":
            self.out_dim = 1
        else:
            raise ValueError(mode)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        dt = dt.float().clamp(min=0)
        if self.mode == "learnable":
            return torch.cos(dt.unsqueeze(-1) * self.w + self.b)
        return (torch.log1p(dt) / 8.0).unsqueeze(-1)


def cyclical(values: torch.Tensor, period: float) -> torch.Tensor:
    ang = values.float() / period * 2 * math.pi
    return torch.stack([torch.sin(ang), torch.cos(ang)], -1)
