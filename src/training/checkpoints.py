"""Checkpoint save/restore (weights + everything needed to rebuild the model)."""
from __future__ import annotations

from pathlib import Path

import torch

from src.models.registry import build_model


def save_checkpoint(path: str | Path, model: torch.nn.Module, arch: str, params: dict, dims: dict, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": arch, "params": params, "dims": dims, "extra": extra or {}}, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(ck["arch"], ck["dims"], ck["params"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
