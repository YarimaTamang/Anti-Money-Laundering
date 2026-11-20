"""YAML configuration loading helpers."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(name: str | Path) -> dict:
    path = Path(name)
    if path.suffix == "":
        path = path.with_suffix(".yaml")
    if not path.is_absolute() and not path.is_file():
        path = CONFIG_DIR / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def processed_dir(data_cfg: dict, mode: str | None = None) -> Path:
    """Directory holding cached artefacts for the given mode (dev/full)."""
    mode = mode or data_cfg["mode"]
    root = os.environ.get("TRACEX_DATA_DIR") or data_cfg["processed_dir"]
    return resolve_path(root) / mode
