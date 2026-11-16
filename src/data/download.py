"""Locate or download the IBM AML HI-Small files.

Resolution order for the transactions CSV:
  1. ``TRACEX_RAW_FILE`` environment variable (absolute path to HI-Small_Trans.csv)
  2. ``<raw_dir>/HI-Small_Trans.csv`` (manual download)
  3. kagglehub download (public dataset; if your account requires auth set
     ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` in the environment or a .env file).
Credentials are never written to disk by this module.
"""
from __future__ import annotations

import os
from pathlib import Path

from src.utils.config import load_yaml, resolve_path


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(resolve_path(".env"))
    except Exception:
        pass


def resolve_file(data_cfg: dict, key: str = "transactions_file", allow_download: bool = True) -> Path:
    _load_dotenv()
    fname = data_cfg["dataset"][key]
    env = os.environ.get("TRACEX_RAW_FILE") if key == "transactions_file" else None
    if env and Path(env).exists():
        return Path(env)
    local = resolve_path(data_cfg["dataset"]["raw_dir"]) / fname
    if local.exists():
        return local
    if not allow_download:
        raise FileNotFoundError(f"{fname} not found in {local.parent}; set TRACEX_RAW_FILE or download manually.")
    import kagglehub

    path = kagglehub.dataset_download(data_cfg["dataset"]["kaggle_handle"], path=fname)
    return Path(path)


if __name__ == "__main__":
    cfg = load_yaml("data")
    for k in ("transactions_file", "patterns_file"):
        print(k, "->", resolve_file(cfg, k))
