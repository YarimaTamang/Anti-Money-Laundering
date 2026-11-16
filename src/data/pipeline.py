"""End-to-end cached preprocessing: raw CSV -> processed arrays for one mode.

Usage:  python -m src.data.pipeline --mode dev|full [--force]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.data.download import resolve_file
from src.data.preprocessing import SPLIT_TRAIN, preprocess
from src.graph.construction import temporal_neighbors
from src.graph.features import (EDGE_CONTINUOUS, NODE_CONTINUOUS, NODE_FEATURES, edge_feature_names,
                                edge_features, fit_standardizer, node_states)
from src.sequences.receiver_history import receiver_history
from src.sequences.sender_history import sender_history
from src.utils.config import load_yaml, processed_dir
from src.utils.logging import get_logger


def run(mode: str, force: bool = False) -> None:
    cfg = load_yaml("data")
    out = processed_dir(cfg, mode)
    log = get_logger("pipeline", out / "pipeline.log")
    if (out / "DONE").exists() and not force:
        log.info(f"Cached artefacts found in {out}; use --force to rebuild")
        return
    t_all = time.time()
    raw = resolve_file(cfg, "transactions_file")
    n_rows = cfg["dev"]["n_rows"] if mode == "dev" else None
    meta = preprocess(raw, out, cfg, n_rows=n_rows, log=log.info)
    df = pd.read_parquet(out / "transactions.parquet")
    src, dst = df["src"].values.astype(np.int64), df["dst"].values.astype(np.int64)
    ts = df["ts"].values.astype(np.int64)
    split = df["split"].values
    train = split == SPLIT_TRAIN

    t = time.time()
    ef = edge_features(df, len(meta["payment_formats"]), len(meta["currencies"]))
    log.info(f"edge features {ef.shape} in {time.time()-t:.1f}s")
    t = time.time()
    ns = node_states(src, dst, ts, df["usd_paid"].values, df["usd_received"].values)
    log.info(f"node states {ns.shape} in {time.time()-t:.1f}s")
    L, K = cfg["history"]["max_seq_len"], cfg["history"]["max_neighbors"]
    t = time.time()
    hs = sender_history(src, ts, L)
    hr = receiver_history(dst, ts, L)
    log.info(f"histories {hs.shape} in {time.time()-t:.1f}s")
    t = time.time()
    nbr = temporal_neighbors(src, dst, ts, K)
    log.info(f"temporal neighbours {nbr.shape} in {time.time()-t:.1f}s")

    # Standardisers are fitted on TRAINING transactions only.
    stats = {
        "edge": fit_standardizer(ef[train], EDGE_CONTINUOUS),
        "node": fit_standardizer(ns[train], NODE_CONTINUOUS),
        "edge_feature_names": edge_feature_names(meta["payment_formats"], meta["currencies"]),
        "node_feature_names": NODE_FEATURES,
        "fitted_on": "train",
        "n_train_rows": int(train.sum()),
    }
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    arrays = {
        "ts": ts, "src": src.astype(np.int32), "dst": dst.astype(np.int32), "label": df["label"].values.astype(np.int8),
        "split": split.astype(np.int8), "tx_id": df["tx_id"].values.astype(np.int64), "dow": df["dow"].values.astype(np.int8),
        "edge_feat": ef, "node_state": ns, "hist_s": hs, "hist_r": hr, "nbr": nbr,
    }
    for k, v in arrays.items():
        np.save(out / f"{k}.npy", v)
    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    log.info(f"Finished {mode} preprocessing in {time.time()-t_all:.1f}s -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=None, choices=["dev", "full"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(a.mode or load_yaml("data")["mode"], a.force)
