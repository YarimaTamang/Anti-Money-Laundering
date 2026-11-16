"""FastAPI service for single-transaction scoring.

uvicorn src.api:app --port 8000
GET  /health
GET  /predict/{tx_id}          score a transaction from the processed dataset (features are built
                               causally from its history, so the transaction must be in the store)
GET  /explain/{tx_id}          Integrated-Gradients history attributions for the transaction
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from src.data.preprocessing import SPLIT_NAMES
from src.graph.temporal_graph import TemporalGraphStore
from src.training.checkpoints import load_checkpoint
from src.utils.config import PROJECT_ROOT, load_yaml, processed_dir

DISCLAIMER = "Decision-support alert, not proof of money laundering."
app = FastAPI(title="TRACE-X AML scoring API", version="1.0")


@lru_cache(1)
def _state():
    mode, exp, seed = "full", "tracex", 0
    store = TemporalGraphStore(processed_dir(load_yaml("data"), mode))
    model, ck = load_checkpoint(PROJECT_ROOT / "checkpoints" / mode / exp / f"seed{seed}.pt")
    m = json.load(open(PROJECT_ROOT / "experiments" / mode / exp / f"seed{seed}" / "metrics.json"))
    tx = np.asarray(store.a["tx_id"])
    order = np.argsort(tx)
    return store, model, ck["extra"]["cfg"], m, tx[order], order


def _eid(tx_id: int) -> int:
    store, _, _, _, tx_sorted, order = _state()
    i = np.searchsorted(tx_sorted, tx_id)
    if i >= len(tx_sorted) or tx_sorted[i] != tx_id:
        raise HTTPException(404, f"transaction {tx_id} not found")
    return int(order[i])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict/{tx_id}")
def predict(tx_id: int):
    store, model, cfg, m, *_ = _state()
    eid = _eid(tx_id)
    b = store.build(np.array([eid]), model.needs, cfg["seq_len"], cfg["k1"], cfg["k2"])
    with torch.no_grad():
        logit = float(model(b)[0])
    p = 1 / (1 + np.exp(-logit))
    pc = 1 / (1 + np.exp(-(m["platt"]["a"] * logit + m["platt"]["b"])))
    thr = m["test"]["threshold"]
    return {"tx_id": tx_id, "split": SPLIT_NAMES[int(store.a["split"][eid])], "score": p, "calibrated_probability": pc,
            "alert": bool(p >= thr), "threshold": thr, "disclaimer": DISCLAIMER}


@app.get("/explain/{tx_id}")
def explain(tx_id: int):
    from src.explainability.common import Explainable
    from src.explainability.integrated_gradients import explain as ig

    store, model, cfg, *_ = _state()
    r = ig(Explainable(model, store, cfg), _eid(tx_id))
    for k in ("att_s", "att_r"):
        r.pop(k)
    r["disclaimer"] = DISCLAIMER
    return r
