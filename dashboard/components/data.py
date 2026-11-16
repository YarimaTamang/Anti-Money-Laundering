"""Cached data / model access for the dashboard (all values come from real artefacts)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocessing import SPLIT_NAMES, SPLIT_TEST  # noqa: E402
from src.evaluation.metrics import sigmoid  # noqa: E402
from src.graph.temporal_graph import TemporalGraphStore  # noqa: E402
from src.training.checkpoints import load_checkpoint  # noqa: E402
from src.utils.config import load_yaml, processed_dir  # noqa: E402

MODE = "full"
MAIN_EXP, MAIN_SEED = "tracex", 0
EXP_DIR = ROOT / "experiments" / MODE / MAIN_EXP / f"seed{MAIN_SEED}"
REPORTS = ROOT / "reports"


@st.cache_resource(show_spinner="Loading temporal graph store…")
def store() -> TemporalGraphStore:
    s = TemporalGraphStore(processed_dir(load_yaml("data"), MODE))
    s.a["tx_id"] = np.asarray(s.a["tx_id"])
    return s


@st.cache_resource(show_spinner="Loading TRACE-X checkpoint…")
def model():
    torch.set_num_threads(4)
    m, ck = load_checkpoint(ROOT / "checkpoints" / MODE / MAIN_EXP / f"seed{MAIN_SEED}.pt")
    for p in m.parameters():
        p.requires_grad_(False)
    return m, ck["extra"]["cfg"]


@st.cache_resource
def accounts() -> np.ndarray:
    return pd.read_parquet(store().dir / "accounts.parquet")["account"].to_numpy()


@st.cache_resource
def txid_index() -> tuple[np.ndarray, np.ndarray]:
    t = store().a["tx_id"]
    o = np.argsort(t)
    return t[o], o


def eid_from_txid(tx_id: int) -> int | None:
    t, o = txid_index()
    i = np.searchsorted(t, tx_id)
    return int(o[i]) if i < len(t) and t[i] == tx_id else None


@st.cache_resource
def account_index() -> dict:
    """CSR-style index: account -> eids where it is sender / receiver."""
    s = store()
    out = {}
    for name in ("src", "dst"):
        a = s.a[name]
        o = np.argsort(a, kind="stable")
        ptr = np.searchsorted(a[o], np.arange(len(accounts()) + 1))
        out[name] = (o, ptr)
    return out


def account_eids(acc: int, direction: str) -> np.ndarray:
    o, ptr = account_index()[direction]
    return o[ptr[acc]:ptr[acc + 1]]


@st.cache_resource(show_spinner="Loading transaction table…")
def transactions() -> pd.DataFrame:
    cols = ["tx_id", "timestamp", "src", "dst", "from_bank", "to_bank", "amount_paid", "currency_paid", "amount_received",
            "currency_received", "payment_format", "usd_paid", "label", "split"]
    return pd.read_parquet(store().dir / "transactions.parquet", columns=cols)


@st.cache_resource
def metrics() -> dict:
    return json.load(open(EXP_DIR / "metrics.json"))


@st.cache_resource(show_spinner="Loading test-set scores…")
def test_scores() -> pd.DataFrame:
    s = store()
    te = s.split_eids(SPLIT_TEST)
    logits = np.load(EXP_DIR / "preds_test.npy")
    m = metrics()
    df = pd.DataFrame({"eid": te, "tx_id": s.a["tx_id"][te], "prob": sigmoid(logits),
                       "prob_cal": 1 / (1 + np.exp(-(m["platt"]["a"] * logits + m["platt"]["b"]))),
                       "label": s.a["label"][te]})
    oc = EXP_DIR / "conformal_test_outcome.npy"
    if oc.exists():
        df["conformal"] = np.load(oc)
    return df


@st.cache_resource
def test_prob_lookup() -> np.ndarray:
    """Dense array eid -> test probability (nan outside the test split)."""
    s = store()
    p = np.full(s.n, np.nan, np.float32)
    ts = test_scores()
    p[ts["eid"].to_numpy()] = ts["prob"].to_numpy()
    return p


def score_eids(eids: np.ndarray) -> np.ndarray:
    m, cfg = model()
    b = store().build(np.asarray(eids), m.needs, cfg["seq_len"], cfg["k1"], cfg["k2"])
    with torch.no_grad():
        return torch.sigmoid(m(b)).numpy()


def conformal_cfg() -> dict | None:
    f = EXP_DIR / "conformal.json"
    return json.load(open(f)) if f.exists() else None


def outcome_for(prob: float) -> tuple[str, str]:
    c = conformal_cfg()
    if c is None:
        return ("n/a", "review")
    n_ok, s_ok = prob <= c["t_normal"], prob >= c["t_suspicious"]
    if s_ok and not n_ok:
        return ("High-confidence suspicious alert", "alert")
    if n_ok and not s_ok:
        return ("High-confidence normal", "ok")
    return ("Uncertain - route to human review", "review")


def split_name(eid: int) -> str:
    return SPLIT_NAMES[int(store().a["split"][eid])]


@st.cache_resource
def profile() -> dict | None:
    f = REPORTS / "profiling" / "profile.json"
    return json.load(open(f)) if f.exists() else None


@st.cache_data
def results() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    a, r = REPORTS / "results" / "aggregate.csv", REPORTS / "results" / "all_runs.csv"
    return (pd.read_csv(a, index_col=0) if a.exists() else None, pd.read_csv(r) if r.exists() else None)


@st.cache_data
def history(exp: str, seed: int = 0) -> pd.DataFrame | None:
    f = ROOT / "experiments" / MODE / exp / f"seed{seed}" / "history.csv"
    return pd.read_csv(f) if f.exists() else None


@st.cache_data
def cases() -> list[dict]:
    d = REPORTS / "explanations" / "cases"
    out = []
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        j = json.load(open(f))
        out.append({"tx_id": int(f.stem), "kind": j["kind"], "prob": j["summary"]["probability"], "path": str(f)})
    return sorted(out, key=lambda c: -c["prob"])


def case(tx_id: int) -> dict | None:
    f = REPORTS / "explanations" / "cases" / f"{tx_id}.json"
    return json.load(open(f)) if f.exists() else None


@st.cache_data
def json_report(rel: str) -> dict | None:
    f = REPORTS / rel
    return json.load(open(f)) if f.exists() else None
