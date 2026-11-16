"""Load, clean and chronologically split the IBM AML transactions.

Event-time protocol (used by every downstream component)
--------------------------------------------------------
* Timestamps have minute resolution and many transactions share a minute. The file
  order inside a minute is NOT a valid event order, so transactions with the same
  timestamp are treated as simultaneous.
* A feature for transaction e (timestamp t) may only use transactions with
  timestamp strictly smaller than t.
* Transactions are sorted by (timestamp, original row id); the resulting position is
  the internal edge id ``eid``. ``tx_id`` keeps the original CSV row index.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Received",
    "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering",
]
CANON = ["ts_str", "from_bank", "from_acct", "to_bank", "to_acct", "amount_received",
         "currency_received", "amount_paid", "currency_paid", "payment_format", "label"]

SPLIT_TRAIN, SPLIT_VAL_SELECT, SPLIT_VAL_CAL, SPLIT_TEST = 0, 1, 2, 3
SPLIT_NAMES = {0: "train", 1: "val_select", 2: "val_cal", 3: "test"}


def load_raw(path: str | Path, n_rows: int | None = None) -> pd.DataFrame:
    """Read the raw CSV with identifiers as strings (bank codes have leading zeros)."""
    df = pd.read_csv(path, dtype=str)
    if list(df.columns) != RAW_COLUMNS:
        raise ValueError(f"Unexpected schema: {list(df.columns)}")
    df.columns = CANON
    df["tx_id"] = np.arange(len(df), dtype=np.int64)
    df["label"] = df["label"].astype(np.int8)
    df["amount_received"] = df["amount_received"].astype(np.float64)
    df["amount_paid"] = df["amount_paid"].astype(np.float64)
    ts = pd.to_datetime(df["ts_str"], format="%Y/%m/%d %H:%M")
    df["timestamp"] = ts
    df = df.drop(columns=["ts_str"])
    # Chronological order; tx_id only breaks ties deterministically (ties are simultaneous).
    df = df.sort_values(["timestamp", "tx_id"], kind="stable").reset_index(drop=True)
    if n_rows is not None:
        df = df.iloc[:n_rows].copy()
    return df


@dataclass
class Vocab:
    accounts: list
    payment_formats: list
    currencies: list

    def to_json(self) -> dict:
        return {"payment_formats": self.payment_formats, "currencies": self.currencies}


def encode(df: pd.DataFrame, formats: list | None = None, currencies: list | None = None):
    """Add integer codes. Account id = bank code + '_' + account (avoids collisions)."""
    df = df.copy()
    t0 = df["timestamp"].min().normalize()  # midnight of first day
    df["ts"] = ((df["timestamp"] - t0).dt.total_seconds() // 60).astype(np.int64)
    sender = df["from_bank"] + "_" + df["from_acct"]
    receiver = df["to_bank"] + "_" + df["to_acct"]
    accounts, inv = np.unique(np.concatenate([sender.values, receiver.values]), return_inverse=True)
    n = len(df)
    df["src"] = inv[:n].astype(np.int32)
    df["dst"] = inv[n:].astype(np.int32)
    formats = formats or sorted(df["payment_format"].unique().tolist())
    currencies = currencies or sorted(set(df["currency_paid"]) | set(df["currency_received"]))
    fmap = {f: i for i, f in enumerate(formats)}
    cmap = {c: i for i, c in enumerate(currencies)}
    df["fmt"] = df["payment_format"].map(fmap).astype(np.int8)
    df["cur_paid"] = df["currency_paid"].map(cmap).astype(np.int8)
    df["cur_recv"] = df["currency_received"].map(cmap).astype(np.int8)
    df["cross_bank"] = (df["from_bank"] != df["to_bank"]).astype(np.int8)
    df["self_loop"] = (df["src"] == df["dst"]).astype(np.int8)
    df["hour"] = df["timestamp"].dt.hour.astype(np.int8)
    df["dow"] = df["timestamp"].dt.dayofweek.astype(np.int8)
    return df, Vocab(accounts.tolist(), formats, currencies), t0


def _snap(ts: np.ndarray, pos: int) -> int:
    """Move a split boundary forward to the next timestamp change."""
    pos = int(min(max(pos, 0), len(ts)))
    while 0 < pos < len(ts) and ts[pos] == ts[pos - 1]:
        pos += 1
    return pos


def chronological_split(ts: np.ndarray, train: float, val: float, val_select_fraction: float) -> np.ndarray:
    """60/20/20 split by count on sorted timestamps; the 20% val block is split again."""
    n = len(ts)
    assert np.all(np.diff(ts) >= 0), "timestamps must be sorted"
    b1 = _snap(ts, int(round(n * train)))
    b3 = _snap(ts, int(round(n * (train + val))))
    b2 = _snap(ts, b1 + int(round((b3 - b1) * val_select_fraction)))
    split = np.full(n, SPLIT_TEST, dtype=np.int8)
    split[:b1] = SPLIT_TRAIN
    split[b1:b2] = SPLIT_VAL_SELECT
    split[b2:b3] = SPLIT_VAL_CAL
    return split


def estimate_fx_rates(df: pd.DataFrame, train_mask: np.ndarray, currencies: list) -> dict:
    """Estimate units-per-USD for each currency from TRAINING cross-currency transactions.

    For a transaction paid in currency c and received in USD, rate[c] ~ amount_paid / amount_received.
    Missing direct pairs are resolved through a pivot (EUR) where possible.
    """
    tr = df[train_mask]
    tr = tr[(tr["currency_paid"] != tr["currency_received"]) & (tr["amount_received"] > 0) & (tr["amount_paid"] > 0)]
    ratio = (tr["amount_paid"] / tr["amount_received"]).values  # units paid per unit received
    pairs = pd.DataFrame({"p": tr["currency_paid"].values, "r": tr["currency_received"].values, "x": ratio})
    med = pairs.groupby(["p", "r"])["x"].median()
    rate = {"US Dollar": 1.0}
    for _ in range(3):  # propagate through pivots
        for (p, r), x in med.items():
            if r in rate and p not in rate:
                rate[p] = x * rate[r]
            if p in rate and r not in rate:
                rate[r] = rate[p] / x
    return {c: float(rate.get(c, np.nan)) for c in currencies}


def preprocess(raw_path: str | Path, out_dir: str | Path, data_cfg: dict, n_rows: int | None = None, log=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Reading {raw_path} (n_rows={n_rows})")
    df = load_raw(raw_path, n_rows)
    df, vocab, t0 = encode(df)
    s = data_cfg["split"]
    split = chronological_split(df["ts"].values, s["train"], s["val"], s["val_select_fraction"])
    df["split"] = split
    fx = estimate_fx_rates(df, split == SPLIT_TRAIN, vocab.currencies)
    missing = [c for c, v in fx.items() if not np.isfinite(v)]
    if missing:
        log(f"WARNING: no FX estimate for {missing}; falling back to 1.0")
        fx = {c: (v if np.isfinite(v) else 1.0) for c, v in fx.items()}
    rate_paid = df["currency_paid"].map(fx).values
    rate_recv = df["currency_received"].map(fx).values
    df["usd_paid"] = df["amount_paid"].values / rate_paid
    df["usd_received"] = df["amount_received"].values / rate_recv

    keep = ["tx_id", "timestamp", "ts", "from_bank", "from_acct", "to_bank", "to_acct", "src", "dst",
            "amount_paid", "currency_paid", "amount_received", "currency_received", "payment_format",
            "fmt", "cur_paid", "cur_recv", "cross_bank", "self_loop", "hour", "dow",
            "usd_paid", "usd_received", "label", "split"]
    df[keep].to_parquet(out_dir / "transactions.parquet", index=False)
    pd.DataFrame({"account": vocab.accounts}).to_parquet(out_dir / "accounts.parquet", index=False)
    meta = {
        "n_transactions": int(len(df)),
        "n_accounts": int(len(vocab.accounts)),
        "t0": str(t0),
        "payment_formats": vocab.payment_formats,
        "currencies": vocab.currencies,
        "fx_units_per_usd_train_estimate": fx,
        "split_counts": {SPLIT_NAMES[k]: int((split == k).sum()) for k in SPLIT_NAMES},
        "split_positives": {SPLIT_NAMES[k]: int(df["label"].values[split == k].sum()) for k in SPLIT_NAMES},
        "split_time_ranges": {
            SPLIT_NAMES[k]: [str(df["timestamp"].values[split == k].min()), str(df["timestamp"].values[split == k].max())]
            for k in SPLIT_NAMES if (split == k).any()
        },
        "event_time_protocol": "features of a transaction use only transactions with strictly earlier timestamps",
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(json.dumps(meta["split_counts"]) + " positives " + json.dumps(meta["split_positives"]))
    return meta
