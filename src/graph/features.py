"""Edge (transaction) features and historical node (account) features.

Node features of an endpoint of transaction e are the account's state computed
from transactions with timestamp strictly before ts[e].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.graph.causal_index import build_events, exclusive_before, group_block_starts

NODE_FEATURES = [
    "log_out_count", "log_in_count", "log_out_volume_usd", "log_in_volume_usd",
    "log_mean_out_usd", "log_mean_in_usd", "log_unique_out_counterparties",
    "log_unique_in_counterparties", "log_tx_per_hour", "log_minutes_since_last", "has_history",
]
NODE_TSL_IDX = NODE_FEATURES.index("log_minutes_since_last")
NODE_HAS_HIST_IDX = NODE_FEATURES.index("has_history")
NODE_CONTINUOUS = [i for i, f in enumerate(NODE_FEATURES) if f != "has_history"]


def edge_feature_names(formats: list, currencies: list) -> list:
    return (["log_usd_paid", "log_amount_paid_orig"]
            + [f"fmt={f}" for f in formats]
            + [f"cur={c}" for c in currencies]
            + ["cross_bank", "currency_mismatch", "self_loop"])


EDGE_CONTINUOUS = [0, 1]


def edge_features(df: pd.DataFrame, n_formats: int, n_currencies: int) -> np.ndarray:
    n = len(df)
    out = np.zeros((n, 2 + n_formats + n_currencies + 3), np.float32)
    out[:, 0] = np.log1p(df["usd_paid"].values)
    out[:, 1] = np.log1p(df["amount_paid"].values)
    out[np.arange(n), 2 + df["fmt"].values.astype(np.int64)] = 1.0
    out[np.arange(n), 2 + n_formats + df["cur_paid"].values.astype(np.int64)] = 1.0
    base = 2 + n_formats + n_currencies
    out[:, base] = df["cross_bank"].values
    out[:, base + 1] = (df["cur_paid"].values != df["cur_recv"].values)
    out[:, base + 2] = df["self_loop"].values
    return out


def node_states(src: np.ndarray, dst: np.ndarray, ts: np.ndarray, usd_paid: np.ndarray, usd_received: np.ndarray) -> np.ndarray:
    """Return [N, 2, F] account states (side 0 = sender, side 1 = receiver) strictly before each transaction."""
    n = len(src)
    a, t, e, sd = build_events(src, dst, ts)
    gs, bs = group_block_starts(a, t)
    is_out = (sd == 0).astype(np.float64)
    is_in = 1.0 - is_out
    amt = np.where(sd == 0, usd_paid[e], usd_received[e])
    cp = np.where(sd == 0, dst[e], src[e]).astype(np.int64)
    # First occurrence of (account, direction, counterparty) in time order.
    pos = np.arange(len(a))
    o2 = np.lexsort((pos, cp, sd, a))
    k_a, k_s, k_c = a[o2], sd[o2], cp[o2]
    first_sorted = np.ones(len(a), bool)
    first_sorted[1:] = (k_a[1:] != k_a[:-1]) | (k_s[1:] != k_s[:-1]) | (k_c[1:] != k_c[:-1])
    is_first = np.zeros(len(a), bool)
    is_first[o2[first_sorted]] = True

    out_cnt = exclusive_before(is_out, gs, bs)
    in_cnt = exclusive_before(is_in, gs, bs)
    out_vol = exclusive_before(amt * is_out, gs, bs)
    in_vol = exclusive_before(amt * is_in, gs, bs)
    uniq_out = exclusive_before(is_first * is_out, gs, bs)
    uniq_in = exclusive_before(is_first * is_in, gs, bs)
    has = bs > gs
    last_t = np.where(has, t[np.maximum(bs - 1, 0)], t)
    tsl = np.where(has, t - last_t, 0)
    age_h = np.where(has, (t - t[gs]) / 60.0, 0.0)
    total = out_cnt + in_cnt
    feats = np.stack([
        np.log1p(out_cnt), np.log1p(in_cnt), np.log1p(out_vol), np.log1p(in_vol),
        np.log1p(np.divide(out_vol, out_cnt, out=np.zeros_like(out_vol), where=out_cnt > 0)),
        np.log1p(np.divide(in_vol, in_cnt, out=np.zeros_like(in_vol), where=in_cnt > 0)),
        np.log1p(uniq_out), np.log1p(uniq_in),
        np.log1p(total / (age_h + 1.0)),
        np.log1p(tsl), has.astype(np.float64),
    ], axis=1).astype(np.float32)
    states = np.zeros((n, 2, len(NODE_FEATURES)), np.float32)
    states[e, sd] = feats
    return states


def fit_standardizer(x: np.ndarray, cols: list) -> dict:
    """Mean/std over the given (training) rows for the selected columns."""
    x = x.reshape(-1, x.shape[-1])
    mean = np.zeros(x.shape[-1], np.float32)
    std = np.ones(x.shape[-1], np.float32)
    mean[cols] = x[:, cols].mean(0)
    std[cols] = x[:, cols].std(0) + 1e-6
    return {"mean": mean.tolist(), "std": std.tolist(), "cols": list(map(int, cols))}
