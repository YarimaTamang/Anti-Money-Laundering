"""Sender outgoing-transaction histories.

hist[e, :] holds the eids of the sender's most recent OUTGOING transactions with
timestamp strictly before ts[e] (right-aligned, -1 padded, oldest -> newest).
"""
from __future__ import annotations

import numpy as np

from src.graph.causal_index import causal_window, group_block_starts


def directional_history(acct: np.ndarray, ts: np.ndarray, length: int) -> np.ndarray:
    n = len(acct)
    eid = np.arange(n, dtype=np.int64)
    order = np.lexsort((eid, ts, acct))
    gs, bs = group_block_starts(acct[order], ts[order])
    win = causal_window(gs, bs, eid[order].astype(np.int32), length)
    out = np.empty_like(win)
    out[order] = win
    return out


def sender_history(src: np.ndarray, ts: np.ndarray, length: int) -> np.ndarray:
    return directional_history(src.astype(np.int64), ts.astype(np.int64), length)
