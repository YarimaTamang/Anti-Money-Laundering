"""Sender / receiver transaction histories (the inputs to the two LSTMs).

For each transaction e:
* sender_history(src, ts, L)[e]   = eids of the sender's last L OUTGOING transactions with ts < ts[e]
* receiver_history(dst, ts, L)[e] = eids of the receiver's last L INCOMING transactions with ts < ts[e]

Rows are right-aligned (oldest -> newest) and padded with -1. Transactions that share the
target's timestamp are excluded, so same-minute transactions never see each other.
"""
from __future__ import annotations

import numpy as np

from src.graph.causal_index import causal_window, group_block_starts


def directional_history(acct: np.ndarray, ts: np.ndarray, length: int) -> np.ndarray:
    """Last `length` transactions of the same account strictly before each transaction's time."""
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


def receiver_history(dst: np.ndarray, ts: np.ndarray, length: int) -> np.ndarray:
    return directional_history(dst.astype(np.int64), ts.astype(np.int64), length)
