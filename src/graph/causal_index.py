"""Vectorised causal-indexing primitives.

All functions operate on rows sorted by (group, ts, ...). For row i we compute
  gs[i] = first row of its group (e.g. account)
  bs[i] = first row of its (group, ts) block
Rows in [gs[i], bs[i]) are exactly the rows of the same group with a STRICTLY
earlier timestamp, so anything derived from that range is leakage-free even when
many transactions share a timestamp.
"""
from __future__ import annotations

import numpy as np


def group_block_starts(group: np.ndarray, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(group)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    pos = np.arange(n, dtype=np.int64)
    new_group = np.empty(n, bool)
    new_group[0] = True
    new_group[1:] = group[1:] != group[:-1]
    new_block = new_group.copy()
    new_block[1:] |= ts[1:] != ts[:-1]
    gs = np.maximum.accumulate(np.where(new_group, pos, 0))
    bs = np.maximum.accumulate(np.where(new_block, pos, 0))
    return gs, bs


def exclusive_before(x: np.ndarray, gs: np.ndarray, bs: np.ndarray) -> np.ndarray:
    """Sum of x over rows of the same group with strictly earlier timestamp."""
    c = np.concatenate([[0], np.cumsum(x, dtype=np.float64)])
    return c[bs] - c[gs]


def causal_window(gs: np.ndarray, bs: np.ndarray, values: np.ndarray, length: int, fill: int = -1) -> np.ndarray:
    """Right-aligned window of the `length` most recent strictly-earlier values.

    Column length-1 holds the most recent earlier value; missing entries are `fill`.
    """
    n = len(gs)
    out = np.full((n, length), fill, dtype=values.dtype)
    for j in range(length):
        src = bs - (length - j)
        ok = src >= gs
        out[ok, j] = values[src[ok]]
    return out


def build_events(src: np.ndarray, dst: np.ndarray, ts: np.ndarray):
    """Two events per transaction: (sender, side 0) and (receiver, side 1), sorted by (account, ts, eid, side)."""
    n = len(src)
    acct = np.concatenate([src, dst]).astype(np.int64)
    t = np.concatenate([ts, ts]).astype(np.int64)
    eid = np.concatenate([np.arange(n), np.arange(n)]).astype(np.int64)
    side = np.concatenate([np.zeros(n, np.int8), np.ones(n, np.int8)])
    order = np.lexsort((side, eid, t, acct))
    return acct[order], t[order], eid[order], side[order]
