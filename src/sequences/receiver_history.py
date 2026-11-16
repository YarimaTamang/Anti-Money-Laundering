"""Receiver incoming-transaction histories.

hist[e, :] holds the eids of the receiver's most recent INCOMING transactions with
timestamp strictly before ts[e] (right-aligned, -1 padded, oldest -> newest).
"""
from __future__ import annotations

import numpy as np

from src.sequences.sender_history import directional_history


def receiver_history(dst: np.ndarray, ts: np.ndarray, length: int) -> np.ndarray:
    return directional_history(dst.astype(np.int64), ts.astype(np.int64), length)
