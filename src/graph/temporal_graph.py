"""Cached temporal-graph store and causal mini-batch construction.

For a batch of target transactions the store assembles, using only information
strictly before each target's timestamp:
  * target transaction features + time-of-day/day-of-week
  * sender / receiver account states
  * sender outgoing and receiver incoming histories (left-aligned for packing)
  * a bounded 2-hop temporal computation subgraph around both endpoints in PyG
    format (edge_index child -> parent), with edge attributes and time deltas.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.graph.features import NODE_HAS_HIST_IDX, NODE_TSL_IDX

ARRAYS = ["ts", "src", "dst", "label", "split", "tx_id", "dow", "edge_feat", "node_state", "hist_s", "hist_r", "nbr"]


class TemporalGraphStore:
    def __init__(self, proc_dir: str | Path, mmap: bool = True):
        self.dir = Path(proc_dir)
        mode = "r" if mmap else None
        self.a = {k: np.load(self.dir / f"{k}.npy", mmap_mode=mode) for k in ARRAYS}
        # Small arrays fully in memory for fast random access.
        for k in ["ts", "src", "dst", "label", "split", "dow"]:
            self.a[k] = np.asarray(self.a[k])
        with open(self.dir / "stats.json") as f:
            self.stats = json.load(f)
        with open(self.dir / "meta.json") as f:
            self.meta = json.load(f)
        self.edge_mean = np.asarray(self.stats["edge"]["mean"], np.float32)
        self.edge_std = np.asarray(self.stats["edge"]["std"], np.float32)
        self.node_mean = np.asarray(self.stats["node"]["mean"], np.float32)
        self.node_std = np.asarray(self.stats["node"]["std"], np.float32)
        self.edge_names = self.stats["edge_feature_names"]
        self.node_names = self.stats["node_feature_names"]
        self.n = len(self.a["ts"])
        self.max_seq_len = self.a["hist_s"].shape[1]
        self.max_neighbors = self.a["nbr"].shape[2]

    # ------------------------------------------------------------------ dims
    @property
    def edge_dim(self) -> int:
        return len(self.edge_names)

    @property
    def node_dim(self) -> int:
        return len(self.node_names)

    @property
    def seq_dim(self) -> int:
        return self.edge_dim + 1  # + same-counterparty flag

    def split_eids(self, split_code: int) -> np.ndarray:
        return np.flatnonzero(self.a["split"] == split_code)

    # -------------------------------------------------------------- features
    def edge_x(self, eids: np.ndarray) -> np.ndarray:
        x = np.asarray(self.a["edge_feat"][eids], np.float32)
        return (x - self.edge_mean) / self.edge_std

    def node_x(self, eids: np.ndarray, sides: np.ndarray) -> np.ndarray:
        x = np.asarray(self.a["node_state"][eids, sides], np.float32)
        return (x - self.node_mean) / self.node_std

    def node_raw(self, eids: np.ndarray, sides: np.ndarray) -> np.ndarray:
        return np.asarray(self.a["node_state"][eids, sides], np.float32)

    # -------------------------------------------------------------- sequences
    def _sequence(self, eids: np.ndarray, hist: np.ndarray, length: int, cp_target: np.ndarray, cp_arr: np.ndarray):
        h = np.asarray(hist[eids][:, -length:]) if length <= hist.shape[1] else None
        if h is None:
            raise ValueError(f"seq_len {length} > stored {hist.shape[1]}")
        b = len(eids)
        lens = (h >= 0).sum(1)
        ar = np.arange(length)
        col = np.minimum(ar[None, :] + (length - lens)[:, None], length - 1)
        idx = np.take_along_axis(h, col, 1)
        valid = ar[None, :] < lens[:, None]
        idx = np.where(valid, idx, -1)
        safe = np.where(valid, idx, 0)
        feats = self.edge_x(safe.reshape(-1)).reshape(b, length, -1)
        same_cp = (cp_arr[safe] == cp_target[:, None]).astype(np.float32)[..., None]
        feats = np.concatenate([feats, same_cp], -1) * valid[..., None]
        ts = self.a["ts"]
        t_hist = ts[safe]
        dt_tgt = np.where(valid, ts[eids][:, None] - t_hist, 0).astype(np.float32)
        dt_prev = np.zeros_like(dt_tgt)
        dt_prev[:, 1:] = np.where(valid[:, 1:], t_hist[:, 1:] - t_hist[:, :-1], 0)
        return feats.astype(np.float32), np.stack([dt_tgt, dt_prev], -1), lens.astype(np.int64), idx

    # ------------------------------------------------------------------ graph
    def _subgraph(self, eids: np.ndarray, k1: int, k2: int):
        ts, nbr = self.a["ts"], self.a["nbr"]
        b = len(eids)
        nb1 = np.asarray(nbr[eids][:, :, -k1:]) if k1 > 0 else np.full((b, 2, 0), -1, np.int32)
        bi, si, _ = np.nonzero(nb1 >= 0)
        c1 = nb1[nb1 >= 0].astype(np.int64)
        e1, s1 = c1 >> 1, c1 & 1
        n1 = len(e1)
        parent1 = 2 * bi + si
        dt1 = ts[eids[bi]] - ts[e1]
        if k2 > 0 and n1 > 0:
            nb2 = np.asarray(nbr[e1, s1][:, -k2:])
            i1, _ = np.nonzero(nb2 >= 0)
            c2 = nb2[nb2 >= 0].astype(np.int64)
        else:
            i1, c2 = np.zeros(0, np.int64), np.zeros(0, np.int64)
        e2, s2 = c2 >> 1, c2 & 1
        n2 = len(e2)
        parent2 = 2 * b + i1
        dt2 = ts[e1[i1]] - ts[e2]
        node_eid = np.concatenate([np.repeat(eids, 2), e1, e2])
        node_side = np.concatenate([np.tile([0, 1], b), s1, s2])
        x = self.node_x(node_eid, node_side)
        child = np.arange(2 * b, 2 * b + n1 + n2)
        parent = np.concatenate([parent1, parent2])
        edge_eid = np.concatenate([e1, e2])
        # direction relative to the parent account: 1 = parent sent the transaction.
        direction = np.concatenate([s1 == 1, s2 == 1]).astype(np.float32)[:, None]
        edge_attr = np.concatenate([self.edge_x(edge_eid), direction], 1) if len(edge_eid) else np.zeros((0, self.edge_dim + 1), np.float32)
        return {
            "x": x, "edge_index": np.stack([child, parent]).astype(np.int64), "edge_attr": edge_attr.astype(np.float32),
            "edge_dt": np.concatenate([dt1, dt2]).astype(np.float32), "edge_eid": edge_eid,
            "edge_hop": np.concatenate([np.ones(n1, np.int8), np.full(n2, 2, np.int8)]),
            "node_eid": node_eid, "node_side": node_side,
            "center_index": np.arange(2 * b).reshape(b, 2),
        }

    # ------------------------------------------------------------------ batch
    def build(self, eids, needs=("seq", "graph"), seq_len: int = 10, k1: int = 10, k2: int = 5, meta: bool = False) -> dict:
        eids = np.asarray(eids, dtype=np.int64)
        ts = self.a["ts"]
        t = ts[eids]
        tod = (t % 1440) / 1440.0 * 2 * np.pi
        dow = self.a["dow"][eids] / 7.0 * 2 * np.pi
        node_raw = np.asarray(self.a["node_state"][eids], np.float32)  # [B,2,F]
        out = {
            "eid": eids,
            "y": self.a["label"][eids].astype(np.float32),
            "edge_tgt": self.edge_x(eids),
            "node_c": (node_raw - self.node_mean) / self.node_std,
            "tgt_time": np.stack([np.sin(tod), np.cos(tod), np.sin(dow), np.cos(dow)], 1).astype(np.float32),
            # minutes since last activity of sender/receiver (0 if none) + has-history flags
            "tgt_dt_last": np.expm1(node_raw[:, :, NODE_TSL_IDX]).round().astype(np.float32),
            "tgt_has_hist": node_raw[:, :, NODE_HAS_HIST_IDX].astype(np.float32),
        }
        if "seq" in needs:
            src, dst = self.a["src"], self.a["dst"]
            fs, ds, ls, is_ = self._sequence(eids, self.a["hist_s"], seq_len, dst[eids], dst)
            fr, dr, lr, ir = self._sequence(eids, self.a["hist_r"], seq_len, src[eids], src)
            out.update(seq_s=fs, seq_s_dt=ds, len_s=ls, seq_r=fr, seq_r_dt=dr, len_r=lr)
            if meta:
                out.update(seq_s_eid=is_, seq_r_eid=ir)
        if "graph" in needs:
            g = self._subgraph(eids, k1, k2)
            keep = ["x", "edge_index", "edge_attr", "edge_dt", "center_index", "edge_hop"]
            if meta:
                keep += ["edge_eid", "node_eid", "node_side"]
            out.update({f"g_{k}": g[k] for k in keep})
        return {k: (torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v) for k, v in out.items()}

    # --------------------------------------------------------------- tables
    def transactions(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "transactions.parquet")
