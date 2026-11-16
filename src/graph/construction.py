"""Directed transaction-graph construction.

* ``build_static_graph``   - full directed multigraph as a PyG ``Data`` object
  (nodes = accounts, edges = transactions, edge ids preserved). Used for graph
  statistics and visual exploration only - never as model input.
* ``snapshot_before``      - causal snapshot containing only edges with ts < t.
* ``temporal_neighbors``   - per (transaction, endpoint) table of the K most recent
  incident transactions strictly before the transaction time. This is the bounded,
  causal neighbourhood that all graph models sample from.

Neighbour code convention: code = 2 * eid' + side', where side' is the side of the
*counterparty* in transaction eid' (0 = sender, 1 = receiver). Because nbr[eid', side']
lists the counterparty's neighbours before ts[eid'], codes can be expanded
recursively to build multi-hop temporal neighbourhoods without ever seeing the future.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from src.graph.causal_index import build_events, causal_window, group_block_starts


def build_static_graph(src, dst, ts, tx_id, edge_attr=None, num_nodes: int | None = None) -> Data:
    data = Data(
        edge_index=torch.as_tensor(np.stack([src, dst]).astype(np.int64)),
        edge_ts=torch.as_tensor(np.asarray(ts, dtype=np.int64)),
        edge_id=torch.as_tensor(np.asarray(tx_id, dtype=np.int64)),
        num_nodes=int(num_nodes if num_nodes is not None else max(src.max(), dst.max()) + 1),
    )
    if edge_attr is not None:
        data.edge_attr = torch.as_tensor(edge_attr)
    return data


def snapshot_before(data: Data, t: int) -> Data:
    """Causal snapshot: only transactions strictly earlier than t."""
    m = data.edge_ts < t
    out = Data(edge_index=data.edge_index[:, m], edge_ts=data.edge_ts[m], edge_id=data.edge_id[m],
               num_nodes=data.num_nodes)
    if getattr(data, "edge_attr", None) is not None:
        out.edge_attr = data.edge_attr[m]
    return out


def temporal_neighbors(src: np.ndarray, dst: np.ndarray, ts: np.ndarray, k: int) -> np.ndarray:
    """Return int32 [N, 2, k] neighbour codes (right-aligned, most recent last, -1 = none)."""
    n = len(src)
    a, t, e, sd = build_events(src, dst, ts)
    gs, bs = group_block_starts(a, t)
    codes = (2 * e + (1 - sd)).astype(np.int32)
    win = causal_window(gs, bs, codes, k)
    nbr = np.full((n, 2, k), -1, np.int32)
    nbr[e, sd] = win
    return nbr


def graph_statistics(src: np.ndarray, dst: np.ndarray, num_nodes: int) -> dict:
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    out_deg = np.bincount(src, minlength=num_nodes)
    in_deg = np.bincount(dst, minlength=num_nodes)
    pairs = np.unique(np.stack([src, dst], 1), axis=0)
    adj = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(num_nodes, num_nodes)).tocsr()
    n_wcc, lab = connected_components(adj, directed=True, connection="weak")
    sizes = np.bincount(lab)
    rev = adj.multiply(adj.T)
    return {
        "num_nodes": int(num_nodes),
        "num_edges_transactions": int(len(src)),
        "num_unique_directed_pairs": int(len(pairs)),
        "self_loop_transactions": int((src == dst).sum()),
        "reciprocal_pair_fraction": float(rev.nnz / max(adj.nnz, 1)),
        "density_unique_pairs": float(len(pairs) / (num_nodes * (num_nodes - 1))),
        "mean_out_degree": float(out_deg.mean()),
        "max_out_degree": int(out_deg.max()),
        "max_in_degree": int(in_deg.max()),
        "weakly_connected_components": int(n_wcc),
        "largest_wcc_size": int(sizes.max()),
        "accounts_only_sending": int(((out_deg > 0) & (in_deg == 0)).sum()),
        "accounts_only_receiving": int(((in_deg > 0) & (out_deg == 0)).sum()),
    }
