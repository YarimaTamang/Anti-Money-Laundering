"""Baseline architectures (shared implementation; thin per-model modules re-export them).

Sequence-only : LSTM, GRU   - one shared recurrent encoder applied to sender-outgoing and
                              receiver-incoming histories; dt given as log-scaled scalars.
Graph-only    : GCN, GraphSAGE, GAT - two PyG message-passing layers over the SAME causal
                              2-hop temporal computation subgraph used by TRACE-X.
                              GCNConv / SAGEConv do not consume edge attributes; GATConv uses
                              [edge features, direction, log dt] via edge_dim.
Simple hybrid : GAT-LSTM    - GAT encoder + shared LSTM, plain concatenation, no time encoding
                              branch and no gating.
All baselines receive the target-transaction embedding z, like TRACE-X.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from src.models.layers import MLP, SeqEncoder, depth_mask


def _head(in_dim: int, d: int, dropout: float) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))


class SequenceBaseline(nn.Module):
    def __init__(self, dims: dict, rnn: str = "lstm", hidden: int = 64, dropout: float = 0.2, **_):
        super().__init__()
        self.edge_enc = MLP([dims["edge_dim"], hidden, hidden], dropout, final_act=True)
        self.seq = SeqEncoder(dims["seq_dim"], hidden, rnn, time_mode="none")
        self.head = _head(3 * hidden, hidden, dropout)
        self.needs = ("seq",)

    def forward(self, b: dict, emask=None) -> torch.Tensor:
        hs = self.seq(b["seq_s"], b["seq_s_dt"], b["len_s"])
        hr = self.seq(b["seq_r"], b["seq_r_dt"], b["len_r"])
        return self.head(torch.cat([hs, hr, self.edge_enc(b["edge_tgt"])], -1)).squeeze(-1)


class GraphEncoder(nn.Module):
    def __init__(self, dims: dict, conv: str, hidden: int = 64, heads: int = 4, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.conv_type = conv
        self.inp = nn.Linear(dims["node_dim"], hidden)
        e_dim = dims["edge_dim"] + 2  # + direction + log dt
        mk = {
            "gcn": lambda: GCNConv(hidden, hidden),
            "sage": lambda: SAGEConv(hidden, hidden, aggr="mean"),
            "gat": lambda: GATConv(hidden, hidden // heads, heads=heads, edge_dim=e_dim, dropout=dropout / 2),
        }[conv]
        self.convs = nn.ModuleList([mk() for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.out_dim = 2 * hidden

    def forward(self, b: dict) -> torch.Tensor:
        h = self.inp(b["g_x"])
        n_layers = len(self.convs)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            sel = depth_mask(b, n_layers - i)
            ei = b["g_edge_index"][:, sel]
            if self.conv_type == "gat":
                ea = torch.cat([b["g_edge_attr"][sel], (torch.log1p(b["g_edge_dt"][sel]) / 8.0).unsqueeze(-1)], -1)
                m = conv(h, ei, edge_attr=ea)
            else:
                m = conv(h, ei)
            h = norm(h + torch.relu(m))
        c = b["g_center_index"]
        return torch.cat([h[c[:, 0]], h[c[:, 1]]], -1)


class GraphBaseline(nn.Module):
    def __init__(self, dims: dict, conv: str = "gat", hidden: int = 64, heads: int = 4, dropout: float = 0.2, **_):
        super().__init__()
        self.edge_enc = MLP([dims["edge_dim"], hidden, hidden], dropout, final_act=True)
        self.graph = GraphEncoder(dims, conv, hidden, heads, dropout=dropout)
        self.head = _head(self.graph.out_dim + hidden, hidden, dropout)
        self.needs = ("graph",)

    def forward(self, b: dict, emask=None) -> torch.Tensor:
        return self.head(torch.cat([self.graph(b), self.edge_enc(b["edge_tgt"])], -1)).squeeze(-1)


class GATLSTM(nn.Module):
    def __init__(self, dims: dict, hidden: int = 64, heads: int = 4, dropout: float = 0.2, **_):
        super().__init__()
        self.edge_enc = MLP([dims["edge_dim"], hidden, hidden], dropout, final_act=True)
        self.graph = GraphEncoder(dims, "gat", hidden, heads, dropout=dropout)
        self.seq = SeqEncoder(dims["seq_dim"], hidden, "lstm", time_mode="none")
        self.head = _head(self.graph.out_dim + 3 * hidden, hidden, dropout)
        self.needs = ("seq", "graph")

    def forward(self, b: dict, emask=None) -> torch.Tensor:
        hs = self.seq(b["seq_s"], b["seq_s_dt"], b["len_s"])
        hr = self.seq(b["seq_r"], b["seq_r_dt"], b["len_r"])
        return self.head(torch.cat([self.graph(b), hs, hr, self.edge_enc(b["edge_tgt"])], -1)).squeeze(-1)
