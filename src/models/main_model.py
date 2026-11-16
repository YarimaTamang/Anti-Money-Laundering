"""TRACE-X: temporal graph-attention + dual-LSTM gated-fusion network (proposed model).

Branches (each can be switched off for ablations):
  graph    - EdgeTemporalGAT encoder over the causal 2-hop temporal neighbourhood
  sender   - LSTM over the sender's outgoing history
  receiver - LSTM over the receiver's incoming history (separate weights)
  temporal - MLP over Phi(time since sender/receiver last activity), has-history flags,
             cyclical time-of-day and day-of-week of the target transaction
The target transaction embedding z (MLP of its own attributes) conditions the gates
and is passed to the classification head in every variant.

Inspired by: GAT (Velickovic et al., 2018), GATv2 scoring (Brody et al., 2022),
functional time encoding of TGAT (Xu et al., 2020). The specific combination -
edge-aware temporal attention over a causal 2-hop neighbourhood + direction-specific
sender/receiver LSTMs + element-wise gated fusion - is this project's design and is
NOT a reproduction of a published architecture.
"""
from __future__ import annotations

import torch
from torch import nn

from src.models.layers import MLP, GatedFusion, SeqEncoder, TemporalGraphEncoder
from src.sequences.temporal_encoding import TimeEncoder

DEFAULTS = dict(hidden=64, heads=4, time_dim=16, dropout=0.2, gnn_layers=2,
                use_graph=True, use_sender=True, use_receiver=True, use_temporal=True,
                time_encoding="learnable", graph_edge_features=True, graph_attention=True, fusion="gated")


class TraceX(nn.Module):
    def __init__(self, dims: dict, **cfg):
        super().__init__()
        c = {**DEFAULTS, **cfg}
        self.cfg = c
        d, tm = c["hidden"], c["time_encoding"]
        self.edge_enc = MLP([dims["edge_dim"], d, d], dropout=c["dropout"], final_act=True)
        reps: list[int] = []
        self.branches: list[str] = []
        if c["use_graph"]:
            self.graph = TemporalGraphEncoder(dims["node_dim"], dims["edge_dim"] + 1, d, layers=c["gnn_layers"],
                                              heads=c["heads"], time_mode=tm, time_dim=c["time_dim"],
                                              use_edge=c["graph_edge_features"], attention=c["graph_attention"],
                                              dropout=c["dropout"] / 2)
            self.graph_out = nn.Linear(self.graph.out_dim, d)
            reps.append(d); self.branches.append("graph")
        if c["use_sender"]:
            self.sender = SeqEncoder(dims["seq_dim"], d, "lstm", tm, c["time_dim"])
            reps.append(d); self.branches.append("sender")
        if c["use_receiver"]:
            self.receiver = SeqEncoder(dims["seq_dim"], d, "lstm", tm, c["time_dim"])
            reps.append(d); self.branches.append("receiver")
        # Without temporal encoding (time_encoding="none") the temporal branch is removed as well.
        self.use_temporal = c["use_temporal"] and tm != "none"
        if self.use_temporal:
            self.tenc = TimeEncoder(c["time_dim"], tm)
            self.temporal = MLP([2 * self.tenc.out_dim + 2 + 4, d, d], dropout=c["dropout"], final_act=True)
            reps.append(d); self.branches.append("temporal")
        assert reps, "at least one branch required"
        if c["fusion"] == "gated":
            self.fusion = GatedFusion(reps, d, d)
            head_in = d + d
        elif c["fusion"] == "concat":
            self.fusion = None
            head_in = sum(reps) + d
        else:
            raise ValueError(c["fusion"])
        self.head = nn.Sequential(nn.Linear(head_in, d), nn.ReLU(), nn.Dropout(c["dropout"]), nn.Linear(d, 1))
        self.needs = tuple(n for n, on in (("seq", c["use_sender"] or c["use_receiver"]), ("graph", c["use_graph"])) if on)

    def branch_reps(self, b: dict, emask=None) -> tuple[list[torch.Tensor], torch.Tensor]:
        z = self.edge_enc(b["edge_tgt"])
        reps = []
        for name in self.branches:
            if name == "graph":
                reps.append(torch.relu(self.graph_out(self.graph(b, emask=emask))))
            elif name == "sender":
                reps.append(self.sender(b["seq_s"], b["seq_s_dt"], b["len_s"]))
            elif name == "receiver":
                reps.append(self.receiver(b["seq_r"], b["seq_r_dt"], b["len_r"]))
            elif name == "temporal":
                t = torch.cat([self.tenc(b["tgt_dt_last"][:, 0]), self.tenc(b["tgt_dt_last"][:, 1]),
                               b["tgt_has_hist"], b["tgt_time"]], -1)
                reps.append(self.temporal(t))
        return reps, z

    def forward(self, b: dict, emask=None) -> torch.Tensor:
        reps, z = self.branch_reps(b, emask)
        h = self.fusion(reps, z) if self.fusion is not None else torch.cat(reps, -1)
        return self.head(torch.cat([h, z], -1)).squeeze(-1)
