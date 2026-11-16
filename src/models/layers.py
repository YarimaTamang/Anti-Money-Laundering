"""Shared neural building blocks."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, softmax

from src.sequences.temporal_encoding import TimeEncoder


class MLP(nn.Module):
    def __init__(self, dims: list[int], dropout: float = 0.0, final_act: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 or final_act:
                layers += [nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SeqEncoder(nn.Module):
    """Recurrent encoder over a (left-aligned, padded) transaction history.

    Step input = [transaction features, enc(dt to target), enc(dt to previous step)].
    Returns the final hidden state of the last valid step (zeros for empty histories).
    """

    def __init__(self, seq_dim: int, hidden: int, rnn: str = "lstm", time_mode: str = "learnable", time_dim: int = 16,
                 layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.time = TimeEncoder(time_dim, time_mode)
        in_dim = seq_dim + 2 * self.time.out_dim
        cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn]
        self.rnn = cls(in_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.rnn_type = rnn
        self.out_dim = hidden

    def step_inputs(self, seq: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        return torch.cat([seq, self.time(dt[..., 0]), self.time(dt[..., 1])], -1)

    def forward(self, seq: torch.Tensor, dt: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.step_inputs(seq, dt)
        lens = lengths.clamp(min=1).cpu()
        packed = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        if self.rnn_type == "lstm":
            h = h[0]
        out = h[-1]
        return out * (lengths > 0).float().unsqueeze(-1).to(out.dtype)


class EdgeTemporalGATConv(MessagePassing):
    """Edge-aware temporal graph attention (TRACE-X graph layer).

    For a message from neighbour j to account i over transaction (i,j) at time-lag dt:
        u_ij   = [edge_attr_ij || Phi(dt_ij)]
        e_ij^h = a_h^T LeakyReLU(W_q^h x_i + W_k^h x_j + W_e^h u_ij)          (GATv2-style scoring)
        alpha  = softmax_j(e_ij)
        h_i    = LayerNorm( x_i + ELU( W_s x_i + ||_h sum_j alpha_ij^h (W_v^h x_j + W_u^h u_ij) ) )
    Ablations: use_edge=False drops edge_attr from u_ij (keeps Phi(dt));
               attention=False replaces alpha with uniform 1/deg (mean aggregation).
    An optional per-edge mask (0..1) multiplies messages (used by GNNExplainer).
    """

    def __init__(self, dim: int, edge_dim: int, heads: int = 4, time_mode: str = "learnable", time_dim: int = 16,
                 use_edge: bool = True, attention: bool = True, dropout: float = 0.1):
        super().__init__(aggr="add", node_dim=0)
        assert dim % heads == 0
        self.h, self.c = heads, dim // heads
        self.use_edge, self.attention = use_edge, attention
        self.time = TimeEncoder(time_dim, time_mode)
        u_dim = (edge_dim if use_edge else 0) + self.time.out_dim
        self.lin_v = nn.Linear(dim, dim)
        self.lin_s = nn.Linear(dim, dim)
        if attention:
            self.lin_q = nn.Linear(dim, dim, bias=False)
            self.lin_k = nn.Linear(dim, dim, bias=False)
            self.att = nn.Parameter(torch.empty(1, heads, self.c))
            nn.init.xavier_uniform_(self.att)
            self.lin_u = nn.Linear(u_dim, 2 * dim)  # key and value edge projections
        else:
            self.lin_u = nn.Linear(u_dim, dim)  # value edge projection only
        self.norm = nn.LayerNorm(dim)
        self.dropout = dropout
        self.last_alpha: torch.Tensor | None = None

    def forward(self, x, edge_index, edge_attr, edge_dt, emask=None):
        u = self.time(edge_dt)
        if self.use_edge:
            u = torch.cat([edge_attr, u], -1)
        agg = self.propagate(edge_index, x=x, u=u, emask=emask)
        return self.norm(x + F.elu(self.lin_s(x) + agg))

    def message(self, x_i, x_j, u, index, ptr, size_i, emask):
        if self.attention:
            uk, uv = self.lin_u(u).chunk(2, -1)
            q = self.lin_q(x_i).view(-1, self.h, self.c)
            k = (self.lin_k(x_j) + uk).view(-1, self.h, self.c)
            score = (self.att * F.leaky_relu(q + k, 0.2)).sum(-1)
            alpha = softmax(score, index, ptr, size_i)
        else:
            uv = self.lin_u(u)
            deg = degree(index, size_i, dtype=x_j.dtype).clamp(min=1)
            alpha = (1.0 / deg[index]).unsqueeze(-1).expand(-1, self.h)
        self.last_alpha = alpha.detach()
        alpha = F.dropout(alpha, self.dropout, self.training)
        m = (self.lin_v(x_j) + uv).view(-1, self.h, self.c) * alpha.unsqueeze(-1)
        if emask is not None:
            m = m * emask.view(-1, 1, 1)
        return m.reshape(-1, self.h * self.c)


class TemporalGraphEncoder(nn.Module):
    """Two-layer EdgeTemporalGAT over the causal 2-hop computation subgraph -> [h_sender || h_receiver]."""

    def __init__(self, node_dim: int, edge_dim: int, dim: int, layers: int = 2, **conv_kw):
        super().__init__()
        self.inp = nn.Linear(node_dim, dim)
        self.convs = nn.ModuleList([EdgeTemporalGATConv(dim, edge_dim, **conv_kw) for _ in range(layers)])
        self.out_dim = 2 * dim

    def forward(self, b: dict, emask=None) -> torch.Tensor:
        h = self.inp(b["g_x"])
        n_layers = len(self.convs)
        for i, conv in enumerate(self.convs):
            m = depth_mask(b, n_layers - i)
            h = conv(h, b["g_edge_index"][:, m], b["g_edge_attr"][m], b["g_edge_dt"][m],
                     emask=None if emask is None else emask[m])
        c = b["g_center_index"]
        return torch.cat([h[c[:, 0]], h[c[:, 1]]], -1)


def depth_mask(b: dict, max_hop: int) -> torch.Tensor:
    """Edges whose parent is still needed by later layers: with r remaining layers only
    edges of hop <= r matter for the centre outputs (identical result, less compute)."""
    return b["g_edge_hop"] <= max_hop


class GatedFusion(nn.Module):
    """Element-wise sigmoid gates conditioned on all branch representations and the target-transaction embedding.

        g_k = sigmoid(W_g^k [r_1 || ... || r_K || z] + b_g^k)
        f   = sum_k g_k * tanh(W_k r_k + b_k)
    """

    def __init__(self, branch_dims: list[int], ctx_dim: int, dim: int):
        super().__init__()
        self.k = len(branch_dims)
        self.proj = nn.ModuleList([nn.Linear(d, dim) for d in branch_dims])
        self.gate = nn.Linear(sum(branch_dims) + ctx_dim, self.k * dim)
        self.dim = dim
        self.last_gates: torch.Tensor | None = None

    def forward(self, reps: list[torch.Tensor], ctx: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat(reps + [ctx], -1))).view(-1, self.k, self.dim)
        self.last_gates = g.detach().mean(-1)
        proj = torch.stack([torch.tanh(p(r)) for p, r in zip(self.proj, reps)], 1)
        return (g * proj).sum(1)
