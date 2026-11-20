"""Model factory: maps the `arch` names used in configs/models.yaml (and stored in checkpoints) to model classes.

The six baselines are thin configurations of the shared classes in `baselines.py`:
LSTM / GRU differ only in the recurrent cell, GCN / GraphSAGE / GAT only in the graph convolution.
"""
from __future__ import annotations

from torch import nn

from src.models.baselines import GATLSTM, GraphBaseline, SequenceBaseline
from src.models.main_model import TraceX

# arch name -> (class, fixed constructor arguments)
MODELS = {
    "tracex": (TraceX, {}),
    "lstm": (SequenceBaseline, {"rnn": "lstm"}),
    "gru": (SequenceBaseline, {"rnn": "gru"}),
    "gcn": (GraphBaseline, {"conv": "gcn"}),
    "graphsage": (GraphBaseline, {"conv": "sage"}),
    "gat": (GraphBaseline, {"conv": "gat"}),
    "gat_lstm": (GATLSTM, {}),
}


def build_model(arch: str, dims: dict, params: dict | None = None) -> nn.Module:
    cls, fixed = MODELS[arch]
    kwargs = {k: v for k, v in (params or {}).items() if k not in fixed}
    return cls(dims, **fixed, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
