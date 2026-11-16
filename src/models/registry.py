"""Model factory."""
from __future__ import annotations

from torch import nn

from src.models.gat import GATBaseline
from src.models.gat_lstm import GATLSTM
from src.models.gcn import GCNBaseline
from src.models.graphsage import GraphSAGEBaseline
from src.models.gru import GRUBaseline
from src.models.lstm import LSTMBaseline
from src.models.main_model import TraceX

MODELS = {
    "tracex": TraceX, "lstm": LSTMBaseline, "gru": GRUBaseline, "gcn": GCNBaseline,
    "graphsage": GraphSAGEBaseline, "gat": GATBaseline, "gat_lstm": GATLSTM,
}


def build_model(arch: str, dims: dict, params: dict | None = None) -> nn.Module:
    return MODELS[arch](dims, **(params or {}))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
