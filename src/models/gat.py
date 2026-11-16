"""Graph-only GAT baseline with edge features (Velickovic et al., 2018)."""
from src.models.baselines import GraphBaseline


class GATBaseline(GraphBaseline):
    def __init__(self, dims: dict, **cfg):
        cfg.pop("rnn", None); cfg.pop("conv", None)
        super().__init__(dims, conv="gat", **cfg)
