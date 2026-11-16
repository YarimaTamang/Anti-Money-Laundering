"""Graph-only GCN baseline (Kipf & Welling, 2017)."""
from src.models.baselines import GraphBaseline


class GCNBaseline(GraphBaseline):
    def __init__(self, dims: dict, **cfg):
        cfg.pop("rnn", None); cfg.pop("conv", None)
        super().__init__(dims, conv="gcn", **cfg)
