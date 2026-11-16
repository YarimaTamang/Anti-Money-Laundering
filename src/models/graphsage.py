"""Graph-only GraphSAGE baseline, mean aggregator (Hamilton et al., 2017)."""
from src.models.baselines import GraphBaseline


class GraphSAGEBaseline(GraphBaseline):
    def __init__(self, dims: dict, **cfg):
        cfg.pop("rnn", None); cfg.pop("conv", None)
        super().__init__(dims, conv="sage", **cfg)
