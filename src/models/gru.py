"""Sequence-only GRU baseline (reset/update gates)."""
from src.models.baselines import SequenceBaseline


class GRUBaseline(SequenceBaseline):
    def __init__(self, dims: dict, **cfg):
        cfg.pop("rnn", None); cfg.pop("conv", None)
        super().__init__(dims, rnn="gru", **cfg)
