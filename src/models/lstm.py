"""Sequence-only LSTM baseline."""
from src.models.baselines import SequenceBaseline


class LSTMBaseline(SequenceBaseline):
    def __init__(self, dims: dict, **cfg):
        cfg.pop("rnn", None); cfg.pop("conv", None)
        super().__init__(dims, rnn="lstm", **cfg)
