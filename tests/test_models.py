"""Model shape, gradient-flow, checkpoint and determinism tests."""
import numpy as np
import pytest
import torch

from src.models.registry import MODELS, build_model
from src.training.checkpoints import load_checkpoint, save_checkpoint

ABLATIONS = [
    {"use_graph": False, "use_temporal": False},
    {"use_sender": False, "use_receiver": False, "use_temporal": False},
    {"time_encoding": "none"},
    {"graph_edge_features": False},
    {"graph_attention": False},
    {"use_sender": False},
    {"use_receiver": False},
    {"fusion": "concat"},
]


def _dims(store):
    return {"edge_dim": store.edge_dim, "node_dim": store.node_dim, "seq_dim": store.seq_dim}


@pytest.fixture(scope="module")
def batch(store):
    rng = np.random.default_rng(0)
    eids = np.sort(rng.choice(store.n, 128, replace=False))
    return store.build(eids, ("seq", "graph"), seq_len=10, k1=10, k2=5)


@pytest.mark.parametrize("arch", list(MODELS))
def test_forward_backward_all_params_get_grad(store, batch, arch):
    torch.manual_seed(0)
    m = build_model(arch, _dims(store))
    out = m(batch)
    assert out.shape == (128,)
    assert torch.isfinite(out).all()
    torch.nn.functional.binary_cross_entropy_with_logits(out, batch["y"]).backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing


@pytest.mark.parametrize("params", ABLATIONS)
def test_tracex_ablations_forward_backward(store, batch, params):
    m = build_model("tracex", _dims(store), params)
    out = m(batch)
    assert out.shape == (128,)
    out.sum().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


@pytest.mark.parametrize("seq_len", [5, 10, 20])
def test_sequence_lengths(store, seq_len):
    b = store.build(np.arange(1000, 1064), ("seq",), seq_len=seq_len)
    assert b["seq_s"].shape == (64, seq_len, store.seq_dim)
    assert (b["len_s"] <= seq_len).all()
    m = build_model("tracex", _dims(store), {"use_graph": False})
    assert m(b).shape == (64,)


def test_gated_fusion_gates_are_trainable(store, batch):
    m = build_model("tracex", _dims(store))
    m(batch).sum().backward()
    assert m.fusion.gate.weight.grad is not None and m.fusion.gate.weight.grad.abs().sum() > 0
    assert m.fusion.last_gates.shape == (128, 4)


def test_checkpoint_roundtrip_and_prediction_consistency(store, batch, tmp_path):
    torch.manual_seed(0)
    m = build_model("tracex", _dims(store)).eval()
    with torch.no_grad():
        a = m(batch)
        a2 = m(batch)
    assert torch.allclose(a, a2), "eval-mode predictions must be deterministic"
    p = tmp_path / "ck.pt"
    save_checkpoint(p, m, "tracex", {}, _dims(store))
    m2, ck = load_checkpoint(p)
    with torch.no_grad():
        b = m2(batch)
    assert torch.allclose(a, b, atol=1e-6)
    assert ck["arch"] == "tracex"


def test_depth_pruning_preserves_center_outputs(store, batch):
    """Layer-2 edge pruning must not change outputs (centres depend only on hop<=remaining layers)."""
    from src.models.layers import EdgeTemporalGATConv

    torch.manual_seed(0)
    m = build_model("tracex", _dims(store)).eval()
    g = m.graph
    with torch.no_grad():
        fast = g(batch)
        h = g.inp(batch["g_x"])
        for conv in g.convs:  # full propagation over all edges in every layer
            h = conv(h, batch["g_edge_index"], batch["g_edge_attr"], batch["g_edge_dt"])
        c = batch["g_center_index"]
        full = torch.cat([h[c[:, 0]], h[c[:, 1]]], -1)
    assert isinstance(g.convs[0], EdgeTemporalGATConv)
    assert torch.allclose(fast, full, atol=1e-5)
