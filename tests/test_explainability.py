"""Explanation-generation and uncertainty tests (use the dev-mode TRACE-X checkpoint)."""
import numpy as np
import pytest

from src.utils.config import PROJECT_ROOT


@pytest.fixture(scope="module")
def ex(store):
    from src.explainability.common import load_explainable

    if not (PROJECT_ROOT / "checkpoints" / "dev" / "tracex" / "seed0.pt").exists():
        pytest.skip("dev checkpoint missing: python -m src.training.run_experiment --exp tracex --mode dev --max-epochs 1")
    return load_explainable("tracex", 0, "dev", store=store)


@pytest.fixture(scope="module")
def eid(store):
    pos = np.flatnonzero(store.a["label"] == 1)
    return int(pos[len(pos) // 2])


def test_integrated_gradients_completeness(ex, eid):
    from src.explainability import integrated_gradients as ig

    r = ig.explain(ex, eid, n_steps=64)
    assert abs(r["completeness_delta"]) < 1e-2 * max(1.0, abs(r["logit"] - r["logit_baseline"]))
    # all attributed history steps are strictly earlier than the target
    ts = ex.store.a["ts"]
    for s in r["sender_steps"] + r["receiver_steps"]:
        assert ts[s["eid"]] < ts[eid]


def test_gnn_explainer_masks_valid_and_causal(ex, eid):
    from src.explainability import gnn_explainer

    g = gnn_explainer.explain(ex, eid, epochs=30)
    m = np.asarray(g["edge_mask"])
    assert len(m) == g["n_subgraph_edges"]
    assert np.all((m >= 0) & (m <= 1))
    ts = ex.store.a["ts"]
    assert all(ts[e["eid"]] < ts[eid] for e in g["edges"])
    f = gnn_explainer.edge_fidelity(ex, eid, m, n_random=3)
    assert 0 <= f["p"] <= 1


def test_shap_wrapper_reproduces_model_and_is_additive(ex, eid):
    from src.explainability import shap_explainer

    view = shap_explainer.TabularView(ex)
    bg = shap_explainer.background(view, 5)
    r = shap_explainer.explain(view, eid, bg, nsamples=60)
    assert len(r["shap"]) == len(view.names) == 31
    assert abs(r["additivity_gap"]) < 1e-3


def test_masked_forward_identity(ex, eid):
    """Masks of ones must reproduce the unmasked prediction exactly."""
    import torch

    from src.explainability.gnn_explainer import masked_forward

    b = ex.batch(eid)
    with torch.no_grad():
        a = ex.model(b)
        E = b["g_edge_index"].shape[1]
        c = masked_forward(ex, b, torch.ones(E), torch.ones(b["g_edge_attr"].shape[1]), torch.ones(b["g_x"].shape[1]))
    assert torch.allclose(a, c, atol=1e-6)


def test_conformal_class_coverage():
    from src.uncertainty.conformal import MondrianConformal

    rng = np.random.default_rng(0)
    n = 40000
    y = (rng.random(n) < 0.05).astype(int)
    p = np.clip(0.5 * y + 0.5 * rng.random(n), 0, 1)
    cal, te = slice(0, n // 2), slice(n // 2, n)
    cp = MondrianConformal().fit(p[cal], y[cal], alpha=0.1)
    r = cp.report(p[te], y[te])
    assert r["coverage_class0"] >= 0.88 and r["coverage_class1"] >= 0.86
    oc = cp.outcome(p[te])
    assert set(np.unique(oc)) <= {0, 1, 2}


def test_mc_dropout_produces_variance(ex, store):
    from src.uncertainty.mc_dropout import mc_predict

    mu, sd = mc_predict(ex.model, store, np.arange(2000, 2064), ex.cfg, T=5, bs=64)
    assert mu.shape == sd.shape == (64,)
    assert (sd > 0).any()
    assert not ex.model.training
