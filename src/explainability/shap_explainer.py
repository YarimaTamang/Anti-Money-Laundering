"""Feature-level explanations with Kernel SHAP (Lundberg & Lee, NeurIPS 2017).

Explainer: ``shap.KernelExplainer`` (model-agnostic). Why Kernel SHAP:
  * TRACE-X mixes LSTMs, PyG message passing over a per-instance subgraph and gating; Deep/
    Gradient explainers need a single differentiable tensor input and cannot perturb a
    variable-size graph batch, whereas Kernel SHAP only needs a black-box function.
Feature space (31 interpretable features): the target transaction's own attributes
(categoricals kept as single features), the sender and receiver historical account states
(as of strictly before the transaction) and time-of-day / day-of-week.
The instance's history sequences and graph neighbourhood are held FIXED at their actual
values (they are explained by Integrated Gradients and GNNExplainer instead), so SHAP
values answer: "given this transaction's history and neighbourhood, how much did each
tabular feature move the logit relative to a training-data background?"
Limitations: Kernel SHAP assumes feature independence when sampling replacements
(interventional SHAP), is approximate (finite nsamples) and costs O(nsamples x background)
model evaluations per instance, so only a sample of transactions is explained.
Values are in logit units.
"""
from __future__ import annotations

import numpy as np
import torch

from src.explainability.common import Explainable

FMT_SLICE = None


class TabularView:
    def __init__(self, ex: Explainable):
        self.ex, st = ex, ex.store
        names = st.edge_names
        self.fmt_names = [n[4:] for n in names if n.startswith("fmt=")]
        self.cur_names = [n[4:] for n in names if n.startswith("cur=")]
        self.nf, self.nc = len(self.fmt_names), len(self.cur_names)
        self.names = (["log_usd_paid", "log_amount_paid_orig", "payment_format", "currency", "cross_bank",
                       "currency_mismatch", "self_loop"]
                      + [f"sender_{n}" for n in st.node_names] + [f"receiver_{n}" for n in st.node_names]
                      + ["hour_of_day", "day_of_week"])
        self.categorical = {"payment_format": self.fmt_names, "currency": self.cur_names}

    def features(self, eids: np.ndarray) -> np.ndarray:
        st = self.ex.store
        eids = np.atleast_1d(eids)
        ef = np.asarray(st.a["edge_feat"][eids])
        ns = np.asarray(st.a["node_state"][eids])
        nf, nc = self.nf, self.nc
        tod = (st.a["ts"][eids] % 1440) / 60.0
        return np.column_stack([
            ef[:, 0], ef[:, 1], ef[:, 2:2 + nf].argmax(1), ef[:, 2 + nf:2 + nf + nc].argmax(1), ef[:, 2 + nf + nc:],
            ns[:, 0], ns[:, 1], tod, st.a["dow"][eids],
        ]).astype(np.float64)

    def apply(self, b: dict, X: np.ndarray) -> dict:
        """Write tabular features X (raw space) into a batch built for the same transaction."""
        st, nf, nc = self.ex.store, self.nf, self.nc
        m = len(X)
        ef = np.zeros((m, st.edge_dim), np.float32)
        ef[:, 0], ef[:, 1] = X[:, 0], X[:, 1]
        ef[np.arange(m), 2 + np.clip(np.rint(X[:, 2]), 0, nf - 1).astype(int)] = 1
        ef[np.arange(m), 2 + nf + np.clip(np.rint(X[:, 3]), 0, nc - 1).astype(int)] = 1
        ef[:, 2 + nf + nc:] = X[:, 4:7]
        k = st.node_dim
        ns = np.stack([X[:, 7:7 + k], X[:, 7 + k:7 + 2 * k]], 1).astype(np.float32)
        nsn = (ns - st.node_mean) / st.node_std
        b = dict(b)
        b["edge_tgt"] = torch.from_numpy((ef - st.edge_mean) / st.edge_std)
        b["node_c"] = torch.from_numpy(nsn)
        b["tgt_dt_last"] = torch.from_numpy(np.expm1(ns[:, :, -2]).round().astype(np.float32))
        b["tgt_has_hist"] = torch.from_numpy(ns[:, :, -1].copy())
        tod = X[:, -2] / 24 * 2 * np.pi
        dow = X[:, -1] / 7 * 2 * np.pi
        b["tgt_time"] = torch.from_numpy(np.stack([np.sin(tod), np.cos(tod), np.sin(dow), np.cos(dow)], 1).astype(np.float32))
        gx = b["g_x"].clone()
        c = b["g_center_index"]
        gx[c[:, 0]] = torch.from_numpy(nsn[:, 0])
        gx[c[:, 1]] = torch.from_numpy(nsn[:, 1])
        b["g_x"] = gx
        return b


def make_predict_fn(view: TabularView, eid: int, chunk: int = 512):
    ex = view.ex
    cache: dict[int, dict] = {}

    def f(X: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(X), chunk):
            xs = X[i:i + chunk]
            if len(xs) not in cache:
                cache[len(xs)] = ex.batch(np.repeat(eid, len(xs)), meta=False)
            with torch.no_grad():
                out.append(ex.model(view.apply(cache[len(xs)], xs)).numpy())
        return np.concatenate(out)

    return f


def background(view: TabularView, n: int = 25, seed: int = 0) -> np.ndarray:
    """Background = random TRAINING transactions (reference distribution)."""
    from src.data.preprocessing import SPLIT_TRAIN

    tr = view.ex.store.split_eids(SPLIT_TRAIN)
    return view.features(np.random.default_rng(seed).choice(tr, n, replace=False))


def explain(view: TabularView, eid: int, bg: np.ndarray, nsamples: int = 400, seed: int = 0) -> dict:
    import shap

    np.random.seed(seed)
    f = make_predict_fn(view, eid)
    x = view.features(np.array([eid]))
    # sanity: the wrapper must reproduce the model's actual output for the unmodified instance
    actual = float(view.ex.model(view.ex.batch(eid, meta=False)).detach())
    assert abs(float(f(x)[0]) - actual) < 1e-3, "SHAP wrapper does not reproduce model output"
    ke = shap.KernelExplainer(f, bg, link="identity")
    sv = ke.shap_values(x, nsamples=nsamples, silent=True)
    sv = np.asarray(sv).reshape(-1)
    return {"eid": int(eid), "names": view.names, "x": x[0].tolist(), "shap": sv.tolist(),
            "expected_value": float(np.asarray(ke.expected_value).reshape(-1)[0]), "logit": actual,
            "additivity_gap": float(sv.sum() + float(np.asarray(ke.expected_value).reshape(-1)[0]) - actual)}


def feature_fidelity(view: TabularView, eid: int, shap_vals: np.ndarray, bg: np.ndarray, k: int = 5,
                     n_random: int = 20, seed: int = 0) -> dict:
    """Replace the top-k SHAP features supporting the prediction by the background median vs.
    random-k features; reports the logit drop in the direction of the predicted class."""
    f = make_predict_fn(view, eid)
    x = view.features(np.array([eid]))[0]
    lg = lambda z: float(f(z[None])[0])  # noqa: E731
    l0 = lg(x)
    ref = np.median(bg, 0)
    sign = 1.0 if l0 >= 0 else -1.0
    top = np.argsort(-sign * shap_vals)[:k]

    def drop(idx):
        z = x.copy(); z[idx] = ref[idx]
        return sign * (l0 - lg(z))

    rng = np.random.default_rng(seed)
    rand = [drop(rng.choice(len(x), k, replace=False)) for _ in range(n_random)]
    a = np.abs(shap_vals)
    return {"logit": l0, "k": k, "drop_topk": float(drop(top)), "drop_random_k": float(np.mean(rand)),
            "sparsity_top5_mass": float(np.sort(a)[::-1][:5].sum() / max(a.sum(), 1e-12))}  # noqa: E501
