"""Integrated Gradients (Sundararajan et al., ICML 2017) over sender / receiver history sequences.

IG_i(x) = (x_i - x'_i) * integral_0^1 dF(x' + a (x - x')) / dx_i  da     (Riemann/trapezoid, n_steps)
Baseline x' = 0 in standardised feature space (training mean for amounts; "no category" for
one-hot columns). Only the sequence feature tensors are interpolated; the graph, temporal and
target-transaction inputs stay at their actual values. Attributions are on the logit.
A Captum implementation was not used because Captum expands additional inputs along the batch
axis, which is invalid for a PyG graph batch; this implementation instead replicates the
causal subgraph per interpolation step via the store. The completeness axiom
sum(IG) = F(x) - F(x') is reported as a check (``delta``).
"""
from __future__ import annotations

import numpy as np
import torch

from src.explainability.common import Explainable


def explain(ex: Explainable, eid: int, n_steps: int = 64) -> dict:
    st = ex.store
    b1 = ex.batch(eid)
    alphas = torch.linspace(0, 1, n_steps + 1)
    bb = ex.batch(np.repeat(eid, n_steps + 1), meta=False)
    xs, xr = b1["seq_s"][0], b1["seq_r"][0]
    seq_s = (alphas.view(-1, 1, 1) * xs).requires_grad_(True)
    seq_r = (alphas.view(-1, 1, 1) * xr).requires_grad_(True)
    bb["seq_s"], bb["seq_r"] = seq_s, seq_r
    logits = ex.model(bb)
    gs, gr = torch.autograd.grad(logits.sum(), [seq_s, seq_r])
    # trapezoid rule
    w = torch.ones(n_steps + 1); w[0] = w[-1] = 0.5; w = w / n_steps
    att_s = (xs * (gs * w.view(-1, 1, 1)).sum(0)).numpy()
    att_r = (xr * (gr * w.view(-1, 1, 1)).sum(0)).numpy()
    f_x, f_base = float(logits[-1].detach()), float(logits[0].detach())
    delta = float(att_s.sum() + att_r.sum() - (f_x - f_base))
    names = st.edge_names + ["same_counterparty_as_target"]

    def rows(att, seq_eid, length, side):
        out = []
        for j in range(int(length)):
            e = int(seq_eid[j])
            out.append({
                "position": j, "eid": e, "tx_id": int(st.a["tx_id"][e]),
                "minutes_before_target": int(st.a["ts"][eid] - st.a["ts"][e]),
                "counterparty_account": int(st.a["dst"][e] if side == "sender" else st.a["src"][e]),
                "label": int(st.a["label"][e]),
                "attribution": float(att[j].sum()),
                "top_features": sorted(((names[i], float(att[j, i])) for i in np.argsort(-np.abs(att[j]))[:3]), key=lambda t: -abs(t[1])),
            })
        return out

    return {
        "eid": int(eid), "tx_id": int(st.a["tx_id"][eid]),
        "logit": f_x, "logit_baseline": f_base, "completeness_delta": delta,
        "sender_steps": rows(att_s, b1["seq_s_eid"][0], b1["len_s"][0], "sender"),
        "receiver_steps": rows(att_r, b1["seq_r_eid"][0], b1["len_r"][0], "receiver"),
        "sender_feature_attr": dict(zip(names, map(float, att_s.sum(0)))),
        "receiver_feature_attr": dict(zip(names, map(float, att_r.sum(0)))),
        "sender_total": float(att_s.sum()), "receiver_total": float(att_r.sum()),
        "att_s": att_s.tolist(), "att_r": att_r.tolist(),
    }


@torch.no_grad()
def step_fidelity(ex: Explainable, eid: int, ig: dict, k: int = 3, n_random: int = 20, seed: int = 0) -> dict:
    """Delete (set to baseline) the top-k attributed history steps vs. random-k steps;
    reports the drop of the logit in the direction of the predicted class."""
    b = ex.batch(eid)
    l0 = float(ex.model(b)[0])
    steps = [("s", r["position"], r["attribution"]) for r in ig["sender_steps"]] + \
            [("r", r["position"], r["attribution"]) for r in ig["receiver_steps"]]
    if not steps:
        return {"logit": l0, "n_steps": 0}
    k = min(k, len(steps))
    sign = 1.0 if l0 >= 0 else -1.0  # evidence for the predicted class
    order = sorted(range(len(steps)), key=lambda i: -sign * steps[i][2])

    def drop(idx):
        bb = dict(b)
        bb["seq_s"], bb["seq_r"] = b["seq_s"].clone(), b["seq_r"].clone()
        for i in idx:
            which, pos, _ = steps[i]
            bb["seq_" + which][0, pos] = 0.0
        return sign * (l0 - float(ex.model(bb)[0]))

    rng = np.random.default_rng(seed)
    rand = [drop(rng.choice(len(steps), k, replace=False)) for _ in range(n_random)]
    att = np.abs([s[2] for s in steps])
    return {"logit": l0, "n_steps": len(steps), "k": k, "drop_topk": drop(order[:k]),
            "drop_random_k": float(np.mean(rand)),
            "sparsity_top3_mass": float(np.sort(att)[::-1][:3].sum() / max(att.sum(), 1e-12))}
