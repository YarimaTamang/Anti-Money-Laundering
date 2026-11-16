"""Adapted GNNExplainer (Ying et al., NeurIPS 2019) for TRACE-X transaction predictions.

For one target transaction we take its causal 2-hop temporal computation subgraph and learn
  * a soft mask over subgraph edges (each edge = a real past transaction),
  * a soft mask over edge-feature columns (shared across edges),
  * a soft mask over node-feature columns (shared across nodes),
by maximising the log-probability of the model's predicted class while penalising mask size
and mask entropy (the original GNNExplainer objective). Masked feature values are moved
towards 0, i.e. the training mean of the standardised features. The sequence branches and
the target transaction's own features are held fixed - this explains the GRAPH evidence.
"""
from __future__ import annotations

import numpy as np
import torch

from src.explainability.common import Explainable

COEFFS = dict(edge_size=0.005, edge_ent=1.0, feat_size=0.05, feat_ent=0.1)


def _entropy(m: torch.Tensor) -> torch.Tensor:
    m = m.clamp(1e-6, 1 - 1e-6)
    return (-m * torch.log(m) - (1 - m) * torch.log(1 - m)).mean()


def masked_forward(ex: Explainable, b: dict, emask=None, efeat=None, nfeat=None) -> torch.Tensor:
    bb = dict(b)
    if efeat is not None:
        bb["g_edge_attr"] = b["g_edge_attr"] * efeat
    if nfeat is not None:
        bb["g_x"] = b["g_x"] * nfeat
    return ex.model(bb, emask=emask)


def explain(ex: Explainable, eid: int, epochs: int = 200, lr: float = 0.01, seed: int = 0, coeffs: dict | None = None) -> dict:
    c = {**COEFFS, **(coeffs or {})}
    torch.manual_seed(seed)
    b = ex.batch(eid)
    E = b["g_edge_index"].shape[1]
    with torch.no_grad():
        logit0 = float(ex.model(b))
    target = 1.0 if logit0 >= 0 else 0.0  # explain the predicted class
    em = torch.nn.Parameter(torch.randn(E) * 0.1)
    fm = torch.nn.Parameter(torch.randn(b["g_edge_attr"].shape[1]) * 0.1)
    nm = torch.nn.Parameter(torch.randn(b["g_x"].shape[1]) * 0.1)
    opt = torch.optim.Adam([em, fm, nm], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        se, sf, sn = em.sigmoid(), fm.sigmoid(), nm.sigmoid()
        logit = masked_forward(ex, b, se, sf, sn)
        nll = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.tensor([target]))
        loss = (nll + c["edge_size"] * se.sum() + c["edge_ent"] * _entropy(se)
                + c["feat_size"] * (sf.mean() + sn.mean()) + c["feat_ent"] * (_entropy(sf) + _entropy(sn)))
        loss.backward()
        opt.step()
    se, sf, sn = em.detach().sigmoid().numpy(), fm.detach().sigmoid().numpy(), nm.detach().sigmoid().numpy()
    with torch.no_grad():
        p_masked = float(torch.sigmoid(masked_forward(ex, b, torch.from_numpy(se), torch.from_numpy(sf), torch.from_numpy(sn))))
    ei = b["g_edge_index"].numpy()
    st = ex.store
    node_eid, node_side = b["g_node_eid"].numpy(), b["g_node_side"].numpy()

    def acct(node):
        e, s = node_eid[node], node_side[node]
        return int(st.a["src"][e] if s == 0 else st.a["dst"][e])

    edges = []
    for j in range(E):
        child, parent = int(ei[0, j]), int(ei[1, j])
        e2 = int(b["g_edge_eid"][j])
        direction_out = bool(b["g_edge_attr"][j, -1] > 0.5)  # parent sent the transaction
        edges.append({
            "edge_index": j, "eid": e2, "tx_id": int(st.a["tx_id"][e2]), "hop": int(b["g_edge_hop"][j]),
            "src_account": int(st.a["src"][e2]), "dst_account": int(st.a["dst"][e2]),
            "parent_account": acct(parent), "neighbor_account": acct(child), "parent_sent": direction_out,
            "minutes_before_parent": float(b["g_edge_dt"][j]), "label": int(st.a["label"][e2]),
            "importance": float(se[j]),
        })
    # account importance = max mask over incident subgraph edges (excluding the two endpoints)
    centre = {int(st.a["src"][eid]), int(st.a["dst"][eid])}
    acc_imp: dict[int, float] = {}
    for e in edges:
        a = e["neighbor_account"]
        if a not in centre:
            acc_imp[a] = max(acc_imp.get(a, 0.0), e["importance"])
    edge_feat_names = st.edge_names + ["direction(parent_sent)"]
    return {
        "eid": int(eid), "tx_id": int(st.a["tx_id"][eid]), "prob": float(1 / (1 + np.exp(-logit0))),
        "explained_class": int(target), "prob_with_soft_masks": p_masked,
        "sender_account": int(st.a["src"][eid]), "receiver_account": int(st.a["dst"][eid]),
        "n_subgraph_edges": E, "n_subgraph_nodes": int(b["g_x"].shape[0]),
        "edges": edges,
        "important_accounts": sorted([{"account": a, "importance": v} for a, v in acc_imp.items()], key=lambda d: -d["importance"]),
        "edge_feature_importance": dict(zip(edge_feat_names, map(float, sf))),
        "node_feature_importance": dict(zip(st.node_names, map(float, sn))),
        "edge_mask": se.tolist(),
    }


@torch.no_grad()
def edge_fidelity(ex: Explainable, eid: int, edge_mask: np.ndarray, k: int = 5, n_random: int = 20, seed: int = 0) -> dict:
    """Fidelity+ (remove top-k edges) and fidelity- (keep only top-k) vs. random-k baselines."""
    b = ex.batch(eid)
    E = len(edge_mask)
    p0 = float(ex.prob(b)[0])
    if E == 0:
        return {"p": p0, "E": 0}
    k = min(k, E)
    top = np.argsort(-edge_mask)[:k]

    def p_with(keep: np.ndarray) -> float:
        return float(ex.prob(b, emask=torch.from_numpy(keep.astype(np.float32)))[0])

    rm = np.ones(E); rm[top] = 0
    only = np.zeros(E); only[top] = 1
    rng = np.random.default_rng(seed)
    rand_rm, rand_only = [], []
    for _ in range(n_random):
        r = rng.choice(E, k, replace=False)
        a = np.ones(E); a[r] = 0
        o = np.zeros(E); o[r] = 1
        rand_rm.append(p0 - p_with(a)); rand_only.append(p0 - p_with(o))
    return {
        "p": p0, "E": int(E), "k": int(k),
        "fid_plus": p0 - p_with(rm), "fid_plus_random": float(np.mean(rand_rm)),
        "fid_minus": p0 - p_with(only), "fid_minus_random": float(np.mean(rand_only)),
        "p_without_graph": p_with(np.zeros(E)),
        "sparsity_frac_mask_gt_0.5": float((edge_mask > 0.5).mean()),
    }
