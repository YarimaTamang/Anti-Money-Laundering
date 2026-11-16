"""Explanation-quality evaluation: fidelity (deletion tests vs random), sparsity, stability across seeds."""
from __future__ import annotations

import numpy as np

from src.explainability import gnn_explainer, integrated_gradients, shap_explainer
from src.explainability.common import Explainable, spearman, topk_jaccard


def summarise(rows: list[dict], keys: list[str]) -> dict:
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r and r[k] is not None and np.isfinite(r[k])], float)
        if len(v):
            out[k] = {"mean": float(v.mean()), "std": float(v.std()), "n": int(len(v))}
    return out


def stability(models: dict[int, Explainable], eids: list[int], bg: np.ndarray, gnn_epochs: int = 150,
              shap_nsamples: int = 300) -> dict:
    """Compare explanations of the same transactions produced by models trained with different seeds."""
    seeds = sorted(models)
    res = {"shap_spearman": [], "shap_top5_jaccard": [], "ig_step_spearman": [], "ig_top3_jaccard": [],
           "gnn_edge_spearman": [], "gnn_top5_jaccard": []}
    for eid in eids:
        sh, ig, gn = {}, {}, {}
        for s in seeds:
            ex = models[s]
            view = shap_explainer.TabularView(ex)
            sh[s] = np.asarray(shap_explainer.explain(view, eid, bg, nsamples=shap_nsamples)["shap"])
            r = integrated_gradients.explain(ex, eid)
            ig[s] = np.array([x["attribution"] for x in r["sender_steps"] + r["receiver_steps"]])
            gn[s] = np.asarray(gnn_explainer.explain(ex, eid, epochs=gnn_epochs)["edge_mask"])
        for i, a in enumerate(seeds):
            for b in seeds[i + 1:]:
                res["shap_spearman"].append(spearman(sh[a], sh[b]))
                res["shap_top5_jaccard"].append(topk_jaccard(sh[a], sh[b], 5))
                if len(ig[a]) > 2:
                    res["ig_step_spearman"].append(spearman(ig[a], ig[b]))
                    res["ig_top3_jaccard"].append(topk_jaccard(ig[a], ig[b], 3))
                if len(gn[a]) > 2:
                    res["gnn_edge_spearman"].append(spearman(gn[a], gn[b]))
                    res["gnn_top5_jaccard"].append(topk_jaccard(gn[a], gn[b], 5))
    return {k: {"mean": float(np.nanmean(v)) if len(v) else None, "std": float(np.nanstd(v)) if len(v) else None,
                "n_pairs": len(v)} for k, v in res.items()}
