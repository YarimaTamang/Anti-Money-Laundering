"""Generate explanations for TRACE-X test predictions.

python scripts/explain.py [--exp tracex] [--seed 0] [--n-global 40] [--stability-seeds 1,2]

Outputs (reports/explanations/):
  cases/<tx_id>.json + .md   per-case GNNExplainer / IG / SHAP + investigator summary
  figures/                   subgraph, IG and SHAP figures
  global_shap.json, quality.json, quality.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.data.preprocessing import SPLIT_TEST  # noqa: E402
from src.evaluation.metrics import sigmoid  # noqa: E402
from src.explainability import gnn_explainer, integrated_gradients, report, shap_explainer  # noqa: E402
from src.explainability.common import load_accounts, load_explainable  # noqa: E402
from src.explainability.fidelity import stability, summarise  # noqa: E402
from src.utils.config import PROJECT_ROOT  # noqa: E402
from src.utils.plotting import NORMAL, SERIES, SUSPICIOUS, apply_style, plt, save  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "explanations"


def select_cases(y, p, eids, thr, rng):
    o = np.argsort(-p)
    tp = [i for i in o if y[i] == 1 and p[i] >= thr]
    fp = [i for i in o if y[i] == 0 and p[i] >= thr]
    fn = [i for i in o if y[i] == 1 and p[i] < thr]
    pick = []
    if tp:
        pick += [("true_positive", int(eids[i])) for i in rng.choice(tp[:200], min(5, len(tp)), replace=False)]
    pick += [("false_positive", int(eids[i])) for i in fp[:2]]
    pick += [("false_negative", int(eids[i])) for i in fn[:2]]
    tn = np.flatnonzero((y == 0) & (p < thr))
    pick += [("true_negative", int(eids[rng.choice(tn)]))]
    return pick


def tx_record(df_row, accounts) -> dict:
    r = df_row
    return {"tx_id": int(r.tx_id), "timestamp": str(r.timestamp), "sender": accounts[int(r.src)], "receiver": accounts[int(r.dst)],
            "amount_paid": float(r.amount_paid), "currency_paid": r.currency_paid, "amount_received": float(r.amount_received),
            "currency_received": r.currency_received, "usd_paid": float(r.usd_paid), "payment_format": r.payment_format,
            "ground_truth_label": int(r.label)}


def plot_subgraph(g: dict, accounts, path, top: int = 25):
    import networkx as nx

    edges = sorted(g["edges"], key=lambda e: -e["importance"])[:top]
    G = nx.MultiDiGraph()
    s, r = g["sender_account"], g["receiver_account"]
    G.add_edge(s, r, w=1.0, target=True)
    for e in edges:
        G.add_edge(e["src_account"], e["dst_account"], w=e["importance"], target=False)
    pos = nx.spring_layout(nx.Graph(G), seed=1, k=0.9)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.grid(False); ax.axis("off")
    for u, v, d in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, width=3 if d["target"] else 0.5 + 3 * d["w"],
                               edge_color=SUSPICIOUS if d["target"] else NORMAL, alpha=1 if d["target"] else 0.25 + 0.75 * d["w"],
                               arrows=True, arrowsize=10, connectionstyle="arc3,rad=0.08")
    colors = [SUSPICIOUS if n in (s, r) else "#8a8984" for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[220 if n in (s, r) else 70 for n in G.nodes], node_color=colors,
                           edgecolors="#fcfcfb", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: accounts[n] for n in (s, r)}, font_size=7)
    ax.set_title(f"GNNExplainer subgraph — tx {g['tx_id']} (p={g['prob']:.3f}); edge width = mask importance")
    save(fig, path)


def plot_ig(ig: dict, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    for ax, key, title, c in ((axes[0], "sender_steps", "Sender outgoing history", NORMAL),
                              (axes[1], "receiver_steps", "Receiver incoming history", SERIES[2])):
        st = ig[key]
        if not st:
            ax.text(0.5, 0.5, "no earlier transactions", ha="center", transform=ax.transAxes, color="#52514e")
        x = [f"-{r['minutes_before_target']}m" for r in st]
        v = [r["attribution"] for r in st]
        ax.bar(range(len(v)), v, color=[SUSPICIOUS if a > 0 else c for a in v], width=0.6)
        ax.set_xticks(range(len(v))); ax.set_xticklabels(x, rotation=60, fontsize=6)
        ax.axhline(0, color="#8a8984", lw=0.8); ax.set_title(title)
    axes[0].set_ylabel("IG attribution (logit)")
    fig.suptitle(f"Integrated Gradients — tx {ig['tx_id']} (oldest → newest; orange = pushes towards suspicious)", fontsize=9)
    save(fig, path)


def plot_shap_local(sh: dict, path, k: int = 12):
    sv = np.asarray(sh["shap"]); o = np.argsort(-np.abs(sv))[:k][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh([sh["names"][i] for i in o], sv[o], color=[SUSPICIOUS if sv[i] > 0 else NORMAL for i in o], height=0.6)
    ax.axvline(0, color="#8a8984", lw=0.8)
    ax.set_xlabel("SHAP value (logit units)")
    ax.set_title(f"Kernel SHAP — tx {sh['tx_id']}  (base {sh['expected_value']:.2f} → logit {sh['logit']:.2f})")
    save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="tracex")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="full")
    ap.add_argument("--n-global", type=int, default=40)
    ap.add_argument("--shap-nsamples", type=int, default=400)
    ap.add_argument("--stability-seeds", default="1,2")
    ap.add_argument("--n-stability", type=int, default=4)
    a = ap.parse_args()
    apply_style()
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    ex = load_explainable(a.exp, a.seed, a.mode)
    st = ex.store
    accounts = load_accounts(st)
    df = st.transactions()
    exp_dir = PROJECT_ROOT / "experiments" / a.mode / a.exp / f"seed{a.seed}"
    metrics = json.load(open(exp_dir / "metrics.json"))
    thr = metrics["test"]["threshold"]
    platt = metrics["platt"]
    te = st.split_eids(SPLIT_TEST)
    p = sigmoid(np.load(exp_dir / "preds_test.npy"))
    y = st.a["label"][te]
    cases = select_cases(y, p, te, thr, rng)
    view = shap_explainer.TabularView(ex)
    bg = shap_explainer.background(view, 12)
    fid_g, fid_ig, fid_sh, all_cases = [], [], [], []
    for kind, eid in cases:
        t0 = time.time()
        g = gnn_explainer.explain(ex, eid)
        ig = integrated_gradients.explain(ex, eid)
        sh = shap_explainer.explain(view, eid, bg, nsamples=a.shap_nsamples)
        sh["tx_id"] = int(st.a["tx_id"][eid])
        fg = gnn_explainer.edge_fidelity(ex, eid, np.asarray(g["edge_mask"]))
        fi = integrated_gradients.step_fidelity(ex, eid, ig)
        fs = shap_explainer.feature_fidelity(view, eid, np.asarray(sh["shap"]), bg)
        for f in (fg, fi, fs):
            f["kind"] = kind
        fid_g.append(fg); fid_ig.append(fi); fid_sh.append(fs)
        state = np.asarray(st.a["node_state"][eid])
        logit = float(np.log(p[np.searchsorted(te, eid)] / (1 - p[np.searchsorted(te, eid)])))
        summ = report.build_summary(
            tx_record(df.iloc[eid], accounts), float(p[np.searchsorted(te, eid)]),
            float(1 / (1 + np.exp(-(platt["a"] * logit + platt["b"])))),
            report.account_profile(state[0], st.node_names), report.account_profile(state[1], st.node_names),
            g, sh, ig, accounts, threshold=thr)
        tid = int(st.a["tx_id"][eid])
        g_small = {k: v for k, v in g.items() if k != "edge_mask"}
        with open(OUT / "cases" / f"{tid}.json", "w") as f:
            json.dump({"kind": kind, "eid": eid, "summary": summ, "gnn": g_small, "ig": ig, "shap": sh,
                       "fidelity": {"gnn": fg, "ig": fi, "shap": fs}}, f, indent=1, default=float)
        (OUT / "cases" / f"{tid}.md").write_text(report.to_markdown(summ), encoding="utf-8")
        plot_subgraph(g, accounts, OUT / "figures" / f"gnn_{tid}.png")
        plot_ig(ig, OUT / "figures" / f"ig_{tid}.png")
        plot_shap_local(sh, OUT / "figures" / f"shap_{tid}.png")
        all_cases.append({"kind": kind, "tx_id": tid, "eid": eid, "prob": summ["probability"], "label": int(st.a["label"][eid]),
                          "ig_completeness_delta": ig["completeness_delta"], "shap_additivity_gap": sh["additivity_gap"]})
        print(f"[{kind}] tx {tid} p={summ['probability']:.3f} fid+={fg.get('fid_plus', 0):.3f} (rand {fg.get('fid_plus_random', 0):.3f}) "
              f"IG drop={fi.get('drop_topk', 0):.3f} SHAP drop={fs['drop_topk']:.3f} [{time.time()-t0:.0f}s]", flush=True)

    # ---------------- global SHAP over a stratified test sample
    pos, neg = te[y == 1], te[y == 0]
    g_eids = np.concatenate([rng.choice(pos, a.n_global // 2, replace=False), rng.choice(neg, a.n_global // 2, replace=False)])
    S, X = [], []
    t0 = time.time()
    for eid in g_eids:
        r = shap_explainer.explain(view, int(eid), bg, nsamples=300)
        S.append(r["shap"]); X.append(r["x"])
    S, X = np.array(S), np.array(X)
    json.dump({"names": view.names, "eids": g_eids.tolist(), "labels": st.a["label"][g_eids].tolist(), "shap": S.tolist(), "x": X.tolist(),
               "mean_abs": dict(zip(view.names, np.abs(S).mean(0).tolist()))}, open(OUT / "global_shap.json", "w"))
    print(f"global SHAP on {len(g_eids)} tx in {time.time()-t0:.0f}s", flush=True)
    import shap

    plt.figure()
    shap.summary_plot(S, X, feature_names=view.names, show=False, max_display=15, plot_size=(8, 5.5))
    plt.title("Global Kernel SHAP summary (test sample: 50% laundering, 50% normal)", fontsize=10)
    save(plt.gcf(), OUT / "figures" / "shap_summary.png")
    ma = np.abs(S).mean(0); o = np.argsort(ma)[-15:]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh([view.names[i] for i in o], ma[o], color=NORMAL, height=0.6)
    ax.set_xlabel("mean |SHAP| (logit units)"); ax.set_title("Global feature importance (Kernel SHAP)")
    save(fig, OUT / "figures" / "shap_global_importance.png")

    # ---------------- stability across seeds
    stab = None
    seeds = [int(s) for s in a.stability_seeds.split(",") if s]
    avail = [s for s in seeds if (PROJECT_ROOT / "checkpoints" / a.mode / a.exp / f"seed{s}.pt").exists()]
    if avail:
        models = {a.seed: ex, **{s: load_explainable(a.exp, s, a.mode, store=st) for s in avail}}
        tp_eids = [e for k, e in cases if k == "true_positive"][: a.n_stability]
        stab = stability(models, tp_eids, bg)
        stab["seeds"] = sorted(models)

    quality = {
        "n_cases": len(cases), "cases": all_cases,
        "gnn_explainer": summarise(fid_g, ["fid_plus", "fid_plus_random", "fid_minus", "fid_minus_random", "sparsity_frac_mask_gt_0.5"]),
        "integrated_gradients": summarise(fid_ig, ["drop_topk", "drop_random_k", "sparsity_top3_mass"]),
        "shap": summarise(fid_sh, ["drop_topk", "drop_random_k", "sparsity_top5_mass"]),
        "ig_completeness_abs_delta_mean": float(np.mean([abs(c["ig_completeness_delta"]) for c in all_cases])),
        "shap_additivity_abs_gap_mean": float(np.mean([abs(c["shap_additivity_gap"]) for c in all_cases])),
        "stability_across_seeds": stab,
        "per_case": {"gnn": fid_g, "ig": fid_ig, "shap": fid_sh},
    }
    json.dump(quality, open(OUT / "quality.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in quality.items() if k not in ("per_case", "cases")}, indent=1, default=float))


if __name__ == "__main__":
    main()
