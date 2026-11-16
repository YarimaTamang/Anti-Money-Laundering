"""Uncertainty estimation and human-review routing for TRACE-X.

python scripts/uncertainty.py [--exp tracex --seed 0 --T 20]

* Split conformal (class-conditional): calibrated on val_cal, evaluated on the FULL test split.
* MC Dropout (T passes): run on val_cal (subsample) to fix an uncertainty threshold, and on a
  weighted test subsample (all test positives + 100k uniformly sampled negatives, negatives
  weighted by 1/sampling-rate) because T x 1M CPU forward passes are too expensive.
Outputs: reports/uncertainty/{uncertainty.json, uncertainty.md, figures}, plus
         experiments/<mode>/<exp>/seed<k>/conformal_test_outcome.npy (per test transaction).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score  # noqa: E402

from src.data.preprocessing import SPLIT_TEST  # noqa: E402
from src.evaluation.metrics import sigmoid  # noqa: E402
from src.training.checkpoints import load_checkpoint  # noqa: E402
from src.training.run_experiment import eval_subset  # noqa: E402
from src.graph.temporal_graph import TemporalGraphStore  # noqa: E402
from src.uncertainty.conformal import MondrianConformal  # noqa: E402
from src.uncertainty.mc_dropout import mc_predict  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_yaml, processed_dir  # noqa: E402
from src.utils.plotting import NORMAL, SERIES, SUSPICIOUS, apply_style, plt, save  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="tracex"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="full"); ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args()
    apply_style()
    out = PROJECT_ROOT / "reports" / "uncertainty"; (out / "figures").mkdir(parents=True, exist_ok=True)
    exp_dir = PROJECT_ROOT / "experiments" / a.mode / a.exp / f"seed{a.seed}"
    store = TemporalGraphStore(processed_dir(load_yaml("data"), a.mode))
    model, ck = load_checkpoint(PROJECT_ROOT / "checkpoints" / a.mode / a.exp / f"seed{a.seed}.pt")
    cfg = ck["extra"]["cfg"]
    y_all = store.a["label"]
    vc = np.load(exp_dir / "preds_val_cal.npz")
    p_cal, y_cal = sigmoid(vc["logits"]), y_all[vc["eids"]]
    te = store.split_eids(SPLIT_TEST)
    p_te, y_te = sigmoid(np.load(exp_dir / "preds_test.npy")), y_all[te]

    res = {"conformal": {}, "notes": "conformal calibrated on val_cal; evaluated on full test split"}
    for alpha in (0.05, 0.1, 0.2):
        cp = MondrianConformal().fit(p_cal, y_cal, alpha)
        res["conformal"][str(alpha)] = cp.report(p_te, y_te)
    cp = MondrianConformal().fit(p_cal, y_cal, a.alpha)
    np.save(exp_dir / "conformal_test_outcome.npy", cp.outcome(p_te))
    json.dump({"alpha": a.alpha, "t_normal": cp.t_normal, "t_suspicious": cp.t_suspicious},
              open(exp_dir / "conformal.json", "w"), indent=1)

    # ---------------- MC dropout
    vc_e = vc["eids"]
    mu_c, sd_c = mc_predict(model, store, vc_e, cfg, T=a.T)
    ts_e, ts_w = eval_subset(store, SPLIT_TEST, cfg["eval_subsample_neg"], cfg["eval_subsample_seed"])
    mu_t, sd_t = mc_predict(model, store, ts_e, cfg, T=a.T)
    y_t = y_all[ts_e].astype(float)
    thr = json.load(open(exp_dir / "metrics.json"))["test"]["threshold"]
    err = ((mu_t >= thr) != (y_t == 1)).astype(int)
    sd_thr = float(np.quantile(sd_c, 0.95))
    review = sd_t > sd_thr
    keep = ~review
    pred = mu_t >= thr

    def wsum(m):
        return float((ts_w * m).sum())

    tp, fp, fn = wsum(keep & pred & (y_t == 1)), wsum(keep & pred & (y_t == 0)), wsum(keep & ~pred & (y_t == 1))
    tp_all, fp_all, fn_all = wsum(pred & (y_t == 1)), wsum(pred & (y_t == 0)), wsum(~pred & (y_t == 1))
    res["mc_dropout"] = {
        "T": a.T, "evaluated_on": "test subsample (all positives + uniform negatives, IPW-weighted)",
        "n_eval": int(len(ts_e)), "std_threshold_val_cal_q95": sd_thr,
        "abstention_rate": wsum(review) / ts_w.sum(),
        "positives_routed_to_review": wsum(review & (y_t == 1)) / wsum(y_t == 1),
        "retained_precision": tp / max(tp + fp, 1e-12), "retained_recall": tp / max(tp + fn, 1e-12),
        "all_precision": tp_all / max(tp_all + fp_all, 1e-12), "all_recall": tp_all / max(tp_all + fn_all, 1e-12),
        "error_detection_auroc_of_std": float(roc_auc_score(err, sd_t, sample_weight=ts_w)) if 0 < err.sum() < len(err) else None,
        "mean_std_positives": float(sd_t[y_t == 1].mean()), "mean_std_negatives": float(np.average(sd_t[y_t == 0], weights=ts_w[y_t == 0])),
        "mc_mean_vs_deterministic_spearman": None,
    }
    np.savez(exp_dir / "mc_dropout_test_subsample.npz", eids=ts_e, w=ts_w, mean=mu_t, std=sd_t)
    json.dump(res, open(out / "uncertainty.json", "w"), indent=1)

    # ---------------- figures
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    bins = np.linspace(0, max(sd_t.max(), 1e-3), 50)
    axes[0].hist(sd_t[y_t == 0], bins=bins, weights=ts_w[y_t == 0], density=True, histtype="step", lw=2, color=NORMAL, label="normal")
    axes[0].hist(sd_t[y_t == 1], bins=bins, density=True, histtype="step", lw=2, color=SUSPICIOUS, label="laundering")
    axes[0].axvline(sd_thr, color="#52514e", ls="--", lw=1); axes[0].set_yscale("log")
    axes[0].set_xlabel("MC-dropout std of p(suspicious)"); axes[0].set_title("Uncertainty distribution (test)"); axes[0].legend()
    al = sorted(res["conformal"])
    cov1 = [res["conformal"][k]["coverage_class1"] for k in al]
    cov0 = [res["conformal"][k]["coverage_class0"] for k in al]
    abst = [res["conformal"][k]["abstention_rate"] for k in al]
    x = np.arange(len(al))
    axes[1].bar(x - 0.25, cov0, 0.25, color=NORMAL, label="coverage (normal)")
    axes[1].bar(x, cov1, 0.25, color=SUSPICIOUS, label="coverage (laundering)")
    axes[1].bar(x + 0.25, abst, 0.25, color=SERIES[2], label="routed to human review")
    for i, k in enumerate(al):
        axes[1].axhline(1 - float(k), xmin=(i) / len(al), xmax=(i + 1) / len(al), color="#52514e", ls=":", lw=1)
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"α={k}" for k in al]); axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Class-conditional conformal (test)"); axes[1].legend(fontsize=7, loc="lower right")
    save(fig, out / "figures" / "uncertainty.png")

    c = res["conformal"][str(a.alpha)]
    m = res["mc_dropout"]
    md = ["# Uncertainty & human review (TRACE-X, seed %d)" % a.seed, "",
          "## Split conformal (class-conditional), calibrated on val_cal, full test split", "",
          "| α | coverage normal | coverage laundering | confident normal | confident suspicious | human review | retained precision | retained recall |",
          "|---|---|---|---|---|---|---|---|"]
    for k in al:
        r = res["conformal"][k]
        md.append(f"| {k} | {r['coverage_class0']:.4f} | {r['coverage_class1']:.4f} | {r['frac_confident_normal']:.4f} | "
                  f"{r['frac_confident_suspicious']:.5f} | {r['abstention_rate']:.4f} | {r['retained_precision']:.4f} | {r['retained_recall']:.4f} |")
    md += ["", f"Thresholds at α={a.alpha}: p ≤ {c['t_normal']:.4f} → normal allowed; p ≥ {c['t_suspicious']:.4f} → suspicious allowed.", "",
           f"## MC Dropout (T={a.T}), {m['evaluated_on']}", "",
           f"- Review rule: std > {m['std_threshold_val_cal_q95']:.4f} (95th percentile on val_cal)",
           f"- Abstention rate: {m['abstention_rate']:.4f}; laundering transactions routed to review: {m['positives_routed_to_review']:.4f}",
           f"- Precision/recall on retained predictions: {m['retained_precision']:.4f} / {m['retained_recall']:.4f} (all: {m['all_precision']:.4f} / {m['all_recall']:.4f})",
           f"- AUROC of MC std for detecting misclassifications: {m['error_detection_auroc_of_std']}",
           "", "_The system is a decision-support tool; an alert or a 'confident' outcome does not establish that money laundering occurred._"]
    (out / "uncertainty.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
