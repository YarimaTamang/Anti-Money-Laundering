"""Aggregate experiment results into tables and figures.

python -m src.evaluation.comparison [--mode full]
Reads experiments/<mode>/<exp>/seed*/metrics.json (+ preds_test.npy, history.csv) and writes
reports/results/: results_models.{md,csv}, results_ablation.{md,csv}, figures/*.png.
Only completed runs are included; the number of completed seeds is reported per row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.data.preprocessing import SPLIT_TEST
from src.evaluation.alert_budget import recall_curve
from src.evaluation.calibration import reliability
from src.evaluation.metrics import sigmoid
from src.utils.config import PROJECT_ROOT, load_yaml, processed_dir
from src.utils.plotting import NORMAL, SERIES, SUSPICIOUS, apply_style, plt, save

DISPLAY = {"tracex": "TRACE-X (proposed)", "lstm": "LSTM", "gru": "GRU", "gcn": "GCN", "graphsage": "GraphSAGE",
           "gat": "GAT", "gat_lstm": "GAT-LSTM", "tracex_focal": "TRACE-X + focal loss",
           "abl_sequence_only": "Sequence branches only", "abl_graph_only": "Graph branch only",
           "abl_no_temporal_enc": "w/o temporal encoding", "abl_no_edge_feats": "w/o edge features (graph)",
           "abl_mean_aggr": "w/o graph attention (mean aggr.)", "abl_no_sender": "w/o sender history",
           "abl_no_receiver": "w/o receiver history", "abl_concat_fusion": "concat instead of gated fusion",
           "abl_seq5": "sequence length 5", "abl_seq20": "sequence length 20"}
MODEL_ORDER = ["lstm", "gru", "gcn", "graphsage", "gat", "gat_lstm", "tracex"]
ABL_ORDER = ["tracex", "abl_sequence_only", "abl_graph_only", "abl_no_temporal_enc", "abl_no_edge_feats", "abl_mean_aggr",
             "abl_no_sender", "abl_no_receiver", "abl_concat_fusion", "abl_seq5", "abl_seq20"]


def flatten(m: dict) -> dict:
    t, ab, e = m["test"], m["test_alert_budget"], m["efficiency"]
    return {
        "roc_auc": t["roc_auc"], "pr_auc": t["pr_auc"], "precision": t["precision"], "recall": t["recall"], "f1": t["f1"],
        "p@0.5%": ab["0.5%"]["precision_at_k"], "r@0.5%": ab["0.5%"]["recall_at_k"],
        "p@0.1%": ab["0.1%"]["precision_at_k"], "r@0.1%": ab["0.1%"]["recall_at_k"], "r@1%": ab["1%"]["recall_at_k"],
        "val_pr_auc": m["val_select"]["pr_auc"], "brier_raw": m["test_calibration_raw"]["brier"],
        "ece_raw": m["test_calibration_raw"]["ece_uniform15"], "brier_platt": m["test_calibration_platt"]["brier"],
        "ece_platt": m["test_calibration_platt"]["ece_uniform15"], "params": m["n_params"],
        "train_min": e["train_time_s"] / 60, "infer_tx_per_s": e["test_throughput_tx_per_s"],
        "latency_ms": e["single_tx_latency_ms_median"], "peak_mem_mb": e.get("peak_memory_mb") or np.nan,
        "best_epoch": m["best_epoch"],
    }


def collect(mode: str) -> pd.DataFrame:
    rows = []
    for f in sorted((PROJECT_ROOT / "experiments" / mode).glob("*/seed*/metrics.json")):
        m = json.load(open(f))
        rows.append({"exp": m["exp"], "seed": m["seed"], **flatten(m)})
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("exp")
    mean, std = g.mean(numeric_only=True), g.std(numeric_only=True, ddof=0)
    out = mean.add_suffix("_mean").join(std.add_suffix("_std"))
    out["n_seeds"] = g.size()
    return out


def fmt(agg, exp, col, digits=4):
    m, s = agg.loc[exp, f"{col}_mean"], agg.loc[exp, f"{col}_std"]
    return f"{m:.{digits}f} ± {s:.{digits}f}" if agg.loc[exp, "n_seeds"] > 1 else f"{m:.{digits}f}"


def md_table(agg, exps, cols, extra_cols=()) -> str:
    head = ["Model", "Seeds"] + [c for c, _ in cols] + [c for c, _ in extra_cols]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for e in exps:
        if e not in agg.index:
            continue
        cells = [DISPLAY.get(e, e), str(int(agg.loc[e, "n_seeds"]))]
        cells += [fmt(agg, e, c, d) for c, d in cols]
        cells += [fmt(agg, e, c, d) for c, d in extra_cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(mode: str = "full"):
    apply_style()
    out = PROJECT_ROOT / "reports" / "results"
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = collect(mode)
    if df.empty:
        print("no completed runs")
        return
    df.to_csv(out / "all_runs.csv", index=False)
    agg = aggregate(df)
    agg.to_csv(out / "aggregate.csv")
    perf = [("roc_auc", 4), ("pr_auc", 4), ("precision", 4), ("recall", 4), ("f1", 4), ("p@0.5%", 4), ("r@0.5%", 4)]
    eff = [("params", 0), ("train_min", 1), ("infer_tx_per_s", 0), ("latency_ms", 1), ("peak_mem_mb", 0)]
    models = [e for e in MODEL_ORDER if e in agg.index] + [e for e in ["tracex_focal"] if e in agg.index]
    abl = [e for e in ABL_ORDER if e in agg.index]
    md = ["# Model comparison (full test split, mean ± std over completed seeds)", "",
          "Threshold for precision/recall/F1 = F1-optimal threshold on val_select; P@K/R@K at K = 0.5% of test transactions.", "",
          md_table(agg, models, perf), "", "## Efficiency", "", md_table(agg, models, eff), "",
          "## Calibration (test)", "",
          md_table(agg, models, [("brier_raw", 5), ("ece_raw", 4), ("brier_platt", 5), ("ece_platt", 4)]), ""]
    (out / "results_models.md").write_text("\n".join(md), encoding="utf-8")
    if len(abl) > 1:
        cols = [("pr_auc", 4), ("roc_auc", 4), ("f1", 4), ("r@0.5%", 4), ("p@0.5%", 4), ("params", 0), ("train_min", 1)]
        base = agg.loc["tracex", "pr_auc_mean"] if "tracex" in agg.index else np.nan
        lines = ["# Ablation study (full test split, mean ± std over completed seeds)", "", md_table(agg, abl, cols), "",
                 "ΔPR-AUC relative to full TRACE-X:", ""]
        for e in abl[1:]:
            lines.append(f"- {DISPLAY.get(e, e)}: {agg.loc[e, 'pr_auc_mean'] - base:+.4f}")
        (out / "results_ablation.md").write_text("\n".join(lines), encoding="utf-8")
    _figures(mode, agg, df, models, abl, fig_dir)
    print((out / "results_models.md").read_text(encoding="utf-8"))


def _bar(ax, agg, exps, col, color_fn):
    m = [agg.loc[e, f"{col}_mean"] for e in exps]
    s = [agg.loc[e, f"{col}_std"] for e in exps]
    ax.barh(range(len(exps)), m, xerr=s, color=[color_fn(e) for e in exps], height=0.6,
            error_kw={"ecolor": "#52514e", "elinewidth": 1, "capsize": 2})
    ax.set_yticks(range(len(exps))); ax.set_yticklabels([DISPLAY.get(e, e) for e in exps])
    for i, v in enumerate(m):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=7, color="#0b0b0b")


def _figures(mode, agg, df, models, abl, fig_dir):
    color = lambda e: SUSPICIOUS if e == "tracex" else NORMAL  # noqa: E731
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, col, title in zip(axes, ["pr_auc", "f1", "r@0.5%"], ["PR-AUC", "F1 (val-selected threshold)", "Recall @ top 0.5% alerts"]):
        _bar(ax, agg, models, col, color); ax.set_title(title)
    fig.suptitle("Model comparison on the test split (mean ± std over seeds)", fontsize=10)
    save(fig, fig_dir / "model_comparison.png")
    if len(abl) > 1:
        fig, ax = plt.subplots(figsize=(7, 0.42 * len(abl) + 1))
        _bar(ax, agg, abl[::-1], "pr_auc", color)
        ax.set_title("Ablations: test PR-AUC (mean ± std)")
        save(fig, fig_dir / "ablation_pr_auc.png")

    store_dir = processed_dir(load_yaml("data"), mode)
    split = np.load(store_dir / "split.npy"); label = np.load(store_dir / "label.npy")
    y = label[split == SPLIT_TEST].astype(float)
    preds = {}
    for e in models:
        f = PROJECT_ROOT / "experiments" / mode / e / "seed0" / "preds_test.npy"
        if f.exists():
            preds[e] = sigmoid(np.load(f))
    if not preds:
        return
    cols = {e: SERIES[i % len(SERIES)] for i, e in enumerate([m for m in MODEL_ORDER if m != "tracex"])}
    cols["tracex"] = "#0b0b0b"
    cols["tracex_focal"] = SERIES[7]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for e, p in preds.items():
        pr, rc, _ = precision_recall_curve(y, p)
        axes[0].plot(rc, pr, color=cols[e], lw=2.4 if e == "tracex" else 1.4, label=DISPLAY[e])
        fpr, tpr, _ = roc_curve(y, p)
        axes[1].plot(fpr, tpr, color=cols[e], lw=2.4 if e == "tracex" else 1.4, label=DISPLAY[e])
    axes[0].set_xlabel("recall"); axes[0].set_ylabel("precision"); axes[0].set_title("Precision–recall (test, seed 0)")
    axes[1].plot([0, 1], [0, 1], color="#c3c2b7", lw=1, ls="--")
    axes[1].set_xlabel("false positive rate"); axes[1].set_ylabel("true positive rate"); axes[1].set_title("ROC (test, seed 0)")
    axes[0].legend(fontsize=7)
    save(fig, fig_dir / "pr_roc_curves.png")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for e, p in preds.items():
        b, r = recall_curve(y, p, 0.03)
        ax.plot(100 * b, r, color=cols[e], lw=2.4 if e == "tracex" else 1.4, label=DISPLAY[e])
    ax.axvline(0.5, color="#8a8984", lw=0.8, ls="--"); ax.text(0.52, 0.03, "0.5% budget", fontsize=7, color="#52514e")
    ax.set_xlabel("alert budget (% of test transactions investigated)"); ax.set_ylabel("recall (detection rate)")
    ax.set_title("Alert-budget evaluation (test, seed 0)"); ax.legend(fontsize=7)
    save(fig, fig_dir / "alert_budget.png")

    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 2.6))
    axes = np.atleast_1d(axes)
    for ax, (e, p) in zip(axes, preds.items()):
        thr = json.load(open(PROJECT_ROOT / "experiments" / mode / e / "seed0" / "metrics.json"))["test"]["threshold"]
        pr = p >= thr
        cm = np.array([[((y == 0) & ~pr).sum(), ((y == 0) & pr).sum()], [((y == 1) & ~pr).sum(), ((y == 1) & pr).sum()]])
        ax.imshow(np.log10(cm + 1), cmap="Blues"); ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=7,
                        color="white" if np.log10(cm[i, j] + 1) > 3.5 else "#0b0b0b")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"], fontsize=7)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true 0", "true 1"], fontsize=7)
        ax.set_title(DISPLAY[e], fontsize=8)
    fig.suptitle("Confusion matrices (test, seed 0, val-selected threshold)", fontsize=9)
    save(fig, fig_dir / "confusion_matrices.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for e in preds:
        h = pd.read_csv(PROJECT_ROOT / "experiments" / mode / e / "seed0" / "history.csv")
        axes[0].plot(h["epoch"], h["train_loss"], color=cols[e], lw=1.4, label=DISPLAY[e])
        axes[0].plot(h["epoch"], h["val_loss"], color=cols[e], lw=1.4, ls="--")
        axes[1].plot(h["epoch"], h["val_pr_auc"], color=cols[e], lw=2.4 if e == "tracex" else 1.4, label=DISPLAY[e])
    axes[0].set_title("Loss (solid = train, dashed = validation)"); axes[0].set_xlabel("epoch")
    axes[1].set_title("Validation PR-AUC (val_select)"); axes[1].set_xlabel("epoch"); axes[1].legend(fontsize=7)
    save(fig, fig_dir / "training_curves.png")

    if "tracex" in preds:
        m = json.load(open(PROJECT_ROOT / "experiments" / mode / "tracex" / "seed0" / "metrics.json"))
        logit = np.log(np.clip(preds["tracex"], 1e-12, 1 - 1e-12) / np.clip(1 - preds["tracex"], 1e-12, 1))
        pc = 1 / (1 + np.exp(-(m["platt"]["a"] * logit + m["platt"]["b"])))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, p, title in ((axes[0], preds["tracex"], "raw (class-weighted training)"), (axes[1], pc, "Platt-scaled on val_cal")):
            conf, acc, cnt = reliability(y, p, 15, strategy="quantile")
            ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=1, ls="--")
            ax.plot(conf, acc, "o-", color=SUSPICIOUS, ms=4)
            ax.set_xscale("symlog", linthresh=1e-3); ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_xlabel("mean predicted probability (quantile bins)"); ax.set_ylabel("observed laundering rate")
            ax.set_title(f"TRACE-X reliability — {title}", fontsize=9)
        save(fig, fig_dir / "calibration_tracex.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full")
    main(ap.parse_args().mode)
