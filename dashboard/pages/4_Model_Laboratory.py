import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components import data  # noqa: E402
from dashboard.components.theme import BLUE, ORANGE, SERIES, setup  # noqa: E402
from src.evaluation.comparison import ABL_ORDER, DISPLAY, MODEL_ORDER  # noqa: E402

setup("Model Laboratory")
st.title("Model Laboratory")
st.caption("All numbers are read from saved experiment outputs (full test split, mean ± std over completed seeds).")
agg, runs = data.results()
if agg is None:
    st.info("Run `python -m src.evaluation.comparison` first."); st.stop()

models = [e for e in MODEL_ORDER if e in agg.index]
abls = [e for e in ABL_ORDER if e in agg.index]
metric = st.selectbox("Metric", ["pr_auc", "roc_auc", "f1", "precision", "recall", "r@0.5%", "p@0.5%", "r@1%"], index=0)


def bar(exps, title):
    fig = go.Figure(go.Bar(
        y=[DISPLAY.get(e, e) for e in exps], x=[agg.loc[e, f"{metric}_mean"] for e in exps], orientation="h",
        error_x=dict(type="data", array=[agg.loc[e, f"{metric}_std"] for e in exps], color="#a9b6cf"),
        marker_color=[ORANGE if e == "tracex" else BLUE for e in exps],
        customdata=[int(agg.loc[e, "n_seeds"]) for e in exps],
        hovertemplate="%{y}: %{x:.4f} (seeds: %{customdata})<extra></extra>"))
    fig.update_layout(title=title, height=60 + 38 * len(exps), xaxis_title=metric)
    return fig


t1, t2, t3, t4 = st.tabs(["Baseline comparison", "Ablations", "Training curves", "Efficiency"])
with t1:
    st.plotly_chart(bar(models, f"Test {metric}: TRACE-X vs baselines"), use_container_width=True)
    cols = ["roc_auc", "pr_auc", "precision", "recall", "f1", "p@0.5%", "r@0.5%"]
    st.dataframe(pd.DataFrame({"model": [DISPLAY[e] for e in models], "seeds": [int(agg.loc[e, "n_seeds"]) for e in models],
                               **{c: [f"{agg.loc[e, c + '_mean']:.4f} ± {agg.loc[e, c + '_std']:.4f}" for e in models] for c in cols}}),
                 hide_index=True, use_container_width=True)
    f = data.REPORTS / "results" / "figures" / "pr_roc_curves.png"
    if f.exists():
        st.image(str(f), caption="Precision–recall and ROC curves (test, seed 0)")
with t2:
    if len(abls) > 1:
        st.plotly_chart(bar(abls, f"Ablations: test {metric}"), use_container_width=True)
        base = agg.loc["tracex", f"{metric}_mean"]
        st.dataframe(pd.DataFrame({"variant": [DISPLAY[e] for e in abls], "seeds": [int(agg.loc[e, "n_seeds"]) for e in abls],
                                   metric: [f"{agg.loc[e, metric + '_mean']:.4f} ± {agg.loc[e, metric + '_std']:.4f}" for e in abls],
                                   "Δ vs TRACE-X": [f"{agg.loc[e, metric + '_mean'] - base:+.4f}" for e in abls]}),
                     hide_index=True, use_container_width=True)
    else:
        st.info("Ablation runs not available yet.")
with t3:
    pick = st.multiselect("Experiments", list(agg.index), default=models)
    fig, fig2 = go.Figure(), go.Figure()
    for i, e in enumerate(pick):
        h = data.history(e)
        if h is None:
            continue
        c = "#ffffff" if e == "tracex" else SERIES[i % len(SERIES)]
        fig.add_scatter(x=h["epoch"], y=h["val_pr_auc"], name=DISPLAY.get(e, e), line=dict(color=c, width=3 if e == "tracex" else 1.6))
        fig2.add_scatter(x=h["epoch"], y=h["train_loss"], name=f"{DISPLAY.get(e, e)} train", line=dict(color=c))
        fig2.add_scatter(x=h["epoch"], y=h["val_loss"], name=f"{DISPLAY.get(e, e)} val", line=dict(color=c, dash="dash"))
    fig.update_layout(title="Validation PR-AUC per epoch (seed 0)", height=380, xaxis_title="epoch")
    fig2.update_layout(title="Training (solid) and validation (dashed) loss, seed 0", height=380, xaxis_title="epoch")
    st.plotly_chart(fig, use_container_width=True); st.plotly_chart(fig2, use_container_width=True)
with t4:
    ex = [e for e in agg.index]
    fig = go.Figure(go.Scatter(x=[agg.loc[e, "train_min_mean"] for e in ex], y=[agg.loc[e, "pr_auc_mean"] for e in ex], mode="markers+text",
                               text=[DISPLAY.get(e, e) for e in ex], textposition="top center",
                               marker=dict(size=[8 + 14 * agg.loc[e, "params_mean"] / agg["params_mean"].max() for e in ex],
                                           color=[ORANGE if e == "tracex" else BLUE for e in ex], line=dict(color="#0a1628", width=2))))
    fig.update_layout(title="PR-AUC vs training time (marker size ∝ parameters)", xaxis_title="training time (min, CPU)", yaxis_title="test PR-AUC", height=480)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame({"experiment": [DISPLAY.get(e, e) for e in ex], "parameters": [int(agg.loc[e, "params_mean"]) for e in ex],
                               "train min": [round(agg.loc[e, "train_min_mean"], 1) for e in ex],
                               "inference tx/s": [int(agg.loc[e, "infer_tx_per_s_mean"]) for e in ex],
                               "latency ms (1 tx)": [round(agg.loc[e, "latency_ms_mean"], 1) for e in ex],
                               "peak RSS MB": [int(agg.loc[e, "peak_mem_mb_mean"]) for e in ex]}), hide_index=True, use_container_width=True)
