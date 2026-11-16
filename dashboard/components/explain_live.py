"""On-demand explanations for any transaction (same code as the offline pipeline)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import data
from dashboard.components.theme import AQUA, BLUE, ORANGE
from src.explainability import gnn_explainer, integrated_gradients, report, shap_explainer
from src.explainability.common import Explainable


def explainable() -> Explainable:
    m, cfg = data.model()
    return Explainable(m, data.store(), cfg)


@st.cache_data(show_spinner="Running GNNExplainer…", max_entries=64)
def gnn(eid: int) -> dict:
    g = gnn_explainer.explain(explainable(), eid, epochs=150)
    g.pop("edge_mask", None)
    return g


@st.cache_data(show_spinner="Running Integrated Gradients…", max_entries=64)
def ig(eid: int) -> dict:
    return integrated_gradients.explain(explainable(), eid)


@st.cache_data(show_spinner="Running Kernel SHAP (≈20–60 s on CPU)…", max_entries=32)
def shap(eid: int, nsamples: int = 200) -> dict:
    view = shap_explainer.TabularView(explainable())
    bg = shap_explainer.background(view, 12)
    r = shap_explainer.explain(view, eid, bg, nsamples=nsamples)
    r["tx_id"] = int(data.store().a["tx_id"][eid])
    return r


def get(eid: int, method: str, allow_compute: bool = True):
    """Prefer the precomputed offline explanation; otherwise compute live."""
    c = data.case(int(data.store().a["tx_id"][eid]))
    key = {"gnn": "gnn", "ig": "ig", "shap": "shap"}[method]
    if c is not None and key in c:
        return c[key], "precomputed"
    if not allow_compute:
        return None, None
    return {"gnn": gnn, "ig": ig, "shap": shap}[method](eid), "computed live"


def shap_bar(sh: dict, k: int = 12) -> go.Figure:
    sv = np.asarray(sh["shap"]); o = np.argsort(-np.abs(sv))[:k][::-1]
    fig = go.Figure(go.Bar(x=sv[o], y=[sh["names"][i] for i in o], orientation="h",
                           marker_color=[ORANGE if sv[i] > 0 else BLUE for i in o],
                           hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    fig.update_layout(title=f"Kernel SHAP (logit units) — base {sh['expected_value']:.2f} → {sh['logit']:.2f}", height=420)
    return fig


def ig_bars(r: dict) -> go.Figure:
    from plotly.subplots import make_subplots

    fig = make_subplots(1, 2, subplot_titles=("Sender outgoing history", "Receiver incoming history"), shared_yaxes=True)
    for col, key, base in ((1, "sender_steps", BLUE), (2, "receiver_steps", AQUA)):
        st_ = r[key]
        fig.add_bar(x=[f"−{s['minutes_before_target']}m" for s in st_], y=[s["attribution"] for s in st_],
                    marker_color=[ORANGE if s["attribution"] > 0 else base for s in st_],
                    customdata=[[s["tx_id"], ", ".join(f"{n} {v:+.2f}" for n, v in s["top_features"])] for s in st_],
                    hovertemplate="tx %{customdata[0]}<br>attribution %{y:+.3f}<br>%{customdata[1]}<extra></extra>",
                    row=1, col=col, showlegend=False)
    fig.update_layout(title=f"Integrated Gradients (oldest → newest; completeness Δ={r['completeness_delta']:.1e})", height=360)
    return fig


def summary_for(eid: int, with_shap: bool) -> dict:
    s = data.store()
    df = data.transactions()
    row = df.iloc[eid]
    acc = data.accounts()
    prob = float(data.score_eids(np.array([eid]))[0])
    m = data.metrics()
    logit = float(np.log(prob / (1 - prob + 1e-12) + 1e-12))
    pc = float(1 / (1 + np.exp(-(m["platt"]["a"] * logit + m["platt"]["b"]))))
    g, _ = get(eid, "gnn")
    i, _ = get(eid, "ig")
    sh = get(eid, "shap", allow_compute=with_shap)[0]
    state = np.asarray(s.a["node_state"][eid])
    tx = {"tx_id": int(row.tx_id), "timestamp": str(row.timestamp), "sender": acc[int(row.src)], "receiver": acc[int(row.dst)],
          "amount_paid": float(row.amount_paid), "currency_paid": row.currency_paid, "amount_received": float(row.amount_received),
          "currency_received": row.currency_received, "usd_paid": float(row.usd_paid), "payment_format": row.payment_format,
          "ground_truth_label": int(row.label)}
    outcome, _ = data.outcome_for(prob)
    return report.build_summary(tx, prob, pc, report.account_profile(state[0], s.node_names),
                                report.account_profile(state[1], s.node_names), g, sh, i, list(acc),
                                uncertainty={"conformal_outcome": outcome, "mc_std": float("nan")},
                                threshold=m["test"]["threshold"])
