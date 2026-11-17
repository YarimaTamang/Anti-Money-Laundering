"""Overview page."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components import data  # noqa: E402
from dashboard.components.theme import AQUA, BLUE, ORANGE, disclaimer, kpi_row, setup  # noqa: E402

setup("Overview")
st.title("AML Monitoring Overview")
st.caption("IBM Transactions for Anti-Money-Laundering · HI-Small · chronological test period scored by TRACE-X (seed 0)")
disclaimer()

prof = data.profile()
ts = data.test_scores()
m = data.metrics()
thr = m["test"]["threshold"]
n_alert = int((ts["prob"] >= thr).sum())
k05 = int(np.ceil(0.005 * len(ts)))

kpi_row([
    ("Transactions (all)", f"{prof['n_transactions']:,}" if prof else "–", f"{prof['laundering_pct']:.3f}% labelled laundering" if prof else ""),
    ("Accounts", f"{prof['n_unique_accounts']:,}" if prof else "–", f"{prof['n_unique_banks']:,} banks" if prof else ""),
    ("Test transactions scored", f"{len(ts):,}", f"{ts['label'].sum():,} labelled laundering"),
    ("Alerts at threshold", f"{n_alert:,}", f"p ≥ {thr:.3f} (selected on validation)"),
    ("Test PR-AUC", f"{m['test']['pr_auc']:.3f}", f"ROC-AUC {m['test']['roc_auc']:.3f}"),
    ("Recall @ top 0.5%", f"{m['test_alert_budget']['0.5%']['recall_at_k']:.3f}", f"{k05:,} alerts reviewed"),
])
st.write("")

c1, c2 = st.columns([1.35, 1])
with c1:
    if prof:
        pdays = pd.DataFrame([{"day": d, "transactions": v["n"], "laundering": v["laundering"]} for d, v in prof["per_day"].items()])
        fig = go.Figure()
        fig.add_bar(x=pdays["day"], y=pdays["transactions"], name="transactions", marker_color=BLUE)
        fig.update_layout(title="Transactions per day (all splits)", height=330, yaxis_title="transactions", showlegend=False)
        st.plotly_chart(fig, width="stretch")
        fig = go.Figure()
        fig.add_bar(x=pdays["day"], y=pdays["laundering"], marker_color=ORANGE)
        fig.update_layout(title="Labelled laundering transactions per day", height=260, yaxis_title="laundering")
        st.plotly_chart(fig, width="stretch")
with c2:
    if prof:
        pf = pd.DataFrame([{"format": k, "count": v["count"], "rate": v["laundering_rate_pct"]} for k, v in prof["payment_format"].items()]).sort_values("count")
        fig = go.Figure(go.Bar(x=pf["count"], y=pf["format"], orientation="h", marker_color=BLUE,
                               customdata=pf["rate"], hovertemplate="%{y}: %{x:,} tx<br>laundering rate %{customdata:.3f}%<extra></extra>"))
        fig.update_layout(title="Payment formats", height=300)
        st.plotly_chart(fig, width="stretch")
    fig = go.Figure()
    bins = np.linspace(0, 1, 41)
    for lab, col, name in ((0, BLUE, "normal"), (1, ORANGE, "laundering (benchmark label)")):
        h, _ = np.histogram(ts.loc[ts["label"] == lab, "prob"], bins=bins)
        fig.add_bar(x=(bins[:-1] + bins[1:]) / 2, y=h, name=name, marker_color=col, opacity=0.9)
    fig.add_vline(x=thr, line_dash="dash", line_color="#a9b6cf", annotation_text="alert threshold")
    fig.update_layout(title="Score distribution on the test period", barmode="overlay", yaxis_type="log", height=290,
                      xaxis_title="TRACE-X suspicious score", legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, width="stretch")

st.subheader("Model performance summary")
agg, _ = data.results()
if agg is not None:
    order = [e for e in ["tracex", "gat_lstm", "gat", "graphsage", "gcn", "gru", "lstm"] if e in agg.index]
    from src.evaluation.comparison import DISPLAY

    tab = pd.DataFrame({
        "Model": [DISPLAY[e] for e in order], "Seeds": [int(agg.loc[e, "n_seeds"]) for e in order],
        "PR-AUC": [f"{agg.loc[e, 'pr_auc_mean']:.4f}" + (f" ± {agg.loc[e, 'pr_auc_std']:.4f}" if agg.loc[e, "n_seeds"] > 1 else "") for e in order],
        "ROC-AUC": [f"{agg.loc[e, 'roc_auc_mean']:.4f}" for e in order],
        "F1": [f"{agg.loc[e, 'f1_mean']:.4f}" for e in order],
        "Recall@0.5%": [f"{agg.loc[e, 'r@0.5%_mean']:.4f}" for e in order],
    })
    st.dataframe(tab, hide_index=True, width="stretch")
else:
    st.info("Run `python -m src.evaluation.comparison` to populate model results.")

st.subheader("Highest-risk test transactions")
top = ts.nlargest(25, "prob").merge(data.transactions()[["tx_id", "timestamp", "payment_format", "usd_paid"]], on="tx_id")
acc = data.accounts()
s = data.store()
top["sender"] = acc[s.a["src"][top["eid"]]]
top["receiver"] = acc[s.a["dst"][top["eid"]]]
top["outcome"] = [data.outcome_for(p)[0] for p in top["prob"]]
st.dataframe(top[["tx_id", "timestamp", "sender", "receiver", "payment_format", "usd_paid", "prob", "outcome", "label"]]
             .rename(columns={"prob": "score", "usd_paid": "amount (USD)", "label": "benchmark label"}),
             hide_index=True, width="stretch",
             column_config={"score": st.column_config.ProgressColumn("score", min_value=0, max_value=1, format="%.3f"),
                            "amount (USD)": st.column_config.NumberColumn(format="$%.0f")})
_ = (px, AQUA)
