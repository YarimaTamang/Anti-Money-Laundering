import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components import data, explain_live  # noqa: E402
from dashboard.components.theme import AQUA, BLUE, ORANGE, disclaimer, kpi_row, pill, setup  # noqa: E402
from src.explainability.report import account_profile  # noqa: E402

setup("Transaction Investigation")
st.title("Transaction Investigation")
disclaimer()

cases = data.cases()
ts = data.test_scores()
default = cases[0]["tx_id"] if cases else int(ts.nlargest(1, "prob")["tx_id"].iloc[0])
with st.sidebar:
    st.markdown("**Find a transaction**")
    pick = st.selectbox("Explained cases", ["—"] + [f"{c['tx_id']} · {c['kind']} · p={c['prob']:.3f}" for c in cases])
    q = st.text_input("Transaction ID (original CSV row index)", value=str(default) if pick == "—" else pick.split(" · ")[0])
try:
    tx_id = int(q)
except ValueError:
    st.error("Transaction IDs are integers."); st.stop()
eid = data.eid_from_txid(tx_id)
if eid is None:
    st.error(f"Transaction {tx_id} not found."); st.stop()

s = data.store()
df = data.transactions()
row = df.iloc[eid]
acc = data.accounts()
m = data.metrics()
thr = m["test"]["threshold"]
prob = float(data.score_eids(np.array([eid]))[0])
logit = np.log(prob / (1 - prob + 1e-12) + 1e-12)
pcal = 1 / (1 + np.exp(-(m["platt"]["a"] * logit + m["platt"]["b"])))
outcome, kind = data.outcome_for(prob)
split = data.split_name(eid)

st.markdown(f"### Transaction `{tx_id}` &nbsp; {pill('ALERT' if prob >= thr else 'below threshold', 'alert' if prob >= thr else 'ok')} "
            f"&nbsp; {pill(outcome, kind)}", unsafe_allow_html=True)
if split != "test":
    st.caption(f"Note: this transaction belongs to the **{split}** period - the model was {'trained' if split == 'train' else 'selected/calibrated'} on this period, so its score is not an out-of-sample estimate.")
kpi_row([
    ("Suspicious score", f"{prob:.4f}", f"alert threshold {thr:.3f}"),
    ("Calibrated probability", f"{pcal:.4f}", "Platt scaling fitted on val_cal"),
    ("Amount", f"{row.amount_paid:,.2f} {row.currency_paid}", f"≈ ${row.usd_paid:,.2f}"),
    ("Payment format", row.payment_format, f"{'cross-bank' if row.from_bank != row.to_bank else 'same bank'}"),
    ("Timestamp", str(row.timestamp)[:16], f"split: {split}"),
])
st.write("")
state = np.asarray(s.a["node_state"][eid])
c1, c2 = st.columns(2)
for col, side, title, acct in ((c1, 0, "Sender", int(row.src)), (c2, 1, "Receiver", int(row.dst))):
    p = account_profile(state[side], s.node_names)
    with col:
        st.markdown(f"<div class='card'><b>{title}</b> · <code>{acc[acct]}</code><br><span class='small'>state computed strictly before this transaction</span></div>", unsafe_allow_html=True)
        if p["has_history"]:
            st.dataframe(pd.DataFrame({
                "metric": ["outgoing tx", "incoming tx", "distinct receivers", "distinct senders", "out volume (USD)", "in volume (USD)",
                           "mean outgoing (USD)", "minutes since last activity"],
                "value": [f"{p['outgoing_count']:,}", f"{p['incoming_count']:,}", f"{p['unique_receivers']:,}", f"{p['unique_senders']:,}",
                          f"${p['outgoing_volume_usd']:,.0f}", f"${p['incoming_volume_usd']:,.0f}", f"${p['mean_outgoing_usd']:,.0f}",
                          f"{p['minutes_since_last_activity']:,}"]}), hide_index=True, width="stretch")
        else:
            st.info("No earlier activity for this account in the dataset.")

st.subheader("Historical behaviour fed to the LSTMs")
mdl, cfg = data.model()
b = s.build(np.array([eid]), ("seq",), cfg["seq_len"], meta=True)
tabs = st.tabs(["Sender outgoing history", "Receiver incoming history"])
for tab, key, cp_col in ((tabs[0], "seq_s_eid", "dst"), (tabs[1], "seq_r_eid", "src")):
    with tab:
        idx = b[key][0].numpy()
        idx = idx[idx >= 0]
        if not len(idx):
            st.info("No earlier transactions in this direction."); continue
        h = df.iloc[idx]
        t_target = s.a["ts"][eid]
        hist = pd.DataFrame({"tx_id": h["tx_id"].values, "timestamp": h["timestamp"].astype(str).str[:16].values,
                             "minutes before": t_target - s.a["ts"][idx], "counterparty": acc[s.a[cp_col][idx]],
                             "amount (USD)": h["usd_paid"].values, "format": h["payment_format"].values})
        fig = go.Figure(go.Scatter(x=-hist["minutes before"], y=hist["amount (USD)"], mode="lines+markers",
                                   marker=dict(size=9, color=BLUE, line=dict(color="#0a1628", width=2)), line=dict(color=BLUE, width=2),
                                   customdata=hist[["tx_id", "format", "counterparty"]],
                                   hovertemplate="tx %{customdata[0]}<br>%{customdata[1]} to/from %{customdata[2]}<br>$%{y:,.0f}<extra></extra>"))
        fig.add_scatter(x=[0], y=[row.usd_paid], mode="markers", marker=dict(size=13, color=ORANGE, symbol="diamond"), name="target")
        fig.update_layout(height=280, yaxis_type="log", xaxis_title="minutes relative to target", yaxis_title="USD (log)", showlegend=False,
                          title="History (left) and target transaction (orange)")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(hist, hide_index=True, width="stretch", column_config={"amount (USD)": st.column_config.NumberColumn(format="$%.2f")})

st.subheader("Explanations")
ig, src_ig = explain_live.get(eid, "ig")
st.plotly_chart(explain_live.ig_bars(ig), width="stretch")
st.caption(f"Integrated Gradients ({src_ig}). Orange bars push the score towards suspicious.")
g, src_g = explain_live.get(eid, "gnn")
cA, cB = st.columns([1.3, 1])
with cA:
    from dashboard.components import network

    top = sorted(g["edges"], key=lambda e: -e["importance"])[:30]
    pl = data.test_prob_lookup()
    edges = [{"src": int(row.src), "dst": int(row.dst), "target": True, "title": f"target tx {tx_id}"}] + [
        {"src": e["src_account"], "dst": e["dst_account"], "weight": e["importance"], "score": float(pl[e["eid"]]), "thr": thr,
         "title": f"tx {e['tx_id']} · hop {e['hop']} · mask {e['importance']:.2f} · {e['minutes_before_parent']:.0f} min before"} for e in top]
    network.render(edges, {int(row.src), int(row.dst)}, height=420)
    network.legend()
with cB:
    ef = pd.Series(g["edge_feature_importance"]).sort_values(ascending=False).head(10)[::-1]
    fig = go.Figure(go.Bar(x=ef.values, y=ef.index, orientation="h", marker_color=AQUA))
    fig.update_layout(title=f"GNNExplainer edge-feature mask ({src_g})", height=420, xaxis_range=[0, 1])
    st.plotly_chart(fig, width="stretch")
sh, src_sh = explain_live.get(eid, "shap", allow_compute=False)
if sh is None and st.button("Compute Kernel SHAP for this transaction"):
    sh, src_sh = explain_live.get(eid, "shap")
if sh is not None:
    st.plotly_chart(explain_live.shap_bar(sh), width="stretch")
    st.caption(f"Kernel SHAP ({src_sh}); history and neighbourhood held fixed. Attention weights are not used as explanations.")
