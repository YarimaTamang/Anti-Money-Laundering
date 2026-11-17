import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components import data, explain_live, network  # noqa: E402
from dashboard.components.theme import AQUA, BLUE, ORANGE, disclaimer, setup  # noqa: E402

setup("Explainable AI")
st.title("Explainable AI")
st.caption("GNNExplainer (graph evidence) · Kernel SHAP (tabular features) · Integrated Gradients (transaction histories). "
           "Attention weights are not presented as explanations.")
disclaimer()

tab_g, tab_l, tab_q = st.tabs(["Global explanations", "Local explanations", "Explanation quality"])
with tab_g:
    gs = data.json_report("explanations/global_shap.json")
    if gs is None:
        st.info("Run `python scripts/explain.py` to generate global SHAP values.")
    else:
        S, X, names = np.array(gs["shap"]), np.array(gs["x"]), gs["names"]
        ma = np.abs(S).mean(0); o = np.argsort(ma)[-15:]
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(x=ma[o], y=[names[i] for i in o], orientation="h", marker_color=BLUE))
            fig.update_layout(title=f"Global importance: mean |SHAP| over {len(S)} test transactions", height=520)
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = go.Figure()
            for r, i in enumerate(o):
                xv = X[:, i]
                z = (xv - xv.min()) / (np.ptp(xv) + 1e-9)
                fig.add_scatter(x=S[:, i], y=np.full(len(S), r) + np.random.default_rng(i).uniform(-0.25, 0.25, len(S)), mode="markers",
                                marker=dict(size=7, color=z, colorscale=[[0, BLUE], [1, ORANGE]], line=dict(width=1, color="#0a1628")),
                                showlegend=False, hovertemplate=f"{names[i]}<br>SHAP %{{x:+.3f}}<extra></extra>")
            fig.update_layout(title="SHAP summary (colour: low → high feature value)", height=520,
                              yaxis=dict(tickvals=list(range(len(o))), ticktext=[names[i] for i in o]), xaxis_title="SHAP (logit)")
            st.plotly_chart(fig, width="stretch")
        lbl = np.array(gs["labels"])
        diff = pd.DataFrame({"feature": names, "mean SHAP | laundering": S[lbl == 1].mean(0), "mean SHAP | normal": S[lbl == 0].mean(0)})
        st.dataframe(diff.reindex(diff["mean SHAP | laundering"].abs().sort_values(ascending=False).index).head(12),
                     hide_index=True, width="stretch")

with tab_l:
    cases = data.cases()
    if not cases:
        st.info("No precomputed cases yet.")
    else:
        pick = st.selectbox("Case", [f"{c['tx_id']} · {c['kind']} · p={c['prob']:.3f}" for c in cases])
        c = data.case(int(pick.split(" · ")[0]))
        s = data.store()
        st.markdown("**Main findings**")
        for f in c["summary"]["findings"]:
            st.markdown(f"- {f}")
        c1, c2 = st.columns([1.25, 1])
        with c1:
            g = c["gnn"]
            top = sorted(g["edges"], key=lambda e: -e["importance"])[:30]
            edges = [{"src": g["sender_account"], "dst": g["receiver_account"], "target": True, "title": "target"}] + [
                {"src": e["src_account"], "dst": e["dst_account"], "weight": e["importance"], "score": e["importance"], "thr": 0.5,
                 "title": f"tx {e['tx_id']} · hop {e['hop']} · mask {e['importance']:.2f}"} for e in top]
            network.render(edges, {g["sender_account"], g["receiver_account"]}, height=430)
            st.caption("GNNExplainer subgraph: orange edges have mask ≥ 0.5; edge width ∝ mask importance.")
        with c2:
            st.plotly_chart(explain_live.shap_bar(c["shap"]), width="stretch")
        st.plotly_chart(explain_live.ig_bars(c["ig"]), width="stretch")
        f = c["fidelity"]
        st.markdown(f"**Fidelity for this case** - GNN fid+ (Δlogit) {f['gnn'].get('fid_plus', float('nan')):+.3f} (random {f['gnn'].get('fid_plus_random', float('nan')):+.3f}); "
                    f"IG top-3 deletion Δlogit {f['ig'].get('drop_topk', float('nan')):+.3f} (random {f['ig'].get('drop_random_k', float('nan')):+.3f}); "
                    f"SHAP top-5 deletion Δlogit {f['shap']['drop_topk']:+.3f} (random {f['shap']['drop_random_k']:+.3f})")

with tab_q:
    q = data.json_report("explanations/quality.json")
    if q is None:
        st.info("Run `python scripts/explain.py`.")
    else:
        rows = []
        for meth, keys in (("gnn_explainer", ["fid_plus", "fid_plus_random", "fid_minus", "fid_minus_random", "sparsity_frac_mask_gt_0.5"]),
                           ("integrated_gradients", ["drop_topk", "drop_random_k", "sparsity_top3_mass"]),
                           ("shap", ["drop_topk", "drop_random_k", "sparsity_top5_mass"])):
            for k in keys:
                v = q[meth].get(k)
                if v:
                    rows.append({"method": meth, "metric": k, "mean": v["mean"], "std": v["std"], "n": v["n"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if q.get("stability_across_seeds"):
            stb = q["stability_across_seeds"]
            st.markdown(f"**Stability across seeds {stb.get('seeds')}** (same transactions, independently trained models)")
            st.dataframe(pd.DataFrame([{"metric": k, **v} for k, v in stb.items() if isinstance(v, dict)]), hide_index=True, width="stretch")
        _ = AQUA
