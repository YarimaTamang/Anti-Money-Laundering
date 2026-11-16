import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components import data, network  # noqa: E402
from dashboard.components.theme import disclaimer, kpi_row, setup  # noqa: E402

setup("Network Explorer")
st.title("Network Explorer")
st.caption("Bounded ego-network around an account (most recent transactions first) - large graphs are never rendered in full.")
disclaimer()

acc = data.accounts()
ts = data.test_scores()
thr = data.metrics()["test"]["threshold"]
s = data.store()
top_acc = pd.Series(s.a["src"][ts.nlargest(50, "prob")["eid"].to_numpy()]).drop_duplicates().head(15)
with st.sidebar:
    st.markdown("**Graph controls**")
    choice = st.selectbox("Account (sender of a high-risk test transaction)", [acc[a] for a in top_acc])
    custom = st.text_input("…or account id (bank_account)", "")
    hops = st.slider("Hops", 1, 2, 2)
    max_edges = st.slider("Max transactions", 20, 400, 120, step=20)
    causal = st.checkbox("Only transactions before the account's highest-risk test transaction", value=True)
    only_flag = st.checkbox("Show only edges scored ≥ alert threshold (plus first-hop context)", value=False)
    show_labels = st.checkbox("Colour by benchmark label instead of model score", value=False)

name = custom.strip() or choice
idx = np.flatnonzero(acc == name)
if not len(idx):
    st.error(f"Account {name} not found"); st.stop()
a = int(idx[0])
mine = np.concatenate([data.account_eids(a, "src"), data.account_eids(a, "dst")])
pl = data.test_prob_lookup()
t_max = None
if causal:
    scored = mine[~np.isnan(pl[mine])]
    if len(scored):
        anchor = scored[np.nanargmax(pl[scored])]
        t_max = int(s.a["ts"][anchor]) + 1
eids = network.ego_edges(a, max_edges=max_edges, hops=hops, t_max=t_max)
df = data.transactions()
lab = s.a["label"]
edges = []
for e in eids:
    sc = float(pl[e])
    if only_flag and not (sc >= thr) and a not in (int(s.a["src"][e]), int(s.a["dst"][e])):
        continue
    edges.append({"src": int(s.a["src"][e]), "dst": int(s.a["dst"][e]),
                  "score": float(lab[e]) if show_labels else sc, "thr": 0.5 if show_labels else thr, "weight": 0.25,
                  "title": f"tx {int(s.a['tx_id'][e])} · {df['payment_format'].iat[e]} · ${df['usd_paid'].iat[e]:,.0f} · "
                           f"{str(df['timestamp'].iat[e])[:16]} · score {'n/a' if np.isnan(sc) else f'{sc:.3f}'}"})
n_acc = len({x for e in edges for x in (e["src"], e["dst"])})
flagged = sum(1 for e in edges if e["score"] == e["score"] and e["score"] >= e["thr"])
kpi_row([("Account", name, f"{len(mine):,} transactions in total"), ("Rendered transactions", f"{len(edges):,}", f"{hops}-hop, bounded"),
         ("Accounts in view", f"{n_acc:,}", ""), ("Flagged edges", f"{flagged:,}", "label = 1" if show_labels else "score ≥ threshold")])
network.render(edges, {a}, height=620)
network.legend()
with st.expander("Transactions in view"):
    view = df.iloc[eids][["tx_id", "timestamp", "payment_format", "usd_paid", "label"]].copy()
    view["sender"] = acc[s.a["src"][eids]]; view["receiver"] = acc[s.a["dst"][eids]]; view["score"] = pl[eids]
    st.dataframe(view, hide_index=True, use_container_width=True)
