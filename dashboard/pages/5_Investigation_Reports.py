import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from dashboard.components import data, explain_live  # noqa: E402
from dashboard.components.theme import disclaimer, setup  # noqa: E402
from src.explainability.report import to_markdown  # noqa: E402

setup("Investigation Reports")
st.title("Investigation Reports")
disclaimer()
cases = data.cases()
c1, c2 = st.columns([2, 1])
with c1:
    opts = [f"{c['tx_id']} · {c['kind']} · p={c['prob']:.3f}" for c in cases]
    pick = st.selectbox("Explained case", opts) if opts else None
    q = st.text_input("…or any transaction ID", "")
with c2:
    with_shap = st.checkbox("Include Kernel SHAP (computed live if not precomputed; slower)", value=True)
tx = q.strip() or (pick.split(" · ")[0] if pick else "")
if not tx:
    st.stop()
eid = data.eid_from_txid(int(tx))
if eid is None:
    st.error("Transaction not found"); st.stop()
if st.button("Generate report", type="primary") or "report_md" in st.session_state and st.session_state.get("report_tx") == tx:
    if st.session_state.get("report_tx") != tx or "report_md" not in st.session_state:
        s = explain_live.summary_for(eid, with_shap)
        st.session_state["report_md"] = to_markdown(s)
        st.session_state["report_tx"] = tx
    md = st.session_state["report_md"]
    st.markdown(md)
    st.download_button("Download report (Markdown)", md, file_name=f"tracex_report_{tx}.md", mime="text/markdown")
    try:
        import markdown as _md

        html = _md.markdown(md, extensions=["tables"])
    except Exception:
        html = "<pre>" + md + "</pre>"
    html = f"<html><head><meta charset='utf-8'><title>TRACE-X report {tx}</title></head><body style='font-family:sans-serif;max-width:900px;margin:auto'>{html}</body></html>"
    st.download_button("Download report (HTML)", html, file_name=f"tracex_report_{tx}.html", mime="text/html")
