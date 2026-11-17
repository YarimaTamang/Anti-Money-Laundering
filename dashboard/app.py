"""TRACE-X dashboard entry point.   Run:  streamlit run dashboard/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

pages = [
    st.Page("pages/0_Overview.py", title="Overview", icon="📊", default=True),
    st.Page("pages/1_Transaction_Investigation.py", title="Transaction Investigation", icon="🔎"),
    st.Page("pages/2_Network_Explorer.py", title="Network Explorer", icon="🕸️"),
    st.Page("pages/3_Explainable_AI.py", title="Explainable AI", icon="🧠"),
    st.Page("pages/4_Model_Laboratory.py", title="Model Laboratory", icon="🧪"),
    st.Page("pages/5_Investigation_Reports.py", title="Investigation Reports", icon="📄"),
]
st.navigation(pages).run()
