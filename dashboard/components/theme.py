"""Dark-navy visual theme: CSS, Plotly template and small UI widgets."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

NAVY, PANEL, PANEL2, BORDER = "#0a1628", "#11213d", "#162a4d", "#22375e"
TEXT, TEXT2, MUTED = "#e6ebf5", "#a9b6cf", "#7d8ba6"
# categorical palette - dark-surface steps of the validated reference palette, fixed order
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
BLUE, ORANGE, AQUA = SERIES[0], SERIES[1], SERIES[2]
DISCLAIMER = "Decision-support alert, not proof of money laundering."

CSS = f"""
<style>
html, body, [class*="css"] {{ font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px; }}
h1, h2, h3 {{ color: {TEXT}; letter-spacing: -0.01em; }}
h1 {{ font-weight: 700; font-size: 1.75rem !important; }}
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0d1c36 0%, #0a1628 100%); border-right: 1px solid {BORDER}; }}
.kpi {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; padding: 14px 16px; height: 100%; }}
.kpi .label {{ color: {MUTED}; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; }}
.kpi .value {{ color: {TEXT}; font-size: 1.55rem; font-weight: 700; margin-top: 2px; font-variant-numeric: tabular-nums; }}
.kpi .sub {{ color: {TEXT2}; font-size: 0.78rem; margin-top: 2px; }}
.disclaimer {{ background: rgba(201,133,0,0.12); border: 1px solid rgba(201,133,0,0.55); color: #f1d9a6;
  border-radius: 8px; padding: 8px 12px; font-size: 0.85rem; margin: 6px 0 14px 0; }}
.pill {{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }}
.pill.alert {{ background: rgba(217,89,38,0.18); color: #ff9b70; border: 1px solid rgba(217,89,38,0.6); }}
.pill.ok {{ background: rgba(25,158,112,0.16); color: #5fd3a6; border: 1px solid rgba(25,158,112,0.6); }}
.pill.review {{ background: rgba(144,133,233,0.16); color: #bdb5ff; border: 1px solid rgba(144,133,233,0.6); }}
.card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; }}
.small {{ color: {TEXT2}; font-size: 0.82rem; }}
div[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; padding: 10px 14px; }}
</style>
"""


def setup(title: str, icon: str = "🛡️") -> None:
    st.set_page_config(page_title=f"TRACE-X · {title}", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    _template()
    with st.sidebar:
        st.markdown(f"<div style='font-weight:800;font-size:1.25rem;color:{TEXT}'>TRACE-X</div>"
                    f"<div class='small'>Explainable temporal graph-deep-learning AML detection</div>", unsafe_allow_html=True)
        st.markdown("---")


def disclaimer() -> None:
    st.markdown(f"<div class='disclaimer'>⚠️ <b>{DISCLAIMER}</b> Scores rank transactions for human review on a synthetic benchmark (IBM AML HI-Small).</div>",
                unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "") -> str:
    return f"<div class='kpi'><div class='label'>{label}</div><div class='value'>{value}</div><div class='sub'>{sub}</div></div>"


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for c, (l, v, s) in zip(cols, items):
        c.markdown(kpi(l, v, s), unsafe_allow_html=True)


def pill(text: str, kind: str) -> str:
    return f"<span class='pill {kind}'>{text}</span>"


def _template() -> None:
    if "tracex" in pio.templates:
        pio.templates.default = "tracex"
        return
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TEXT2, size=12),
        title=dict(font=dict(color=TEXT, size=14)), colorway=SERIES,
        xaxis=dict(gridcolor="#1e3157", zerolinecolor="#2b4372", linecolor=BORDER),
        yaxis=dict(gridcolor="#1e3157", zerolinecolor="#2b4372", linecolor=BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=40, r=20, t=50, b=40),
        hoverlabel=dict(bgcolor=PANEL2, bordercolor=BORDER, font=dict(color=TEXT)),
    )
    pio.templates["tracex"] = t
    pio.templates.default = "tracex"
