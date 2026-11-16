"""PyVis network rendering + bounded, causal subgraph extraction."""
from __future__ import annotations

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from dashboard.components import data
from dashboard.components.theme import AQUA, BLUE, NAVY, ORANGE, PANEL, TEXT2


def ego_edges(account: int, max_edges: int = 150, hops: int = 2, t_max: int | None = None, per_node: int = 15) -> list[int]:
    """Bounded ego network: most recent transactions (optionally before t_max) around an account."""
    s = data.store()
    ts = s.a["ts"]
    seen_e: list[int] = []
    frontier = [account]
    visited = {account}
    for h in range(hops):
        nxt = []
        for a in frontier:
            e = np.concatenate([data.account_eids(a, "src"), data.account_eids(a, "dst")])
            if t_max is not None:
                e = e[ts[e] < t_max]
            e = e[np.argsort(-ts[e])][: (max_edges if h == 0 else per_node)]
            for x in e:
                seen_e.append(int(x))
                for b in (int(s.a["src"][x]), int(s.a["dst"][x])):
                    if b not in visited:
                        visited.add(b)
                        nxt.append(b)
            if len(seen_e) >= max_edges:
                break
        frontier = nxt[:40]
        if len(seen_e) >= max_edges:
            break
    return list(dict.fromkeys(seen_e))[:max_edges]


def render(edges: list[dict], highlight: set[int] = frozenset(), height: int = 560, key: str = "net") -> None:
    """edges: dicts with src, dst, title, weight (0..1), score (0..1 or nan), target(bool)."""
    acc = data.accounts()
    net = Network(height=f"{height}px", width="100%", directed=True, bgcolor=PANEL, font_color="#e6ebf5", cdn_resources="in_line")
    net.barnes_hut(gravity=-6000, central_gravity=0.25, spring_length=110)
    nodes = set()
    for e in edges:
        nodes.update([e["src"], e["dst"]])
    for n in nodes:
        is_h = n in highlight
        net.add_node(int(n), label=str(acc[n]) if is_h else " ", title=str(acc[n]),
                     color={"background": ORANGE if is_h else "#5b6f94", "border": NAVY},
                     size=22 if is_h else 9, borderWidth=2)
    for e in edges:
        sc = e.get("score", np.nan)
        if e.get("target"):
            col, width = ORANGE, 5
        elif sc == sc and sc is not None:  # not nan
            col = ORANGE if sc >= e.get("thr", 0.5) else BLUE
            width = 1 + 4 * float(e.get("weight", 0.3))
        else:
            col, width = "#41557d", 1 + 4 * float(e.get("weight", 0.3))
        net.add_edge(int(e["src"]), int(e["dst"]), title=e.get("title", ""), color=col, width=width,
                     arrows="to", smooth={"type": "curvedCW", "roundness": 0.15})
    net.set_options('{"interaction": {"hover": true, "tooltipDelay": 80}, "physics": {"stabilization": {"iterations": 150}}}')
    html = net.generate_html(notebook=False)
    components.html(html, height=height + 20, scrolling=False)


def legend() -> None:
    st.markdown(
        f"<div class='small'>"
        f"<span style='color:{ORANGE}'>━</span> target transaction / score ≥ alert threshold &nbsp; "
        f"<span style='color:{BLUE}'>━</span> scored below threshold (test period) &nbsp; "
        f"<span style='color:#41557d'>━</span> not scored (train/validation period) &nbsp; "
        f"<span style='color:{AQUA}'>●</span> width = explanation importance where shown"
        f"</div>", unsafe_allow_html=True)
    _ = TEXT2
