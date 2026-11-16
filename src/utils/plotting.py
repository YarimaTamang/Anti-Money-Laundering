"""Shared static-figure style (validated categorical palette, fixed order, thin marks)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NORMAL, SUSPICIOUS = SERIES[0], SERIES[1]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT, "axes.titleweight": "bold",
        "axes.titlesize": 11, "axes.labelsize": 9, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False,
        "legend.fontsize": 8, "font.family": "DejaVu Sans", "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "figure.dpi": 110, "savefig.dpi": 140, "savefig.bbox": "tight",
    })


def save(fig, path) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
