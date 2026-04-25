"""Plot styling: paper + dark themes plus a reusable colour palette."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


# Palette — same hues as src/visualization/style.py so figures from the new
# framework match the existing deck visuals. Keys are semantic.
PALETTE: dict[str, str] = {
    "bg": "#080B14",
    "panel": "#0E1220",
    "border": "#1C2340",
    "text": "#DCE0EC",
    "dim": "#5A6080",
    "cyan": "#22D3EE",
    "amber": "#F59E0B",
    "green": "#10B981",
    "red": "#F43F5E",
    "violet": "#8B5CF6",
    "orange": "#FB923C",
    "earth": "#3B82F6",
    "moon": "#C8CDD8",
}

# Cycle used by the Estimator zoo (EKF/IEKF/UKF) and by ablation grids.
_DEFAULT_CYCLE = ["#22D3EE", "#F59E0B", "#8B5CF6", "#10B981", "#F43F5E", "#FB923C"]


def apply_dark_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["panel"],
        "axes.edgecolor":    PALETTE["border"],
        "axes.labelcolor":   PALETTE["text"],
        "axes.titlecolor":   PALETTE["text"],
        "text.color":        PALETTE["text"],
        "xtick.color":       PALETTE["text"],
        "ytick.color":       PALETTE["text"],
        "grid.color":        PALETTE["border"],
        "grid.linestyle":    "--",
        "grid.alpha":        0.45,
        "axes.prop_cycle":   mpl.cycler(color=_DEFAULT_CYCLE),
        "savefig.facecolor": PALETTE["bg"],
        "savefig.edgecolor": PALETTE["bg"],
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "legend.fontsize":   9,
        "font.family":       "sans-serif",
    })


def apply_paper_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "#222222",
        "axes.labelcolor":   "#111111",
        "axes.titlecolor":   "#111111",
        "text.color":        "#111111",
        "xtick.color":       "#222222",
        "ytick.color":       "#222222",
        "grid.color":        "#CFCFCF",
        "grid.linestyle":    ":",
        "grid.alpha":        0.7,
        "axes.prop_cycle":   mpl.cycler(color=_DEFAULT_CYCLE),
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "legend.fontsize":   9,
        "font.family":       "serif",
    })


def style_axis(ax) -> None:
    """Light axis cleanup applied after a plotter draws data."""
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
