"""Standardised plotting API for the cisopt framework (item 10).

The legacy 06_monte_carlo.py plots stay the source of truth for the deck
build; this module gives the new framework its own consistent figure style
so Parquet results can be turned into paper-ready figures with one call.

Two themes are exposed: a dark theme that mirrors the existing palette
(deck-friendly) and a paper theme (light bg, large fonts, distinct colours)
for academic figures. Apply one before plotting; both are idempotent.
"""

from .distributions import (
    plot_box,
    plot_cdf,
    plot_hist,
    plot_kde,
    plot_scatter,
)
from .observability import plot_gramian_eigvals, plot_observability_summary
from .reports import build_report
from .style import (
    PALETTE,
    apply_dark_theme,
    apply_paper_theme,
    style_axis,
)
from .trajectories import plot_trajectory_xy

__all__ = [
    "PALETTE",
    "apply_dark_theme",
    "apply_paper_theme",
    "build_report",
    "plot_box",
    "plot_cdf",
    "plot_gramian_eigvals",
    "plot_hist",
    "plot_kde",
    "plot_observability_summary",
    "plot_scatter",
    "plot_trajectory_xy",
    "style_axis",
]
