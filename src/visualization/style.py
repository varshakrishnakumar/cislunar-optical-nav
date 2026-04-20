from __future__ import annotations

import os
from pathlib import Path

import numpy as np


BG = "#080B14"
PANEL = "#0E1220"
BORDER = "#1C2340"
TEXT = "#DCE0EC"
DIM = "#5A6080"
CYAN = "#22D3EE"
AMBER = "#F59E0B"
GREEN = "#10B981"
RED = "#F43F5E"
VIOLET = "#8B5CF6"
ORANGE = "#FB923C"
MOON = "#C8CDD8"
EARTH = "#3B82F6"


_DEFAULT_MPL_CACHE = Path.home() / ".cache" / "cislunar_optical_nav" / "mpl"
_DEFAULT_XDG_CACHE = Path.home() / ".cache" / "cislunar_optical_nav"


def ensure_batch_matplotlib_backend(
    *,
    cache_dir: str | Path | None = None,
    xdg_cache_dir: str | Path | None = None,
) -> None:
    # Prefer ~/.cache over /tmp so the font cache survives reboots.
    cache_path = Path(cache_dir) if cache_dir is not None else _DEFAULT_MPL_CACHE
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_path))

    xdg_path = Path(xdg_cache_dir) if xdg_cache_dir is not None else _DEFAULT_XDG_CACHE
    xdg_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_path))

    import matplotlib

    matplotlib.use("Agg", force=True)


ensure_batch_matplotlib_backend()

import matplotlib.pyplot as plt  # noqa: E402


def apply_dark_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": BORDER,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "text.color": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "grid.color": BORDER,
            "grid.alpha": 1.0,
            "grid.linestyle": "--",
            "lines.linewidth": 2.0,
            "legend.facecolor": PANEL,
            "legend.edgecolor": BORDER,
            "legend.labelcolor": TEXT,
            "savefig.facecolor": BG,
            "savefig.edgecolor": BG,
            "font.size": 11,
        }
    )


def style_axis(ax, *, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(True)
    if title is not None:
        ax.set_title(title, color=TEXT)
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=TEXT)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=TEXT)


def plot_xy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    outpath: str | Path,
    color: str = CYAN,
    marker_color: str = AMBER,
    logx: bool = False,
    logy: bool = False,
) -> Path:
    apply_dark_theme()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(BG)
    style_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.plot(x, y, color=color, lw=2.0, zorder=3)
    ax.scatter(x, y, s=50, color=marker_color, zorder=4, edgecolors=BG, lw=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, facecolor=BG)
    plt.close(fig)
    return outpath


def plot_xy_band(
    x: np.ndarray,
    y_med: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    outpath: str | Path,
    color: str = CYAN,
    marker_color: str = AMBER,
    band_label: str = "p05-p95 band",
    logx: bool = False,
    logy: bool = False,
) -> Path:
    apply_dark_theme()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(BG)
    style_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.fill_between(x, y_lo, y_hi, color=color, alpha=0.18, label=band_label)
    ax.plot(x, y_med, color=color, lw=2.0, zorder=3, label="median")
    ax.scatter(x, y_med, s=50, color=marker_color, zorder=5, edgecolors=BG, lw=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, facecolor=BG)
    plt.close(fig)
    return outpath


def plot_xy_with_err(
    x: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    outpath: str | Path,
    color: str = CYAN,
    marker_color: str = AMBER,
) -> Path:
    apply_dark_theme()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(BG)
    style_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax.fill_between(
        x,
        y_mean - y_std,
        y_mean + y_std,
        color=color,
        alpha=0.18,
        label="+/-1 sigma band",
    )
    ax.plot(x, y_mean, color=color, lw=2.0, zorder=3, label="mean")
    ax.errorbar(
        x,
        y_mean,
        yerr=y_std,
        fmt="none",
        ecolor=marker_color,
        capsize=4,
        elinewidth=1.0,
        zorder=4,
    )
    ax.scatter(x, y_mean, s=50, color=marker_color, zorder=5, edgecolors=BG, lw=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, facecolor=BG)
    plt.close(fig)
    return outpath
