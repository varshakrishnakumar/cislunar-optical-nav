"""Distribution plots: histogram, CDF, KDE, box, scatter.

Every plotter takes either a list of dicts (Parquet rows) or numpy arrays.
Use ``style_axis`` after each draw to keep theme effects consistent.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from .style import PALETTE, style_axis


Array = np.ndarray


def _values(rows_or_array: Iterable, metric: str | None) -> Array:
    if metric is None:
        arr = np.asarray(list(rows_or_array), dtype=float).reshape(-1)
    else:
        arr = np.asarray(
            [r[metric] for r in rows_or_array if metric in r], dtype=float,
        )
    return arr[np.isfinite(arr)]


def _grouped(rows: Iterable[dict], metric: str, by: str) -> dict[Any, Array]:
    out: dict[Any, list[float]] = {}
    for r in rows:
        if metric in r and by in r:
            out.setdefault(r[by], []).append(float(r[metric]))
    return {k: np.asarray(v, dtype=float)[np.isfinite(v)] for k, v in out.items()}


def plot_hist(
    rows: Iterable[dict] | Iterable[float],
    metric: str | None = None,
    *,
    ax,
    bins: int = 20,
    color: str = None,
    log_x: bool = False,
    show_stats: bool = True,
    label: str | None = None,
    alpha: float = 0.75,
) -> dict[str, float]:
    vals = _values(rows, metric)
    if vals.size == 0:
        ax.text(0.5, 0.5, "no finite data", ha="center", va="center",
                transform=ax.transAxes, color=PALETTE["dim"])
        style_axis(ax)
        return {"n": 0, "mean": float("nan"), "std": float("nan")}

    if log_x:
        vals_pos = vals[vals > 0]
        if vals_pos.size > 1:
            edges = np.logspace(np.log10(vals_pos.min()), np.log10(vals_pos.max()), bins + 1)
            ax.set_xscale("log")
        else:
            edges = bins
    else:
        edges = bins

    color = color or PALETTE["cyan"]
    ax.hist(vals, bins=edges, color=color, alpha=alpha,
            edgecolor=PALETTE["border"], lw=0.6, label=label)

    stats = {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
    }
    if show_stats:
        ax.axvline(stats["mean"], color=PALETTE["amber"], lw=1.2, ls="--",
                   label=f"mean = {stats['mean']:.3g}")
        ax.axvline(stats["median"], color=PALETTE["green"], lw=1.0, ls=":",
                   label=f"median = {stats['median']:.3g}")
        ax.legend(loc="best", fontsize=8)

    if metric:
        ax.set_xlabel(metric)
    ax.set_ylabel("count")
    style_axis(ax)
    return stats


def plot_cdf(
    rows: Iterable[dict] | Iterable[float],
    metric: str | None = None,
    *,
    ax,
    by: str | None = None,
    log_x: bool = False,
    label: str | None = None,
    legend_outside: bool = False,
) -> None:
    if by is None:
        groups = {label or (metric or "data"): _values(rows, metric)}
    else:
        groups = _grouped(list(rows), metric, by)

    colors = list(PALETTE.values())[5:]  # skip text/border/dim/bg
    plotted_any = False
    for i, (name, vals) in enumerate(sorted(groups.items(), key=lambda x: str(x[0]))):
        if vals.size == 0:
            continue
        sv = np.sort(vals)
        # Anchor the empirical CDF at 0 with a leading point so the curve
        # starts from F=0 instead of F=1/n -- otherwise small samples make
        # matplotlib auto-trim the y-axis around the first observation.
        x = np.concatenate([[sv[0]], sv])
        F = np.concatenate([[0.0], np.arange(1, sv.size + 1) / sv.size])
        ax.plot(x, F, lw=1.6, color=colors[i % len(colors)],
                label=str(name), drawstyle="steps-post")
        plotted_any = True

    if log_x:
        ax.set_xscale("log")
    if metric:
        ax.set_xlabel(metric)
    ax.set_ylabel("F(x)")
    ax.set_ylim(-0.02, 1.04)
    if plotted_any and len(groups) > 1:
        if legend_outside:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                      fontsize=7, frameon=True)
        else:
            ax.legend(loc="lower right", fontsize=8)
    style_axis(ax)


def plot_box(
    rows: Iterable[dict],
    metric: str,
    by: str = "combo_id",
    *,
    ax,
    log_y: bool = False,
    rotate_labels: int = 30,
) -> None:
    groups = _grouped(list(rows), metric, by)
    keys = sorted(groups.keys(), key=lambda x: str(x))
    data = [groups[k] for k in keys]
    if not data:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color=PALETTE["dim"])
        style_axis(ax)
        return

    bp = ax.boxplot(
        data, tick_labels=[str(k) for k in keys],
        patch_artist=True, widths=0.6,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE["cyan"])
        patch.set_alpha(0.55)
        patch.set_edgecolor(PALETTE["border"])
    for whisker in bp["whiskers"]:
        whisker.set_color(PALETTE["dim"])
    for med in bp["medians"]:
        med.set_color(PALETTE["amber"])
        med.set_linewidth(1.5)

    if log_y:
        ax.set_yscale("log")
    if rotate_labels:
        for t in ax.get_xticklabels():
            t.set_rotation(rotate_labels)
            t.set_ha("right")
    ax.set_ylabel(metric)
    ax.set_xlabel(by)
    style_axis(ax)


def plot_kde(
    rows: Iterable[dict] | Iterable[float],
    metric: str | None = None,
    *,
    ax,
    by: str | None = None,
    log_x: bool = False,
    n_points: int = 256,
) -> None:
    from scipy.stats import gaussian_kde

    if by is None:
        groups = {metric or "data": _values(rows, metric)}
    else:
        groups = _grouped(list(rows), metric, by)

    colors = list(PALETTE.values())[5:]
    for i, (name, vals) in enumerate(sorted(groups.items(), key=lambda x: str(x[0]))):
        if vals.size < 3:
            continue
        if log_x:
            v = np.log10(vals[vals > 0])
            kde = gaussian_kde(v)
            x = np.linspace(v.min(), v.max(), n_points)
            ax.plot(10 ** x, kde(x), lw=1.6, color=colors[i % len(colors)], label=str(name))
        else:
            kde = gaussian_kde(vals)
            x = np.linspace(vals.min(), vals.max(), n_points)
            ax.plot(x, kde(x), lw=1.6, color=colors[i % len(colors)], label=str(name))

    if log_x:
        ax.set_xscale("log")
    if metric:
        ax.set_xlabel(metric)
    ax.set_ylabel("density")
    if len(groups) > 1:
        ax.legend(loc="best", fontsize=8)
    style_axis(ax)


def plot_scatter(
    rows: Iterable[dict],
    x_metric: str,
    y_metric: str,
    *,
    ax,
    by: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
    s: int = 18,
) -> None:
    rows = list(rows)
    if by is None:
        xs = np.asarray([r[x_metric] for r in rows], dtype=float)
        ys = np.asarray([r[y_metric] for r in rows], dtype=float)
        m = np.isfinite(xs) & np.isfinite(ys)
        ax.scatter(xs[m], ys[m], s=s, alpha=0.65,
                   edgecolor=PALETTE["border"], lw=0.4, color=PALETTE["cyan"])
    else:
        groups: dict[Any, tuple[list[float], list[float]]] = {}
        for r in rows:
            key = r.get(by)
            xs, ys = groups.setdefault(key, ([], []))
            xs.append(float(r[x_metric]))
            ys.append(float(r[y_metric]))
        colors = list(PALETTE.values())[5:]
        for i, (name, (xs, ys)) in enumerate(sorted(groups.items(), key=lambda x: str(x[0]))):
            xv = np.asarray(xs); yv = np.asarray(ys)
            m = np.isfinite(xv) & np.isfinite(yv)
            ax.scatter(xv[m], yv[m], s=s, alpha=0.65,
                       color=colors[i % len(colors)], label=str(name),
                       edgecolor=PALETTE["border"], lw=0.4)
        ax.legend(loc="best", fontsize=8)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    style_axis(ax)
