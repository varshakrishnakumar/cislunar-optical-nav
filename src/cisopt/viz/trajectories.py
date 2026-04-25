"""Trajectory plotters: planar XY overlay used in deck figures."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .style import PALETTE, style_axis


Array = np.ndarray


def plot_trajectory_xy(
    *,
    ax,
    nominal: Array | None = None,
    truth: Array | None = None,
    perfect: Array | None = None,
    ekf: Array | None = None,
    primary_xy: Sequence[float] | None = None,
    secondary_xy: Sequence[float] | None = None,
    target_xy: Sequence[float] | None = None,
    correction_xy: Sequence[float] | None = None,
    title: str | None = None,
) -> None:
    """Overlay planar trajectories. Each trajectory should be (T, >=2)."""
    if nominal is not None:
        ax.plot(nominal[:, 0], nominal[:, 1], lw=1.4, color=PALETTE["cyan"],
                alpha=0.80, label="nominal")
    if truth is not None:
        ax.plot(truth[:, 0], truth[:, 1], lw=1.4, color=PALETTE["amber"],
                ls="--", alpha=0.85, label="truth")
    if perfect is not None:
        ax.plot(perfect[:, 0], perfect[:, 1], lw=1.6, color=PALETTE["green"],
                label="perfect Δv")
    if ekf is not None:
        ax.plot(ekf[:, 0], ekf[:, 1], lw=1.6, color=PALETTE["violet"],
                ls=(0, (5, 3)), label="filter Δv")

    if primary_xy is not None:
        ax.scatter([primary_xy[0]], [primary_xy[1]],
                   s=80, color=PALETTE["earth"], zorder=5, label="primary")
    if secondary_xy is not None:
        ax.scatter([secondary_xy[0]], [secondary_xy[1]],
                   s=55, color=PALETTE["moon"], zorder=5, label="secondary")
    if target_xy is not None:
        ax.scatter([target_xy[0]], [target_xy[1]],
                   s=80, marker="*", color=PALETTE["amber"], zorder=6, label="target")
    if correction_xy is not None:
        ax.scatter([correction_xy[0]], [correction_xy[1]],
                   s=60, color=PALETTE["red"], zorder=6, label="t_c")

    if title:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8)
    style_axis(ax)
