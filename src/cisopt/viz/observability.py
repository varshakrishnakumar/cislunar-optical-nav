"""Observability plots: eigenvalue evolution + weak-direction summary."""

from __future__ import annotations

import numpy as np

from .style import PALETTE, style_axis


Array = np.ndarray


def plot_gramian_eigvals(
    eig_hist: Array,
    *,
    ax,
    t_axis: Array | None = None,
    log_y: bool = True,
    title: str | None = None,
) -> None:
    """eig_hist: (T, n) ascending-sorted eigenvalues per measurement step."""
    eig = np.asarray(eig_hist, dtype=float)
    if eig.size == 0:
        ax.text(0.5, 0.5, "empty Gramian history", ha="center", va="center",
                transform=ax.transAxes, color=PALETTE["dim"])
        style_axis(ax)
        return

    n = eig.shape[1]
    t = np.arange(eig.shape[0]) if t_axis is None else np.asarray(t_axis, dtype=float)

    cmap = ["#22D3EE", "#10B981", "#F59E0B", "#FB923C", "#F43F5E", "#8B5CF6"]
    for j in range(n):
        # +1e-30 keeps the log-scale plot drawable when an eigenvalue is 0
        # at the very first step (pre-Gramian-accumulation).
        ax.plot(t, eig[:, j] + 1e-30, lw=1.4, color=cmap[j % len(cmap)],
                label=f"λ_{j+1}")

    if log_y:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("eigenvalue of W_obs")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    style_axis(ax)


def plot_observability_summary(
    *,
    ax_eig,
    ax_dirs,
    eig_hist: Array,
    eigvals_final: Array,
    eigvecs_final: Array,
    state_labels: list[str] | None = None,
) -> None:
    plot_gramian_eigvals(eig_hist, ax=ax_eig, title="Gramian eigenvalues vs step")

    if state_labels is None:
        state_labels = ["x", "y", "z", "vx", "vy", "vz"]

    eigvals = np.asarray(eigvals_final, dtype=float)
    V = np.asarray(eigvecs_final, dtype=float)
    n = V.shape[0]
    weights = np.abs(V)

    im = ax_dirs.imshow(weights, aspect="auto", cmap="magma_r", origin="lower")
    ax_dirs.set_xticks(range(n))
    ax_dirs.set_xticklabels(
        [f"λ={eigvals[i]:.1e}" for i in range(n)], rotation=30, ha="right",
    )
    ax_dirs.set_yticks(range(n))
    ax_dirs.set_yticklabels(state_labels)
    ax_dirs.set_title("Eigenvector magnitudes (cols=modes)")
    cbar = ax_dirs.figure.colorbar(im, ax=ax_dirs, fraction=0.04, pad=0.02)
    cbar.set_label("|component|", fontsize=8)
