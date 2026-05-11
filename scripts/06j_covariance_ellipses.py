"""Phase 2 — covariance ellipses at correction time.

Visualizes the 6×6 EKF covariance at the burn time tc by projecting
into 2-D position planes (x-y, x-z, y-z) and 2-D velocity planes
(vx-vy, vx-vz, vy-vz). The 1σ / 3σ ellipses reveal which directions
the filter is most/least confident about — this is the visual link
between estimation uncertainty and guidance sensitivity.

A position-error sample (the actual EKF residual at tc) is overlaid in
each panel as a sanity check that the ellipse statistically encloses
the realized error.

Usage
-----
python scripts/06j_covariance_ellipses.py            # CR3BP, default seed
python scripts/06j_covariance_ellipses.py --seed 42
python scripts/06j_covariance_ellipses.py --truth spice --n-workers 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from _analysis_common import (  # noqa: E402
    add_truth_arg,
    apply_dark_theme,
    apply_truth_suffix,
    load_midcourse_run_case,
    AMBER, BG, BORDER, CYAN, GREEN, PANEL, RED, TEXT, VIOLET,
)
from _common import ensure_src_on_path, repo_path

ensure_src_on_path()
from utils.units import RunUnits  # noqa: E402

from _paper_constants import KM_PER_LU as _KM_PER_LU
_KMPS_PER_VU = _KM_PER_LU / (4.343 * 86_400.0)


def _confidence_ellipse(
    cov2: np.ndarray,
    mean: np.ndarray,
    n_sigma: float,
    color: str,
    ls: str = "-",
    lw: float = 1.5,
    alpha: float = 0.9,
    label: str | None = None,
) -> Ellipse:
    if cov2.shape != (2, 2):
        raise ValueError(f"cov2 must be 2x2, got {cov2.shape}")
    eigvals, eigvecs = np.linalg.eigh(cov2)
    # Sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    width = 2.0 * n_sigma * float(np.sqrt(max(eigvals[0], 0.0)))
    height = 2.0 * n_sigma * float(np.sqrt(max(eigvals[1], 0.0)))
    return Ellipse(
        xy=tuple(mean), width=width, height=height, angle=angle,
        edgecolor=color, facecolor="none", lw=lw, ls=ls, alpha=alpha,
        label=label,
    )


def _draw_ellipse_panel(
    ax: plt.Axes,
    cov2: np.ndarray,
    mean: np.ndarray,
    err2: np.ndarray | None,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, lw=0.3)
    ax.set_axisbelow(True)

    ax.add_patch(
        _confidence_ellipse(cov2, mean, 1.0, CYAN, ls="-", lw=1.6,
                            label="1σ")
    )
    ax.add_patch(
        _confidence_ellipse(cov2, mean, 3.0, VIOLET, ls="--", lw=1.4,
                            label="3σ")
    )
    ax.scatter([mean[0]], [mean[1]], s=24, c=AMBER, zorder=4,
               label="filter mean")
    if err2 is not None:
        ax.scatter([mean[0] + err2[0]], [mean[1] + err2[1]],
                   s=42, marker="x", c=RED, zorder=5,
                   label="truth (residual)")
    # Auto-scale around 3σ ellipse plus residual
    eig = np.linalg.eigvalsh(cov2)
    r = 3.5 * float(np.sqrt(max(eig.max(), 0.0)))
    if err2 is not None:
        r = max(r, 1.4 * float(np.linalg.norm(err2)))
    ax.set_xlim(mean[0] - r, mean[0] + r)
    ax.set_ylim(mean[1] - r, mean[1] + r)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel, color=TEXT)
    ax.set_ylabel(ylabel, color=TEXT)
    ax.set_title(title, color=TEXT, fontsize=10)


def _plot_six_panel(
    P_tc: np.ndarray,
    err6: np.ndarray,
    units: RunUnits,
    *,
    truth: str,
    seed: int,
    outpath: Path,
) -> None:
    apply_dark_theme()

    if units.truth == "cr3bp":
        # Convert ND→km / ND→km·s for human-readable axes.
        Lscale = _KM_PER_LU
        Vscale = _KMPS_PER_VU * 1000.0  # km/s → m/s
        len_lab = "km"
        vel_lab = "m/s"
        scale_pos = Lscale
        scale_vel = Vscale
        Ppos = P_tc[:3, :3] * Lscale * Lscale
        Pvel = P_tc[3:, 3:] * Vscale * Vscale
        err_pos = err6[:3] * Lscale
        err_vel = err6[3:] * Vscale
    else:
        Ppos = P_tc[:3, :3].copy()
        Pvel = P_tc[3:, 3:] * 1e6  # (km/s)² → (m/s)²
        err_pos = err6[:3].copy()
        err_vel = err6[3:] * 1000.0
        len_lab = "km"
        vel_lab = "m/s"

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fig.patch.set_facecolor(BG)

    pos_panels = [
        ((0, 1), (0, 1), "x – y"),
        ((0, 2), (0, 2), "x – z"),
        ((1, 2), (1, 2), "y – z"),
    ]
    vel_panels = [
        ((0, 1), (0, 1), "vx – vy"),
        ((0, 2), (0, 2), "vx – vz"),
        ((1, 2), (1, 2), "vy – vz"),
    ]

    for col, ((i, j), _, name) in enumerate(pos_panels):
        cov2 = Ppos[np.ix_([i, j], [i, j])]
        err2 = err_pos[[i, j]]
        _draw_ellipse_panel(
            axes[0, col], cov2, np.zeros(2), err2,
            title=f"position  {name}",
            xlabel=f"{name.split(' – ')[0]} − x̂  [{len_lab}]",
            ylabel=f"{name.split(' – ')[1]} − ŷ  [{len_lab}]",
        )

    for col, ((i, j), _, name) in enumerate(vel_panels):
        cov2 = Pvel[np.ix_([i, j], [i, j])]
        err2 = err_vel[[i, j]]
        _draw_ellipse_panel(
            axes[1, col], cov2, np.zeros(2), err2,
            title=f"velocity  {name}",
            xlabel=f"{name.split(' – ')[0]} − v̂  [{vel_lab}]",
            ylabel=f"{name.split(' – ')[1]} − v̂  [{vel_lab}]",
        )
    axes[0, 0].legend(loc="upper right", fontsize=8, framealpha=0.85)

    pos_3sigma_km = 3.0 * float(np.sqrt(max(np.trace(Ppos), 0.0)) / np.sqrt(3.0))
    fig.suptitle(
        f"Filter Covariance at Correction Time tc  ·  truth={truth}  seed={seed}\n"
        f"trace(P_pos)^½/√3 ≈ 1σ_radial = {pos_3sigma_km/3.0:.2f} {len_lab}",
        color=TEXT, fontsize=12,
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, facecolor=BG)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render 2-D covariance ellipses at burn time tc.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--mu",   type=float, default=0.0121505856)
    p.add_argument("--t0",   type=float, default=0.0)
    p.add_argument("--tf",   type=float, default=6.0)
    p.add_argument("--tc",   type=float, default=2.0)
    p.add_argument("--dt-meas",  type=float, default=0.02)
    p.add_argument("--sigma-px", type=float, default=1.0)
    p.add_argument("--sigma-att-deg", type=float, default=0.0)
    p.add_argument("--P0-scale", type=float, default=1.0)
    p.add_argument("--out", type=str, default="results/mc/covariance_ellipses")
    add_truth_arg(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_case = load_midcourse_run_case(truth=args.truth)
    units = RunUnits.for_truth(args.truth)

    dx0 = np.array([1e-4, -1e-4, 0.0, 0.0, 0.0, 0.0])
    est_err = np.array([1e-4, 1e-4, 0.0, 0.0, 0.0, 0.0])

    out = run_case(
        mu=args.mu, t0=args.t0, tf=args.tf, tc=args.tc,
        dt_meas=args.dt_meas, sigma_px=args.sigma_px, dropout_prob=0.0,
        seed=int(args.seed), dx0=dx0, est_err=est_err,
        camera_mode="estimate_tracking",
        sigma_att_rad=float(args.sigma_att_deg) * np.pi / 180.0,
        P0_scale=float(args.P0_scale),
        return_debug=True, accumulate_gramian=False,
    )
    dbg = out["debug"]
    P_tc = np.asarray(dbg["P_tc"], dtype=float)
    k_tc = int(dbg["k_tc"])
    xs_true = np.asarray(dbg["xs_true"], dtype=float)
    x_hat_hist = np.asarray(dbg["x_hat_hist"], dtype=float)
    err6 = x_hat_hist[-1] - xs_true[k_tc]

    out_dir = apply_truth_suffix(repo_path(args.out), args.truth)
    fname = f"06j_covariance_ellipses_seed{args.seed}.png"
    _plot_six_panel(
        P_tc, err6, units,
        truth=args.truth, seed=int(args.seed),
        outpath=out_dir / fname,
    )
    print(f"Wrote {out_dir / fname}")
    print(f"  miss_ekf = {out['miss_ekf']:.3e}  pos_err_tc = {out['pos_err_tc']:.3e}")
    print(f"  trace(P_pos) = {float(np.trace(P_tc[:3,:3])):.3e}  "
          f"trace(P_vel) = {float(np.trace(P_tc[3:,3:])):.3e}")


if __name__ == "__main__":
    main()
