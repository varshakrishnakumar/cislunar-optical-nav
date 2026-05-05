"""Phase 1 — attitude (pointing) noise sweep.

Sweeps σ_attitude ∈ {0, 0.001, 0.01, 0.1, 1.0} deg and reports terminal
miss, NIS, NEES, and the in-band fraction at each level. Produces the
"performance degrades gracefully" headline that Phase 1.1's realism
upgrade is meant to demonstrate.

Usage
-----
python scripts/06q_attitude_noise_sweep.py --n-seeds 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

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

_KM_PER_LU = 384_400.0


def _run_one(
    *, truth: str, sigma_deg: float, n_seeds: int, base_seed: int,
    config_kwargs: dict, n_workers: int,
) -> list[dict]:
    from _parallel_seeds import run_seeds_parallel
    sigma_rad = float(sigma_deg) * np.pi / 180.0
    return run_seeds_parallel(
        truth=truth, n_seeds=int(n_seeds), base_seed=int(base_seed),
        n_workers=int(n_workers),
        kwargs_extra={**config_kwargs, "sigma_att_rad": sigma_rad},
        extract_fields=[
            ("miss_ekf",     "miss_ekf"),
            ("pos_err_tc",   "pos_err_tc"),
            ("nis_mean",     "nis_mean"),
            ("nis_mean_all", "nis_mean_all"),
            ("nees_mean",    "nees_mean"),
            ("valid_rate",   "valid_rate"),
        ],
        extra_row_fields={"sigma_deg": float(sigma_deg)},
    )


def _plot_summary(
    rows_by_sigma: dict[float, list[dict]],
    units: RunUnits,
    *, truth: str, n_seeds: int, outpath: Path,
) -> None:
    apply_dark_theme()
    sigmas = sorted(rows_by_sigma.keys())
    miss = [
        np.array([r["miss_ekf"] for r in rows_by_sigma[s]
                  if np.isfinite(r["miss_ekf"])], dtype=float)
        for s in sigmas
    ]
    nis  = [
        np.array([r["nis_mean"] for r in rows_by_sigma[s]
                  if np.isfinite(r["nis_mean"])], dtype=float)
        for s in sigmas
    ]
    nees = [
        np.array([r["nees_mean"] for r in rows_by_sigma[s]
                  if np.isfinite(r["nees_mean"])], dtype=float)
        for s in sigmas
    ]
    if units.truth == "cr3bp":
        miss = [m * _KM_PER_LU for m in miss]

    miss_med = np.array([np.median(m) if m.size else np.nan for m in miss])
    miss_p95 = np.array([np.percentile(m, 95) if m.size else np.nan for m in miss])
    nis_med  = np.array([np.median(n) if n.size else np.nan for n in nis])
    nees_med = np.array([np.median(n) if n.size else np.nan for n in nees])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, lw=0.3)
    # Use a small floor for log scale
    sigmas_plot = np.array([max(s, 1e-4) for s in sigmas])
    ax.fill_between(sigmas_plot, miss_med, miss_p95, color=VIOLET, alpha=0.18,
                    label="median–p95")
    ax.plot(sigmas_plot, miss_med, "o-", color=VIOLET, lw=1.8, ms=6,
            label="median miss")
    ax.plot(sigmas_plot, miss_p95, "s--", color=AMBER, lw=1.4, ms=5,
            label="p95 miss")
    ax.axhline(39.0, color=GREEN, lw=0.8, ls=":", alpha=0.7,
               label="tight tol (39 km)")
    ax.axhline(390.0, color=RED, lw=0.8, ls=":", alpha=0.7,
               label="paper tol (390 km)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("σ_attitude  [deg]", color=TEXT)
    ax.set_ylabel("miss_ekf  [km]", color=TEXT)
    ax.set_title("Graceful degradation under pointing noise",
                 color=TEXT, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, lw=0.3)
    nis_lo, nis_hi = chi2.ppf(0.025, df=2), chi2.ppf(0.975, df=2)
    ax.fill_between(sigmas_plot, nis_lo, nis_hi, color=GREEN, alpha=0.10,
                    label=f"NIS 95% χ²(2)")
    ax.plot(sigmas_plot, nis_med,  "o-", color=CYAN, lw=1.8, ms=6,
            label="median NIS")
    ax.plot(sigmas_plot, nees_med / 3.0, "s--", color=AMBER, lw=1.4, ms=5,
            label="median NEES / 3")
    ax.axhline(2.0, color=GREEN, lw=0.8, ls="--", alpha=0.5,
               label="ideal NIS = 2")
    ax.set_xscale("log")
    ax.set_xlabel("σ_attitude  [deg]", color=TEXT)
    ax.set_ylabel("filter consistency", color=TEXT)
    ax.set_title("Innovation / state consistency vs pointing noise",
                 color=TEXT, fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle(
        f"Attitude-Noise Sweep  ·  truth={truth}  n_seeds={n_seeds}",
        color=TEXT, fontsize=13,
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, facecolor=BG)
    plt.close(fig)


def _write_summary(
    rows_by_sigma: dict[float, list[dict]],
    units: RunUnits,
    out_txt: Path,
) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    scale = _KM_PER_LU if units.truth == "cr3bp" else 1.0
    nees_lo, nees_hi = chi2.ppf(0.025, df=6), chi2.ppf(0.975, df=6)

    lines = [
        "Attitude-noise sweep — per-σ summary",
        "=" * 80,
        f"{'σ_att [deg]':>11}  {'n':>4}  {'miss_med [km]':>14}  "
        f"{'miss_p95 [km]':>14}  {'NIS_med':>9}  {'NEES_inband%':>13}",
    ]
    for s in sorted(rows_by_sigma.keys()):
        rows = rows_by_sigma[s]
        miss = np.array([r["miss_ekf"] for r in rows], dtype=float) * scale
        nis  = np.array([r["nis_mean"] for r in rows], dtype=float)
        nees = np.array([r["nees_mean"] for r in rows], dtype=float)
        miss = miss[np.isfinite(miss)]
        nees = nees[np.isfinite(nees)]
        in_band = (
            float(np.mean((nees >= nees_lo) & (nees <= nees_hi)))
            if nees.size else float("nan")
        )
        lines.append(
            f"{s:11.4f}  {len(rows):4d}  {np.median(miss):14.2f}  "
            f"{np.percentile(miss, 95):14.2f}  {np.median(nis):9.2f}  "
            f"{in_band * 100:12.1f}%"
        )
    out_txt.write_text("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Attitude-noise (σ_att) sweep — graceful-degradation "
                    "headline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--sigma-att-deg-list", type=float, nargs="+",
                   default=[0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    p.add_argument("--mu",  type=float, default=0.0121505856)
    p.add_argument("--t0",  type=float, default=0.0)
    p.add_argument("--tf",  type=float, default=6.0)
    p.add_argument("--tc",  type=float, default=2.0)
    p.add_argument("--dt-meas",  type=float, default=0.02)
    p.add_argument("--sigma-px", type=float, default=1.0)
    p.add_argument("--q-acc",  type=float, default=1e-14)
    p.add_argument("--out", type=str, default="results/mc/attitude_noise_sweep")
    p.add_argument("--base-seed", type=int, default=7)
    p.add_argument("--n-workers", type=int, default=-1,
                   help="Process-pool size for parallel seed evaluation. "
                        "-1 = cpu_count(); 1 = serial (use for SPICE).")
    add_truth_arg(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    apply_dark_theme()
    units = RunUnits.for_truth(args.truth)

    config_kwargs = dict(
        mu=args.mu, t0=args.t0, tf=args.tf, tc=args.tc,
        dt_meas=args.dt_meas, sigma_px=args.sigma_px, dropout_prob=0.0,
        camera_mode="estimate_tracking", q_acc=args.q_acc,
    )

    rows_by_sigma: dict[float, list[dict]] = {}
    for s in args.sigma_att_deg_list:
        print(f"\n▸ σ_att = {s:.4f} deg  n_workers={args.n_workers}")
        rows_by_sigma[float(s)] = _run_one(
            truth=str(args.truth), sigma_deg=float(s),
            n_seeds=int(args.n_seeds), base_seed=int(args.base_seed),
            config_kwargs=config_kwargs, n_workers=int(args.n_workers),
        )

    out_dir = apply_truth_suffix(repo_path(args.out), args.truth)
    _plot_summary(rows_by_sigma, units, truth=args.truth,
                  n_seeds=int(args.n_seeds),
                  outpath=out_dir / "06q_attitude_noise_sweep.png")
    _write_summary(rows_by_sigma, units,
                   out_dir / "06q_attitude_noise_sweep.txt")
    print(f"\nWrote:")
    print(f"  {out_dir / '06q_attitude_noise_sweep.png'}")
    print(f"  {out_dir / '06q_attitude_noise_sweep.txt'}")


if __name__ == "__main__":
    main()
