"""Multi-seed sensitivity sweeps for the EKF midcourse-correction pipeline.

Sweeps pixel-noise σ and correction-time tc and reports median + p05/p95
bands across a small Monte Carlo ensemble per grid point. Robust to the
single-seed outliers that contaminated the deterministic single-seed version.
"""
from __future__ import annotations

import argparse

import numpy as np

from _analysis_common import (
    AMBER as _AMBER,
    CYAN as _CYAN,
    GREEN as _GREEN,
    RED as _RED,
    VIOLET as _VIOLET,
    apply_dark_theme as _apply_dark_theme,
    load_midcourse_run_case as _load_run_case,
    plot_xy_band as _plot_band,
    write_dict_rows_csv as _write_csv,
)
from _common import repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-seed sensitivity sweeps over σ_px and tc.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--plots-dir", default="results/plots")
    parser.add_argument("--n-seeds",   type=int, default=8,
                        help="Seeds per grid point.")
    parser.add_argument("--q-acc",     type=float, default=1e-14,
                        help="EKF process-noise density (ND CR3BP units).")
    return parser.parse_args()


def _sweep_point(
    run_case,
    *,
    mu: float, t0: float, tf: float, tc: float, dt_meas: float,
    sigma_px: float, dropout_prob: float, camera_mode: str,
    dx0: np.ndarray, est_err: np.ndarray,
    n_seeds: int, q_acc: float,
) -> dict:
    """Run n_seeds trials at one (σ, tc, dropout, camera) point.

    Returns median and p05/p95 of the key metrics so the outer sweep can
    plot a robust band instead of a single-seed line.
    """
    metrics = {k: [] for k in ("dv_mag_bias", "dv_delta", "miss_ekf", "pos_err_tc")}
    n_failed = 0
    for s in range(n_seeds):
        try:
            out = run_case(
                mu, t0, tf, tc, dt_meas,
                float(sigma_px), float(dropout_prob), int(s),
                dx0, est_err,
                camera_mode=camera_mode,
                q_acc=float(q_acc),
                return_debug=False,
                accumulate_gramian=False,
            )
        except Exception:
            # A diverged targeting solve counts as a catastrophic failure for
            # this seed — keep it in the distribution as NaN so the band width
            # still reflects the failure rate.
            n_failed += 1
            for v in metrics.values():
                v.append(float("nan"))
            continue
        metrics["dv_mag_bias"].append(out["dv_mag_bias"])
        metrics["dv_delta"].append(out["dv_delta_mag"])
        metrics["miss_ekf"].append(out["miss_ekf"])
        metrics["pos_err_tc"].append(out["pos_err_tc"])

    row = {"sigma_px": float(sigma_px), "tc": float(tc),
           "dropout_prob": float(dropout_prob), "camera_mode": camera_mode,
           "n_seeds": int(n_seeds), "n_failed": int(n_failed)}
    for k, vals in metrics.items():
        arr = np.array(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            row[f"{k}_median"] = float("nan")
            row[f"{k}_p05"]    = float("nan")
            row[f"{k}_p95"]    = float("nan")
        else:
            row[f"{k}_median"] = float(np.median(finite))
            row[f"{k}_p05"]    = float(np.percentile(finite,  5))
            row[f"{k}_p95"]    = float(np.percentile(finite, 95))
    return row


def _plot_sweep(rows, x_key, y_base, *, xlabel, ylabel, title, outpath, color):
    x   = np.array([r[x_key]             for r in rows], dtype=float)
    med = np.array([r[f"{y_base}_median"] for r in rows], dtype=float)
    lo  = np.array([r[f"{y_base}_p05"]    for r in rows], dtype=float)
    hi  = np.array([r[f"{y_base}_p95"]    for r in rows], dtype=float)
    _plot_band(x, med, lo, hi,
               xlabel=xlabel, ylabel=ylabel, title=title,
               outpath=outpath, color=color, marker_color=_AMBER)


def main() -> None:
    args = parse_args()
    _apply_dark_theme()
    run_case = _load_run_case()

    plots_dir = repo_path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    mu       = 0.0121505856
    t0, tf   = 0.0, 6.0
    dt_meas  = 0.02

    dx0     = np.array([2e-4, -1e-4, 0.0, 0.0,  2e-3,   0.0], dtype=float)
    est_err = np.array([3e-4,  2e-4, 0.0, 0.0, -1.5e-3, 0.0], dtype=float)

    sigma_grid = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    tc_fixed   = 2.0
    sigma_rows = [
        _sweep_point(run_case, mu=mu, t0=t0, tf=tf, tc=tc_fixed, dt_meas=dt_meas,
                     sigma_px=s, dropout_prob=0.0, camera_mode="estimate_tracking",
                     dx0=dx0, est_err=est_err, n_seeds=args.n_seeds, q_acc=args.q_acc)
        for s in sigma_grid
    ]

    tc_grid     = [0.8, 1.2, 1.6, 2.0, 2.5, 3.0]
    sigma_fixed = 1.5
    tc_rows = [
        _sweep_point(run_case, mu=mu, t0=t0, tf=tf, tc=tc, dt_meas=dt_meas,
                     sigma_px=sigma_fixed, dropout_prob=0.0, camera_mode="estimate_tracking",
                     dx0=dx0, est_err=est_err, n_seeds=args.n_seeds, q_acc=args.q_acc)
        for tc in tc_grid
    ]

    # Bonus: stressed camera (fixed pointing + dropouts).
    bonus_rows = [
        _sweep_point(run_case, mu=mu, t0=t0, tf=tf, tc=tc_fixed, dt_meas=dt_meas,
                     sigma_px=s, dropout_prob=0.05, camera_mode="fixed",
                     dx0=dx0, est_err=est_err, n_seeds=args.n_seeds, q_acc=args.q_acc)
        for s in sigma_grid
    ]

    # Tag rows so they round-trip cleanly through a flat CSV.
    for r in sigma_rows: r["sweep"] = "sigma"
    for r in tc_rows:    r["sweep"] = "tc"
    for r in bonus_rows: r["sweep"] = "sigma_bonus"
    all_rows = sigma_rows + tc_rows + bonus_rows
    _write_csv(plots_dir / "06b_sensitivity.csv", all_rows)

    # ── σ sweep plots (tracking) ────────────────────────────────────────────
    _plot_sweep(sigma_rows, "sigma_px", "dv_mag_bias",
        xlabel="σ_px  [px]",
        ylabel="|Δv_EKF| − |Δv_perfect|  [dimensionless CR3BP velocity]",
        title=f"Burn-Magnitude Bias vs Pixel Noise  (tc={tc_fixed}, n_seeds={args.n_seeds})",
        outpath=plots_dir / "06b_dv_mag_bias_vs_sigma.png", color=_CYAN)
    _plot_sweep(sigma_rows, "sigma_px", "dv_delta",
        xlabel="σ_px  [px]",
        ylabel="‖Δv_EKF − Δv_perfect‖  [dimensionless CR3BP velocity]",
        title=f"Absolute Burn Error vs Pixel Noise  (tc={tc_fixed}, n_seeds={args.n_seeds})",
        outpath=plots_dir / "06b_dv_delta_vs_sigma.png", color=_CYAN)
    _plot_sweep(sigma_rows, "sigma_px", "miss_ekf",
        xlabel="σ_px  [px]",
        ylabel="Terminal miss  [dimensionless CR3BP length]",
        title=f"Terminal Miss vs Pixel Noise  (tc={tc_fixed}, n_seeds={args.n_seeds})",
        outpath=plots_dir / "06b_miss_vs_sigma.png", color=_VIOLET)
    _plot_sweep(sigma_rows, "sigma_px", "pos_err_tc",
        xlabel="σ_px  [px]",
        ylabel="‖r̂(tc) − r(tc)‖  [dimensionless CR3BP length]",
        title=f"Position Error at tc vs Pixel Noise  (tc={tc_fixed}, n_seeds={args.n_seeds})",
        outpath=plots_dir / "06b_poserr_tc_vs_sigma.png", color=_GREEN)

    # ── tc sweep plots ─────────────────────────────────────────────────────
    _plot_sweep(tc_rows, "tc", "dv_mag_bias",
        xlabel="Correction time tc  [dimensionless CR3BP time]",
        ylabel="|Δv_EKF| − |Δv_perfect|  [dimensionless CR3BP velocity]",
        title=f"Burn-Magnitude Bias vs Correction Time  (σ_px={sigma_fixed}, n_seeds={args.n_seeds})",
        outpath=plots_dir / "06b_dv_mag_bias_vs_tc.png", color=_CYAN)
    _plot_sweep(tc_rows, "tc", "dv_delta",
        xlabel="Correction time tc  [dimensionless CR3BP time]",
        ylabel="‖Δv_EKF − Δv_perfect‖  [dimensionless CR3BP velocity]",
        title=f"Absolute Burn Error vs Correction Time  (σ_px={sigma_fixed}, n_seeds={args.n_seeds})",
        outpath=plots_dir / "06b_dv_delta_vs_tc.png", color=_CYAN)
    _plot_sweep(tc_rows, "tc", "miss_ekf",
        xlabel="Correction time tc  [dimensionless CR3BP time]",
        ylabel="Terminal miss  [dimensionless CR3BP length]",
        title=f"Terminal Miss vs Correction Time  (σ_px={sigma_fixed}, n_seeds={args.n_seeds})",
        outpath=plots_dir / "06b_miss_vs_tc.png", color=_VIOLET)
    _plot_sweep(tc_rows, "tc", "pos_err_tc",
        xlabel="Correction time tc  [dimensionless CR3BP time]",
        ylabel="‖r̂(tc) − r(tc)‖  [dimensionless CR3BP length]",
        title=f"Position Error at tc vs Correction Time  (σ_px={sigma_fixed}, n_seeds={args.n_seeds})",
        outpath=plots_dir / "06b_poserr_tc_vs_tc.png", color=_GREEN)

    # ── Bonus σ sweep plots (stressed) ──────────────────────────────────────
    _plot_sweep(bonus_rows, "sigma_px", "dv_mag_bias",
        xlabel="σ_px  [px]",
        ylabel="|Δv_EKF| − |Δv_perfect|  [dimensionless CR3BP velocity]",
        title=f"Burn-Magnitude Bias vs σ_px  (fixed pointing + 5% dropout, n_seeds={args.n_seeds})",
        outpath=plots_dir / "06b_bonus_dv_mag_bias_vs_sigma.png", color=_RED)
    _plot_sweep(bonus_rows, "sigma_px", "miss_ekf",
        xlabel="σ_px  [px]",
        ylabel="Terminal miss  [dimensionless CR3BP length]",
        title=f"Terminal Miss vs σ_px  (fixed pointing + 5% dropout, n_seeds={args.n_seeds})",
        outpath=plots_dir / "06b_bonus_miss_vs_sigma.png", color=_RED)
    _plot_sweep(bonus_rows, "sigma_px", "pos_err_tc",
        xlabel="σ_px  [px]",
        ylabel="‖r̂(tc) − r(tc)‖  [dimensionless CR3BP length]",
        title=f"Position Error at tc vs σ_px  (fixed pointing + 5% dropout, n_seeds={args.n_seeds})",
        outpath=plots_dir / "06b_bonus_poserr_tc_vs_sigma.png", color=_RED)

    print(f"06B sensitivity complete  (n_seeds={args.n_seeds}, q_acc={args.q_acc:.0e}).")
    print(f"Wrote CSV: {plots_dir / '06b_sensitivity.csv'}")


if __name__ == "__main__":
    main()
