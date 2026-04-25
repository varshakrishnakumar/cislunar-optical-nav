from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, List

import numpy as np

from _analysis_common import (
    AMBER as _AMBER,
    CYAN as _CYAN,
    GREEN as _GREEN,
    RED as _RED,
    VIOLET as _VIOLET,
    apply_dark_theme as _apply_dark_theme,
    default_dx0_est_err as _default_dx0_est_err,
    load_midcourse_run_case as _load_run_case,
    maybe_import_sampler as _maybe_import_sampler,
    plot_xy_with_err as _plot_xy_with_err,
    safe_mean as _safe_mean,
    safe_std as _safe_std,
    sample_errors as _sample_errors,
    write_dict_rows_csv as _write_csv,
)
from _common import repo_path


def _run_point(
    run_case: Callable[..., Dict[str, Any]],
    *,
    mu: float,
    t0: float,
    tf: float,
    tc: float,
    dt_meas: float,
    sigma_px: float,
    dropout_prob: float,
    fixed_camera_pointing: bool,
    n_mc: int,
    base_seed: int,
    vary_errors: bool,
    sampler_mod,
    sigma_r_inj: float,
    sigma_v_inj: float,
    sigma_r_est: float,
    sigma_v_est: float,
    planar_only: bool,
) -> Dict[str, float]:
    dv_delta_vals: List[float] = []
    miss_vals:     List[float] = []
    poserr_vals:   List[float] = []

    for j in range(n_mc):
        seed_j = base_seed + j
        if vary_errors and sampler_mod is not None:
            dx0, est_err = _sample_errors(
                sampler_mod,
                base_seed=base_seed, trial_id=j,
                sigma_r_inj=sigma_r_inj, sigma_v_inj=sigma_v_inj,
                sigma_r_est=sigma_r_est, sigma_v_est=sigma_v_est,
                planar_only=planar_only,
            )
        else:
            dx0, est_err = _default_dx0_est_err()

        out = run_case(
            mu, t0, tf, tc, dt_meas, sigma_px, dropout_prob, seed_j,
            dx0, est_err,
            camera_mode="fixed" if fixed_camera_pointing else "estimate_tracking",
            return_debug=False,
            accumulate_gramian=False,
        )
        dv_delta_vals.append(float(out.get("dv_delta_mag", np.nan)))
        miss_vals.append(float(out.get("miss_ekf",         np.nan)))
        poserr_vals.append(float(out.get("pos_err_tc",     np.nan)))

    return {
        "dv_delta_mean":  _safe_mean(dv_delta_vals),
        "dv_delta_std":   _safe_std(dv_delta_vals),
        "miss_ekf_mean":  _safe_mean(miss_vals),
        "miss_ekf_std":   _safe_std(miss_vals),
        "pos_err_tc_mean":_safe_mean(poserr_vals),
        "pos_err_tc_std": _safe_std(poserr_vals),
    }


def main() -> None:
    _apply_dark_theme()

    p = argparse.ArgumentParser()
    p.add_argument("--plots-dir",    type=str,   default="results/mc/sensitivity_live")
    p.add_argument("--n-mc",         type=int,   default=10)
    p.add_argument("--base-seed",    type=int,   default=7)
    p.add_argument("--mu",           type=float, default=0.0121505856)
    p.add_argument("--t0",           type=float, default=0.0)
    p.add_argument("--tf",           type=float, default=6.0)
    p.add_argument("--tc",           type=float, default=2.0)
    p.add_argument("--dt-meas",      type=float, default=0.02)
    p.add_argument("--sigma-px",     type=float, default=1.5)
    p.add_argument("--dropout-prob", type=float, default=0.0)
    p.add_argument("--vary-errors",  action="store_true")
    p.add_argument("--planar-only",  action="store_true")
    p.add_argument("--sigma-r-inj",  type=float, default=1e-4)
    p.add_argument("--sigma-v-inj",  type=float, default=1e-4)
    p.add_argument("--sigma-r-est",  type=float, default=1e-4)
    p.add_argument("--sigma-v-est",  type=float, default=1e-4)

    args = p.parse_args()

    run_case = _load_run_case()
    sampler_mod = _maybe_import_sampler()

    plots_dir = repo_path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    sigma_grid = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 5.0], dtype=float)
    tc_grid    = np.array([0.8, 1.2, 1.6, 2.0, 2.5, 3.0], dtype=float)

    rows: List[Dict[str, float]] = []

    _common = dict(
        mu=args.mu, t0=args.t0, tf=args.tf, dt_meas=args.dt_meas,
        n_mc=int(args.n_mc), vary_errors=bool(args.vary_errors),
        sampler_mod=sampler_mod,
        sigma_r_inj=float(args.sigma_r_inj), sigma_v_inj=float(args.sigma_v_inj),
        sigma_r_est=float(args.sigma_r_est), sigma_v_est=float(args.sigma_v_est),
        planar_only=bool(args.planar_only),
    )

    for sigma_px in sigma_grid:
        stats = _run_point(
            run_case,
            tc=args.tc, sigma_px=float(sigma_px),
            dropout_prob=float(args.dropout_prob),
            fixed_camera_pointing=False,
            base_seed=int(args.base_seed),
            **_common,
        )
        rows.append({"sweep": 0.0, "sigma_px": float(sigma_px),
                     "tc": float(args.tc), **stats})

    for tc in tc_grid:
        stats = _run_point(
            run_case,
            tc=float(tc), sigma_px=float(args.sigma_px),
            dropout_prob=float(args.dropout_prob),
            fixed_camera_pointing=False,
            base_seed=int(args.base_seed) + 1000,
            **_common,
        )
        rows.append({"sweep": 1.0, "sigma_px": float(args.sigma_px),
                     "tc": float(tc), **stats})

    for sigma_px in sigma_grid:
        stats = _run_point(
            run_case,
            tc=args.tc, sigma_px=float(sigma_px),
            dropout_prob=0.05,
            fixed_camera_pointing=True,
            base_seed=int(args.base_seed) + 2000,
            **_common,
        )
        rows.append({"sweep": 2.0, "sigma_px": float(sigma_px),
                     "tc": float(args.tc), **stats})

    csv_path = plots_dir / "06b_sensitivity_mc.csv"
    _write_csv(csv_path, rows)

    def _sel(sweep_id: float) -> List[Dict[str, float]]:
        return [r for r in rows if float(r["sweep"]) == sweep_id]

    r0 = _sel(0.0)
    x  = np.array([r["sigma_px"] for r in r0])
    _plot_xy_with_err(
        x, np.array([r["dv_delta_mean"] for r in r0]),
              np.array([r["dv_delta_std"]  for r in r0]),
        xlabel="σ_px  [px]",
        ylabel="‖Delta-v error‖ mean ± std  [dimensionless CR3BP velocity]",
        title=f"Burn Error vs Pixel Noise  (mini-MC n={args.n_mc})",
        outpath=plots_dir / "06b_dv_error_vs_sigma.png",
        color=_CYAN, marker_color=_AMBER,
    )
    _plot_xy_with_err(
        x, np.array([r["miss_ekf_mean"]  for r in r0]),
              np.array([r["miss_ekf_std"]   for r in r0]),
        xlabel="σ_px  [px]",
        ylabel="Terminal miss mean ± std  [dimensionless CR3BP length]",
        title=f"Terminal Miss vs Pixel Noise  (mini-MC n={args.n_mc})",
        outpath=plots_dir / "06b_miss_vs_sigma.png",
        color=_VIOLET, marker_color=_AMBER,
    )
    _plot_xy_with_err(
        x, np.array([r["pos_err_tc_mean"] for r in r0]),
              np.array([r["pos_err_tc_std"]  for r in r0]),
        xlabel="σ_px  [px]",
        ylabel="‖r̂(tc)−r(tc)‖ mean ± std  [dimensionless CR3BP length]",
        title=f"Position Error at tc vs Pixel Noise  (mini-MC n={args.n_mc})",
        outpath=plots_dir / "06b_poserr_tc_vs_sigma.png",
        color=_GREEN, marker_color=_AMBER,
    )

    r1 = _sel(1.0)
    x  = np.array([r["tc"] for r in r1])
    _plot_xy_with_err(
        x, np.array([r["dv_delta_mean"] for r in r1]),
              np.array([r["dv_delta_std"]  for r in r1]),
        xlabel="Correction time  tc  [dimensionless CR3BP time]",
        ylabel="‖Delta-v error‖ mean ± std  [dimensionless CR3BP velocity]",
        title=f"Burn Error vs Correction Time  (mini-MC n={args.n_mc})",
        outpath=plots_dir / "06b_dv_error_vs_tc.png",
        color=_CYAN, marker_color=_AMBER,
    )
    _plot_xy_with_err(
        x, np.array([r["miss_ekf_mean"] for r in r1]),
              np.array([r["miss_ekf_std"]  for r in r1]),
        xlabel="Correction time  tc  [dimensionless CR3BP time]",
        ylabel="Terminal miss mean ± std  [dimensionless CR3BP length]",
        title=f"Terminal Miss vs Correction Time  (mini-MC n={args.n_mc})",
        outpath=plots_dir / "06b_miss_vs_tc.png",
        color=_VIOLET, marker_color=_AMBER,
    )
    _plot_xy_with_err(
        x, np.array([r["pos_err_tc_mean"] for r in r1]),
              np.array([r["pos_err_tc_std"]  for r in r1]),
        xlabel="Correction time  tc  [dimensionless CR3BP time]",
        ylabel="‖r̂(tc)−r(tc)‖ mean ± std  [dimensionless CR3BP length]",
        title=f"Position Error at tc vs Correction Time  (mini-MC n={args.n_mc})",
        outpath=plots_dir / "06b_poserr_tc_vs_tc.png",
        color=_GREEN, marker_color=_AMBER,
    )

    r2 = _sel(2.0)
    x  = np.array([r["sigma_px"] for r in r2])
    _plot_xy_with_err(
        x, np.array([r["dv_delta_mean"] for r in r2]),
              np.array([r["dv_delta_std"]  for r in r2]),
        xlabel="σ_px  [px]",
        ylabel="‖Delta-v error‖ mean ± std  [dimensionless CR3BP velocity]",
        title=f"Burn Error vs σ_px  (dropout=0.05, fixed pointing, n={args.n_mc})",
        outpath=plots_dir / "06b_bonus_dv_error_vs_sigma.png",
        color=_RED, marker_color=_AMBER,
    )
    _plot_xy_with_err(
        x, np.array([r["miss_ekf_mean"] for r in r2]),
              np.array([r["miss_ekf_std"]  for r in r2]),
        xlabel="σ_px  [px]",
        ylabel="Terminal miss mean ± std  [dimensionless CR3BP length]",
        title=f"Terminal Miss vs σ_px  (dropout=0.05, fixed pointing, n={args.n_mc})",
        outpath=plots_dir / "06b_bonus_miss_vs_sigma.png",
        color=_RED, marker_color=_AMBER,
    )
    _plot_xy_with_err(
        x, np.array([r["pos_err_tc_mean"] for r in r2]),
              np.array([r["pos_err_tc_std"]  for r in r2]),
        xlabel="σ_px  [px]",
        ylabel="‖r̂(tc)−r(tc)‖ mean ± std  [dimensionless CR3BP length]",
        title=f"Position Error at tc vs σ_px  (dropout=0.05, fixed pointing, n={args.n_mc})",
        outpath=plots_dir / "06b_bonus_poserr_tc_vs_sigma.png",
        color=_RED, marker_color=_AMBER,
    )

    print("\n06B mini-MC sensitivity complete.")
    print(f"Wrote CSV:   {csv_path}")
    print(f"Wrote plots: {plots_dir}")


if __name__ == "__main__":
    main()
