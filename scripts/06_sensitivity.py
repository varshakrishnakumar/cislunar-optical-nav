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
    plot_xy as _plot_xy,
    write_dict_rows_csv as _write_csv,
)
from _common import repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic sensitivity sweeps for the EKF midcourse-correction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--plots-dir", default="results/plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _apply_dark_theme()
    run_case = _load_run_case()

    plots_dir = repo_path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    mu       = 0.0121505856
    t0       = 0.0
    tf       = 6.0
    dt_meas  = 0.02
    seed     = 7

    dx0     = np.array([2e-4, -1e-4, 0.0, 0.0,  2e-3,   0.0], dtype=float)
    est_err = np.array([3e-4,  2e-4, 0.0, 0.0, -1.5e-3, 0.0], dtype=float)

    rows: list[dict] = []

    tc_fixed     = 2.0
    dropout_prob = 0.0
    sigma_grid   = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    for sigma_px in sigma_grid:
        out = run_case(
            mu, t0, tf, tc_fixed, dt_meas,
            float(sigma_px), float(dropout_prob), int(seed),
            dx0, est_err,
            camera_mode="estimate_tracking",
        )
        rows.append({
            "sweep": "sigma", "bonus": 0,
            "sigma_px":              float(sigma_px),
            "tc":                    float(out["tc"]),
            "dropout_prob":          float(dropout_prob),
            "camera_mode":           out["camera_mode"],
            "dv_perfect":            float(out["dv_perfect_mag"]),
            "dv_ekf":                float(out["dv_ekf_mag"]),
            "dv_delta":              float(out["dv_delta_mag"]),
            "dv_inflation":          float(out["dv_inflation"]),
            "dv_inflation_pct":      float(out["dv_inflation_pct"]),
            "miss_unc":              float(out["miss_uncorrected"]),
            "miss_perf":             float(out["miss_perfect"]),
            "miss_ekf":              float(out["miss_ekf"]),
            "pos_err_tc":            float(out["pos_err_tc"]),
            "tracePpos_tc":          float(out["tracePpos_tc"]),
            "nis_mean":              float(out["nis_mean"]),
            "valid_rate":            float(out["valid_rate"]),
        })

    _sel = lambda s, b: [r for r in rows if r["sweep"] == s and r["bonus"] == b]

    s0 = _sel("sigma", 0)
    sigma_arr = np.array([r["sigma_px"]    for r in s0], dtype=float)
    dv_infl   = np.array([r["dv_inflation"] for r in s0], dtype=float)
    miss_ekf  = np.array([r["miss_ekf"]    for r in s0], dtype=float)
    poserr_tc = np.array([r["pos_err_tc"]  for r in s0], dtype=float)

    _plot_xy(sigma_arr, dv_infl,
             xlabel="σ_px  [px]",
             ylabel="Delta-V inflation  [dimensionless CR3BP velocity]",
             title="ΔV Inflation vs Pixel Noise  (tc fixed)",
             outpath=plots_dir / "06b_dv_inflation_vs_sigma.png",
             color=_CYAN, marker_color=_AMBER)
    _plot_xy(sigma_arr, miss_ekf,
             xlabel="σ_px  [px]",
             ylabel="Terminal miss  [dimensionless CR3BP length]",
             title="Terminal Miss vs Pixel Noise  (tc fixed)",
             outpath=plots_dir / "06b_miss_vs_sigma.png",
             color=_VIOLET, marker_color=_AMBER)
    _plot_xy(sigma_arr, poserr_tc,
             xlabel="σ_px  [px]",
             ylabel="‖r̂(tc) − r(tc)‖  [dimensionless CR3BP length]",
             title="Position Error at tc vs Pixel Noise  (tc fixed)",
             outpath=plots_dir / "06b_poserr_tc_vs_sigma.png",
             color=_GREEN, marker_color=_AMBER)

    sigma_fixed = 1.5
    tc_grid     = [0.8, 1.2, 1.6, 2.0, 2.5, 3.0]

    for tc in tc_grid:
        out = run_case(
            mu, t0, tf, float(tc), dt_meas,
            float(sigma_fixed), float(dropout_prob), int(seed),
            dx0, est_err,
            camera_mode="estimate_tracking",
        )
        rows.append({
            "sweep": "tc", "bonus": 0,
            "sigma_px":              float(sigma_fixed),
            "tc":                    float(out["tc"]),
            "dropout_prob":          float(dropout_prob),
            "camera_mode":         out["camera_mode"],
            "dv_perfect":            float(out["dv_perfect_mag"]),
            "dv_ekf":                float(out["dv_ekf_mag"]),
            "dv_delta":              float(out["dv_delta_mag"]),
            "dv_inflation":          float(out["dv_inflation"]),
            "dv_inflation_pct":      float(out["dv_inflation_pct"]),
            "miss_unc":              float(out["miss_uncorrected"]),
            "miss_perf":             float(out["miss_perfect"]),
            "miss_ekf":              float(out["miss_ekf"]),
            "pos_err_tc":            float(out["pos_err_tc"]),
            "tracePpos_tc":          float(out["tracePpos_tc"]),
            "nis_mean":              float(out["nis_mean"]),
            "valid_rate":            float(out["valid_rate"]),
        })

    t0_sel = _sel("tc", 0)
    tc_arr       = np.array([r["tc"]           for r in t0_sel], dtype=float)
    dv_infl_tc   = np.array([r["dv_inflation"]  for r in t0_sel], dtype=float)
    miss_ekf_tc  = np.array([r["miss_ekf"]      for r in t0_sel], dtype=float)
    poserr_tc_tc = np.array([r["pos_err_tc"]    for r in t0_sel], dtype=float)

    _plot_xy(tc_arr, dv_infl_tc,
             xlabel="Correction time  tc  [dimensionless CR3BP time]",
             ylabel="Delta-V inflation  [dimensionless CR3BP velocity]",
             title="ΔV Inflation vs Correction Time  (σ_px fixed)",
             outpath=plots_dir / "06b_dv_inflation_vs_tc.png",
             color=_CYAN, marker_color=_AMBER)
    _plot_xy(tc_arr, miss_ekf_tc,
             xlabel="Correction time  tc  [dimensionless CR3BP time]",
             ylabel="Terminal miss  [dimensionless CR3BP length]",
             title="Terminal Miss vs Correction Time  (σ_px fixed)",
             outpath=plots_dir / "06b_miss_vs_tc.png",
             color=_VIOLET, marker_color=_AMBER)
    _plot_xy(tc_arr, poserr_tc_tc,
             xlabel="Correction time  tc  [dimensionless CR3BP time]",
             ylabel="‖r̂(tc) − r(tc)‖  [dimensionless CR3BP length]",
             title="Position Error at tc vs Correction Time  (σ_px fixed)",
             outpath=plots_dir / "06b_poserr_tc_vs_tc.png",
             color=_GREEN, marker_color=_AMBER)

    dropout_prob_bonus = 0.05
    tc_bonus           = 2.0

    for sigma_px in sigma_grid:
        out = run_case(
            mu, t0, tf, float(tc_bonus), dt_meas,
            float(sigma_px), float(dropout_prob_bonus), int(seed),
            dx0, est_err,
            camera_mode="fixed",
        )
        rows.append({
            "sweep": "sigma", "bonus": 1,
            "sigma_px":              float(sigma_px),
            "tc":                    float(out["tc"]),
            "dropout_prob":          float(dropout_prob_bonus),
            "camera_mode":         out["camera_mode"],
            "dv_perfect":            float(out["dv_perfect_mag"]),
            "dv_ekf":                float(out["dv_ekf_mag"]),
            "dv_delta":              float(out["dv_delta_mag"]),
            "dv_inflation":          float(out["dv_inflation"]),
            "dv_inflation_pct":      float(out["dv_inflation_pct"]),
            "miss_unc":              float(out["miss_uncorrected"]),
            "miss_perf":             float(out["miss_perfect"]),
            "miss_ekf":              float(out["miss_ekf"]),
            "pos_err_tc":            float(out["pos_err_tc"]),
            "tracePpos_tc":          float(out["tracePpos_tc"]),
            "nis_mean":              float(out["nis_mean"]),
            "valid_rate":            float(out["valid_rate"]),
        })

    s1 = _sel("sigma", 1)
    sigma_bonus   = np.array([r["sigma_px"]    for r in s1], dtype=float)
    dv_infl_bonus = np.array([r["dv_inflation"] for r in s1], dtype=float)
    miss_bonus    = np.array([r["miss_ekf"]     for r in s1], dtype=float)
    poserr_bonus  = np.array([r["pos_err_tc"]   for r in s1], dtype=float)

    _plot_xy(sigma_bonus, dv_infl_bonus,
             xlabel="σ_px  [px]",
             ylabel="Delta-V inflation  [dimensionless CR3BP velocity]",
             title="ΔV Inflation vs σ_px  (dropout=0.05, fixed pointing)",
             outpath=plots_dir / "06b_bonus_dv_inflation_vs_sigma.png",
             color=_RED, marker_color=_AMBER)
    _plot_xy(sigma_bonus, miss_bonus,
             xlabel="σ_px  [px]",
             ylabel="Terminal miss  [dimensionless CR3BP length]",
             title="Terminal Miss vs σ_px  (dropout=0.05, fixed pointing)",
             outpath=plots_dir / "06b_bonus_miss_vs_sigma.png",
             color=_RED, marker_color=_AMBER)
    _plot_xy(sigma_bonus, poserr_bonus,
             xlabel="σ_px  [px]",
             ylabel="‖r̂(tc) − r(tc)‖  [dimensionless CR3BP length]",
             title="Position Error at tc vs σ_px  (dropout=0.05, fixed pointing)",
             outpath=plots_dir / "06b_bonus_poserr_tc_vs_sigma.png",
             color=_RED, marker_color=_AMBER)

    csv_path = plots_dir / "06b_sensitivity.csv"
    _write_csv(csv_path, rows)

    print("06B sensitivity complete.")
    print(f"Wrote CSV: {csv_path}")
    print("Wrote plots:")
    for fname in [
        "06b_dv_inflation_vs_sigma.png", "06b_miss_vs_sigma.png",
        "06b_poserr_tc_vs_sigma.png",
        "06b_dv_inflation_vs_tc.png", "06b_miss_vs_tc.png",
        "06b_poserr_tc_vs_tc.png",
        "06b_bonus_dv_inflation_vs_sigma.png", "06b_bonus_miss_vs_sigma.png",
        "06b_bonus_poserr_tc_vs_sigma.png",
    ]:
        print(f"  - {plots_dir / fname}")


if __name__ == "__main__":
    main()
