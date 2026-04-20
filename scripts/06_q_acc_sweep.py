"""Sweep the EKF process-noise density q_acc and measure filter consistency.

Motivation: the baseline q_acc=1e-14 produces mean NEES ≈ 21 vs the χ²(6) 95%
band of [1.2, 14.4] — the filter is overconfident. This script runs run_case at
multiple q_acc values, records mean NIS/NEES plus terminal miss and ΔV bias,
and plots the trends so we can pick a better baseline.
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
    plot_xy as _plot_xy,
    write_dict_rows_csv as _write_csv,
)
from _common import repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep q_acc and record filter-consistency metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--plots-dir", default="results/plots")
    parser.add_argument("--n-seeds",   type=int, default=8,
                        help="Monte-Carlo seeds per q_acc point.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _apply_dark_theme()
    run_case = _load_run_case()

    plots_dir = repo_path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    mu       = 0.0121505856
    t0, tf   = 0.0, 6.0
    tc       = 2.0
    dt_meas  = 0.02
    sigma_px = 1.5
    dropout  = 0.0

    dx0     = np.array([2e-4, -1e-4, 0.0, 0.0,  2e-3,   0.0], dtype=float)
    est_err = np.array([3e-4,  2e-4, 0.0, 0.0, -1.5e-3, 0.0], dtype=float)

    q_grid = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6]

    rows: list[dict] = []
    for q in q_grid:
        nis_vals, nees_vals, miss_vals, bias_vals = [], [], [], []
        for s in range(args.n_seeds):
            out = run_case(
                mu, t0, tf, tc, dt_meas, sigma_px, dropout, int(s),
                dx0, est_err,
                camera_mode="estimate_tracking",
                q_acc=float(q),
                return_debug=False,
                accumulate_gramian=False,
            )
            nis_vals.append(out["nis_mean"])
            nees_vals.append(out["nees_mean"])
            miss_vals.append(out["miss_ekf"])
            bias_vals.append(out["dv_mag_bias"])

        rows.append({
            "q_acc":         float(q),
            "nis_median":    float(np.median(nis_vals)),
            "nis_p95":       float(np.percentile(nis_vals, 95)),
            "nees_median":   float(np.median(nees_vals)),
            "nees_p95":      float(np.percentile(nees_vals, 95)),
            "miss_median":   float(np.median(miss_vals)),
            "miss_p95":      float(np.percentile(miss_vals, 95)),
            "dv_bias_median":float(np.median(bias_vals)),
            "dv_bias_p95":   float(np.percentile(bias_vals, 95)),
        })
        print(f"q_acc={q:.0e}  NEES median={rows[-1]['nees_median']:.2f}  "
              f"NIS median={rows[-1]['nis_median']:.2f}  "
              f"miss median={rows[-1]['miss_median']:.3e}")

    csv_path = plots_dir / "06d_q_acc_sweep.csv"
    _write_csv(csv_path, rows)

    q_arr     = np.array([r["q_acc"] for r in rows])
    nees_med  = np.array([r["nees_median"] for r in rows])
    nis_med   = np.array([r["nis_median"]  for r in rows])
    miss_med  = np.array([r["miss_median"] for r in rows])
    bias_med  = np.array([r["dv_bias_median"] for r in rows])

    _plot_xy(q_arr, nees_med,
             xlabel="q_acc  [dimensionless CR3BP]",
             ylabel="median NEES  (expected ≈ 6)",
             title="NEES Consistency vs q_acc",
             outpath=plots_dir / "06d_nees_vs_qacc.png",
             color=_RED, marker_color=_AMBER, logx=True)
    _plot_xy(q_arr, nis_med,
             xlabel="q_acc  [dimensionless CR3BP]",
             ylabel="median NIS  (expected ≈ 2)",
             title="NIS Consistency vs q_acc",
             outpath=plots_dir / "06d_nis_vs_qacc.png",
             color=_CYAN, marker_color=_AMBER, logx=True)
    _plot_xy(q_arr, miss_med,
             xlabel="q_acc  [dimensionless CR3BP]",
             ylabel="median terminal miss  [dimensionless CR3BP length]",
             title="Terminal Miss vs q_acc",
             outpath=plots_dir / "06d_miss_vs_qacc.png",
             color=_VIOLET, marker_color=_AMBER, logx=True)
    _plot_xy(q_arr, bias_med,
             xlabel="q_acc  [dimensionless CR3BP]",
             ylabel="median ΔV magnitude bias  [dimensionless CR3BP velocity]",
             title="Burn-Magnitude Bias vs q_acc",
             outpath=plots_dir / "06d_dv_bias_vs_qacc.png",
             color=_GREEN, marker_color=_AMBER, logx=True)

    print("\n06D q_acc sweep complete.")
    print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
