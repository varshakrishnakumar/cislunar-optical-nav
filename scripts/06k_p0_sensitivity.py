"""Phase 2 — initial-covariance (P₀) sensitivity sweep.

Drives several Monte-Carlo runs across P0_scale ∈ {1.0, 3.0, 10.0,
30.0, 100.0} (the diagonal multiplier on the baseline P₀ in
`scripts/06_midcourse_ekf_correction.py`) and renders a comparison of
terminal miss, pos_err at tc, NEES and NIS distributions across scales.

The story this answers: "is the published headline an artifact of an
optimistic P₀, or does the filter still converge when started with a
much wider initial uncertainty?"

Usage
-----
python scripts/06k_p0_sensitivity.py --n-trials 100        # CR3BP
python scripts/06k_p0_sensitivity.py --n-trials 200 \
    --P0-scales 1 10 100 \
    --out results/mc/P0_sensitivity_n200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _analysis_common import (  # noqa: E402
    add_truth_arg,
    apply_dark_theme,
    apply_truth_suffix,
    inject_truth_column_into_csv,
    load_midcourse_run_case,
    AMBER, BG, BORDER, CYAN, GREEN, PANEL, RED, TEXT, VIOLET,
)
from _common import ensure_src_on_path, repo_path

ensure_src_on_path()

from utils.units import RunUnits  # noqa: E402

_DEFAULT_SCALES = [1.0, 3.0, 10.0, 30.0, 100.0]
from _paper_constants import KM_PER_LU as _KM_PER_LU


def _run_one_sweep(
    *, truth: str, n_trials: int, P0_scale: float, base_seed: int,
    config_kwargs: dict, n_workers: int,
) -> list[dict]:
    """Run an MC sweep at a fixed P0_scale; return per-trial dict rows."""
    from _parallel_seeds import run_seeds_parallel
    return run_seeds_parallel(
        truth=truth, n_seeds=int(n_trials), base_seed=int(base_seed),
        n_workers=int(n_workers),
        kwargs_extra={**config_kwargs, "P0_scale": float(P0_scale)},
        extract_fields=[
            ("miss_ekf",       "miss_ekf"),
            ("pos_err_tc",     "pos_err_tc"),
            ("tracePpos_tc",   "tracePpos_tc"),
            ("nis_mean",       "nis_mean"),
            ("nees_mean",      "nees_mean"),
            ("valid_rate",     "valid_rate"),
            ("dv_delta_mag",   "dv_delta_mag"),
        ],
        extra_row_fields={"P0_scale": float(P0_scale)},
    )


def _box_panel(
    ax: plt.Axes, scales: list[float], data: list[np.ndarray],
    *, ylabel: str, color: str, log_y: bool = False,
    ref_line: float | None = None, ref_label: str | None = None,
) -> None:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, lw=0.3)
    bp = ax.boxplot(
        data, positions=range(len(scales)),
        widths=0.55, patch_artist=True, showfliers=True,
        medianprops=dict(color=AMBER, lw=1.6),
        whiskerprops=dict(color=color, lw=1.0),
        capprops=dict(color=color, lw=1.0),
        flierprops=dict(marker=".", color=RED, alpha=0.5, ms=3,
                        markeredgecolor=RED),
    )
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(0.45)
        box.set_edgecolor(color)
    if log_y:
        ax.set_yscale("log")
    if ref_line is not None:
        ax.axhline(ref_line, color=GREEN, lw=1.0, ls="--", alpha=0.7,
                   label=ref_label)
        if ref_label:
            ax.legend(fontsize=8)
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels([f"{s:g}×" for s in scales])
    ax.set_xlabel("P₀ scale", color=TEXT)
    ax.set_ylabel(ylabel, color=TEXT)


def _plot_summary(
    rows_by_scale: dict[float, list[dict]],
    units: RunUnits,
    *,
    truth: str,
    n_trials: int,
    outpath: Path,
) -> None:
    apply_dark_theme()
    scales = sorted(rows_by_scale.keys())

    def _col(name: str) -> list[np.ndarray]:
        return [
            np.array([r[name] for r in rows_by_scale[s]
                      if np.isfinite(r[name])], dtype=float)
            for s in scales
        ]

    miss = _col("miss_ekf")
    pos  = _col("pos_err_tc")
    nees = _col("nees_mean")
    nis  = _col("nis_mean")

    if units.truth == "cr3bp":
        miss = [m * _KM_PER_LU for m in miss]
        pos  = [p * _KM_PER_LU for p in pos]
        len_lab = "km"
    else:
        len_lab = "km"

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.patch.set_facecolor(BG)

    _box_panel(axes[0, 0], scales, miss,
               ylabel=f"miss_ekf  [{len_lab}]", color=VIOLET, log_y=True,
               ref_line=39.0, ref_label="tight tol = 39 km")
    axes[0, 0].set_title("Terminal Miss vs P₀ Scale", color=TEXT)

    _box_panel(axes[0, 1], scales, pos,
               ylabel=f"pos_err at tc  [{len_lab}]", color=CYAN, log_y=True)
    axes[0, 1].set_title("Pos-error at burn time vs P₀ Scale", color=TEXT)

    _box_panel(axes[1, 0], scales, nees,
               ylabel="mean NEES (expected ≈ 6)", color=GREEN,
               ref_line=6.0, ref_label="ideal = 6")
    axes[1, 0].set_title("NEES Consistency vs P₀ Scale", color=TEXT)

    _box_panel(axes[1, 1], scales, nis,
               ylabel="mean NIS (expected ≈ 2)", color=AMBER,
               ref_line=2.0, ref_label="ideal = 2")
    axes[1, 1].set_title("NIS Consistency vs P₀ Scale", color=TEXT)

    fig.suptitle(
        f"P₀ Sensitivity Sweep  ·  truth={truth}  n_trials={n_trials} per scale",
        color=TEXT, fontsize=13,
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, facecolor=BG)
    plt.close(fig)


def _write_summary_txt(
    rows_by_scale: dict[float, list[dict]],
    units: RunUnits,
    out_txt: Path,
) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    scales = sorted(rows_by_scale.keys())
    lines = []
    lines.append("P₀ sensitivity sweep — per-scale summary")
    lines.append("=" * 70)
    lines.append(f"{'P0_scale':>10}  {'n':>4}  "
                 f"{'miss_med [km]':>14}  {'miss_p95 [km]':>14}  "
                 f"{'NEES_med':>9}  {'NIS_med':>8}")
    for s in scales:
        rows = rows_by_scale[s]
        miss = np.array([r["miss_ekf"] for r in rows], dtype=float)
        pos  = np.array([r["pos_err_tc"] for r in rows], dtype=float)
        nees = np.array([r["nees_mean"] for r in rows], dtype=float)
        nis  = np.array([r["nis_mean"]  for r in rows], dtype=float)
        miss = miss[np.isfinite(miss)]
        if units.truth == "cr3bp":
            miss = miss * _KM_PER_LU
        nees = nees[np.isfinite(nees)]
        nis  = nis[np.isfinite(nis)]
        lines.append(
            f"{s:10g}  {len(rows):4d}  "
            f"{np.median(miss):14.2f}  {np.percentile(miss,95):14.2f}  "
            f"{np.median(nees):9.2f}  {np.median(nis):8.2f}"
        )
    out_txt.write_text("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Initial-covariance (P₀) sensitivity sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-trials", type=int, default=80)
    p.add_argument("--P0-scales", type=float, nargs="+", default=_DEFAULT_SCALES)
    p.add_argument("--mu",   type=float, default=0.0121505856)
    p.add_argument("--t0",   type=float, default=0.0)
    p.add_argument("--tf",   type=float, default=6.0)
    p.add_argument("--tc",   type=float, default=2.0)
    p.add_argument("--dt-meas",  type=float, default=0.02)
    p.add_argument("--sigma-px", type=float, default=1.0)
    p.add_argument("--q-acc",    type=float, default=1e-14)
    p.add_argument("--out", type=str, default="results/mc/P0_sensitivity")
    p.add_argument("--base-seed", type=int, default=7)
    p.add_argument("--n-workers", type=int, default=-1,
                   help="Process-pool size; -1 = cpu_count(); 1 = serial.")
    add_truth_arg(p)
    return p.parse_args()


def main() -> None:
    import csv as _csv

    args = _parse_args()
    apply_dark_theme()
    units = RunUnits.for_truth(args.truth)

    config_kwargs = dict(
        mu=args.mu, t0=args.t0, tf=args.tf, tc=args.tc,
        dt_meas=args.dt_meas, sigma_px=args.sigma_px, dropout_prob=0.0,
        camera_mode="estimate_tracking", q_acc=args.q_acc,
    )

    rows_by_scale: dict[float, list[dict]] = {}
    for scale in args.P0_scales:
        print(f"\n▸ P0_scale = {scale}  n_trials={args.n_trials}  "
              f"truth={args.truth}  workers={args.n_workers}")
        rows = _run_one_sweep(
            truth=str(args.truth), n_trials=int(args.n_trials),
            P0_scale=float(scale), base_seed=int(args.base_seed),
            config_kwargs=config_kwargs, n_workers=int(args.n_workers),
        )
        rows_by_scale[float(scale)] = rows

    out_dir = apply_truth_suffix(repo_path(args.out), args.truth)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "06k_p0_sensitivity_results.csv"
    with csv_path.open("w", newline="") as f:
        if rows_by_scale and any(rows_by_scale.values()):
            sample = next(iter(rows for rows in rows_by_scale.values() if rows))[0]
            w = _csv.DictWriter(f, fieldnames=list(sample.keys()))
            w.writeheader()
            for s in sorted(rows_by_scale):
                for r in rows_by_scale[s]:
                    w.writerow(r)
    inject_truth_column_into_csv(csv_path, args.truth)

    _plot_summary(
        rows_by_scale, units,
        truth=args.truth, n_trials=int(args.n_trials),
        outpath=out_dir / "06k_p0_sensitivity.png",
    )
    _write_summary_txt(rows_by_scale, units, out_dir / "06k_p0_sensitivity.txt")
    print(f"\nWrote:")
    print(f"  {csv_path}")
    print(f"  {out_dir / '06k_p0_sensitivity.png'}")
    print(f"  {out_dir / '06k_p0_sensitivity.txt'}")


if __name__ == "__main__":
    main()
