"""Phase 4 — pointing bias and lag (imperfect pointing beyond noise).

Goes beyond the zero-mean attitude noise of Phase 1 to model two
deterministic pointing pathologies:

  1. *Bias*  : a constant residual misalignment of the camera DCM
                (e.g. uncalibrated mounting offset). Sweep magnitude.
  2. *Lag*   : the camera tracks an estimate from N steps ago instead
                of the current one. Sweep N (≈ control-loop latency).

Both surface as un-modeled bearing error in the filter — the goal is
to characterize the regime where pointing degrades performance vs the
regime where pointing is "good enough".

Usage
-----
python scripts/06o_pointing_bias_lag.py --n-seeds 30
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
    load_midcourse_run_case,
    AMBER, BG, BORDER, CYAN, GREEN, PANEL, RED, TEXT, VIOLET,
)
from _common import ensure_src_on_path, repo_path

ensure_src_on_path()
from utils.units import RunUnits  # noqa: E402

_KM_PER_LU = 384_400.0


def _run_perturbation(
    *, truth: str, kind: str, value: float, n_seeds: int,
    base_seed: int, config_kwargs: dict, n_workers: int,
) -> list[dict]:
    from _parallel_seeds import run_seeds_parallel

    kw: dict = {}
    if kind == "bias_deg":
        theta = float(value) * np.pi / 180.0
        kw["bias_att_rad"] = (0.0, theta, 0.0)
    elif kind == "lag_steps":
        kw["pointing_lag_steps"] = int(value)
    elif kind == "att_noise_deg":
        kw["sigma_att_rad"] = float(value) * np.pi / 180.0
    else:
        raise ValueError(f"Unknown perturbation kind: {kind}")

    return run_seeds_parallel(
        truth=truth, n_seeds=int(n_seeds), base_seed=int(base_seed),
        n_workers=int(n_workers),
        kwargs_extra={**config_kwargs, **kw},
        extract_fields=[
            ("miss_ekf",   "miss_ekf"),
            ("pos_err_tc", "pos_err_tc"),
            ("nis_mean",   "nis_mean"),
            ("nees_mean",  "nees_mean"),
            ("valid_rate", "valid_rate"),
        ],
        extra_row_fields={"kind": kind, "value": float(value)},
    )


def _violin_panel(
    ax: plt.Axes, xs: list[float], data: list[np.ndarray],
    *, color: str, ylabel: str, log_y: bool, baseline_xs: list[float] | None = None,
) -> None:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, lw=0.3)

    valid_data = [d for d in data if d.size]
    valid_xs   = [x for d, x in zip(data, xs) if d.size]
    if not valid_data:
        return
    parts = ax.violinplot(valid_data, positions=valid_xs, widths=
                          (max(valid_xs) - min(valid_xs)) * 0.05
                          if len(valid_xs) > 1 else 0.4,
                          showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_alpha(0.55)
        body.set_edgecolor(color)
    if "cmedians" in parts:
        parts["cmedians"].set_color(AMBER)
        parts["cmedians"].set_lw(1.6)

    medians = np.array([np.nanmedian(d) for d in valid_data])
    ax.plot(valid_xs, medians, "o-", color=color, lw=1.6, ms=6,
            alpha=0.9, label="median")

    if log_y:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel, color=TEXT)


def _plot_sweep(
    bias_rows_by_value: dict[float, list[dict]],
    lag_rows_by_value:  dict[float, list[dict]],
    units: RunUnits,
    *, truth: str, n_seeds: int, outpath: Path,
) -> None:
    apply_dark_theme()

    bias_xs = sorted(bias_rows_by_value.keys())
    lag_xs  = sorted(lag_rows_by_value.keys())
    bias_miss = [
        np.array([r["miss_ekf"] for r in bias_rows_by_value[v]
                  if np.isfinite(r["miss_ekf"])], dtype=float)
        for v in bias_xs
    ]
    lag_miss = [
        np.array([r["miss_ekf"] for r in lag_rows_by_value[v]
                  if np.isfinite(r["miss_ekf"])], dtype=float)
        for v in lag_xs
    ]
    if units.truth == "cr3bp":
        bias_miss = [m * _KM_PER_LU for m in bias_miss]
        lag_miss  = [m * _KM_PER_LU for m in lag_miss]

    bias_nis = [
        np.array([r["nis_mean"] for r in bias_rows_by_value[v]
                  if np.isfinite(r["nis_mean"])], dtype=float)
        for v in bias_xs
    ]
    lag_nis  = [
        np.array([r["nis_mean"] for r in lag_rows_by_value[v]
                  if np.isfinite(r["nis_mean"])], dtype=float)
        for v in lag_xs
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.patch.set_facecolor(BG)

    _violin_panel(axes[0, 0], bias_xs, bias_miss, color=VIOLET,
                  ylabel="miss_ekf [km]", log_y=True)
    axes[0, 0].axhline(39.0, color=GREEN, lw=0.8, ls=":", alpha=0.7,
                        label="tight tol (39 km)")
    axes[0, 0].axhline(390.0, color=AMBER, lw=0.8, ls=":", alpha=0.7,
                        label="paper tol (390 km)")
    axes[0, 0].set_xlabel("pointing bias  [deg]", color=TEXT)
    axes[0, 0].set_title("Bias → Terminal Miss", color=TEXT, fontweight="bold")
    axes[0, 0].legend(fontsize=8)

    _violin_panel(axes[0, 1], lag_xs, lag_miss, color=CYAN,
                  ylabel="miss_ekf [km]", log_y=True)
    axes[0, 1].axhline(39.0, color=GREEN, lw=0.8, ls=":", alpha=0.7)
    axes[0, 1].axhline(390.0, color=AMBER, lw=0.8, ls=":", alpha=0.7)
    axes[0, 1].set_xlabel("pointing lag  [measurement steps]", color=TEXT)
    axes[0, 1].set_title("Lag → Terminal Miss", color=TEXT, fontweight="bold")

    _violin_panel(axes[1, 0], bias_xs, bias_nis, color=VIOLET,
                  ylabel="mean NIS  (≈ 2)", log_y=False)
    axes[1, 0].axhline(2.0, color=GREEN, lw=0.8, ls="--", alpha=0.6)
    axes[1, 0].set_xlabel("pointing bias  [deg]", color=TEXT)
    axes[1, 0].set_title("Bias → NIS Consistency", color=TEXT, fontweight="bold")

    _violin_panel(axes[1, 1], lag_xs, lag_nis, color=CYAN,
                  ylabel="mean NIS  (≈ 2)", log_y=False)
    axes[1, 1].axhline(2.0, color=GREEN, lw=0.8, ls="--", alpha=0.6)
    axes[1, 1].set_xlabel("pointing lag  [measurement steps]", color=TEXT)
    axes[1, 1].set_title("Lag → NIS Consistency", color=TEXT, fontweight="bold")

    fig.suptitle(
        f"Imperfect Pointing — Bias and Lag Sweeps  ·  truth={truth}  "
        f"n_seeds={n_seeds}",
        color=TEXT, fontsize=13,
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, facecolor=BG)
    plt.close(fig)


def _write_summary(
    bias_rows_by_value: dict[float, list[dict]],
    lag_rows_by_value:  dict[float, list[dict]],
    units: RunUnits,
    out_txt: Path,
) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Pointing-bias / lag sweeps", "=" * 60]
    scale = _KM_PER_LU if units.truth == "cr3bp" else 1.0
    lines.append("\nBIAS  (deg):")
    lines.append(
        f"{'bias':>6}  {'miss_med [km]':>14}  {'miss_p95 [km]':>14}  "
        f"{'NIS_med':>8}"
    )
    for v in sorted(bias_rows_by_value.keys()):
        rows = bias_rows_by_value[v]
        m = np.array([r["miss_ekf"] for r in rows], dtype=float) * scale
        ni = np.array([r["nis_mean"] for r in rows], dtype=float)
        m = m[np.isfinite(m)]; ni = ni[np.isfinite(ni)]
        lines.append(
            f"{v:6g}  {np.median(m):14.2f}  {np.percentile(m,95):14.2f}  "
            f"{np.median(ni):8.2f}"
        )
    lines.append("\nLAG (steps):")
    lines.append(
        f"{'steps':>6}  {'miss_med [km]':>14}  {'miss_p95 [km]':>14}  "
        f"{'NIS_med':>8}"
    )
    for v in sorted(lag_rows_by_value.keys()):
        rows = lag_rows_by_value[v]
        m = np.array([r["miss_ekf"] for r in rows], dtype=float) * scale
        ni = np.array([r["nis_mean"] for r in rows], dtype=float)
        m = m[np.isfinite(m)]; ni = ni[np.isfinite(ni)]
        lines.append(
            f"{int(v):6d}  {np.median(m):14.2f}  {np.percentile(m,95):14.2f}  "
            f"{np.median(ni):8.2f}"
        )
    out_txt.write_text("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pointing-bias and lag sweeps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--bias-deg-list", type=float, nargs="+",
                   default=[0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0])
    p.add_argument("--lag-steps-list", type=int, nargs="+",
                   default=[0, 1, 2, 5, 10, 20])
    p.add_argument("--mu",  type=float, default=0.0121505856)
    p.add_argument("--t0",  type=float, default=0.0)
    p.add_argument("--tf",  type=float, default=6.0)
    p.add_argument("--tc",  type=float, default=2.0)
    p.add_argument("--dt-meas",  type=float, default=0.02)
    p.add_argument("--sigma-px", type=float, default=1.0)
    p.add_argument("--q-acc", type=float, default=1e-14)
    p.add_argument("--out", type=str, default="results/mc/pointing_bias_lag")
    p.add_argument("--base-seed", type=int, default=7)
    p.add_argument("--n-workers", type=int, default=-1,
                   help="Process-pool size; -1 = cpu_count(); 1 = serial.")
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

    bias_rows_by_value: dict[float, list[dict]] = {}
    for bias in args.bias_deg_list:
        print(f"\n▸ BIAS sweep — bias = {bias:.3f} deg  workers={args.n_workers}")
        bias_rows_by_value[float(bias)] = _run_perturbation(
            truth=str(args.truth), kind="bias_deg", value=float(bias),
            n_seeds=int(args.n_seeds), base_seed=int(args.base_seed),
            config_kwargs=config_kwargs, n_workers=int(args.n_workers),
        )

    lag_rows_by_value: dict[float, list[dict]] = {}
    for lag in args.lag_steps_list:
        print(f"\n▸ LAG sweep — lag = {lag} steps  workers={args.n_workers}")
        lag_rows_by_value[float(lag)] = _run_perturbation(
            truth=str(args.truth), kind="lag_steps", value=float(lag),
            n_seeds=int(args.n_seeds), base_seed=int(args.base_seed),
            config_kwargs=config_kwargs, n_workers=int(args.n_workers),
        )

    out_dir = apply_truth_suffix(repo_path(args.out), args.truth)
    _plot_sweep(bias_rows_by_value, lag_rows_by_value, units,
                truth=args.truth, n_seeds=int(args.n_seeds),
                outpath=out_dir / "06o_pointing_bias_lag.png")
    _write_summary(bias_rows_by_value, lag_rows_by_value, units,
                   out_dir / "06o_pointing_bias_lag.txt")
    print(f"\nWrote:")
    print(f"  {out_dir / '06o_pointing_bias_lag.png'}")
    print(f"  {out_dir / '06o_pointing_bias_lag.txt'}")


if __name__ == "__main__":
    main()
