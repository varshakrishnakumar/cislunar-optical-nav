"""Phase 5 — Estimator ablation: EKF vs IEKF vs UKF on matched seeds.

Question this experiment answers
--------------------------------
The paper currently uses an IEKF without an empirical comparison. Reviewers
will ask whether a plain EKF would have sufficed (less compute, less code)
or whether a UKF would have done better (more nonlinearity capture,
different cost). The locked thesis is not estimator-novelty — the goal here
is to *justify the estimator choice* with matched-seed evidence rather
than to claim a new filter.

Design — the user's two cautions
--------------------------------
1. EKF/IEKF must be a TRULY controlled comparison: same gating policy
   (gating_enabled=False everywhere), same process noise, same measurement
   generation, same RNG seeds, same pointing policy. The only difference
   is bearing_update_tangent's max_iterations (1 for EKF, 3 for IEKF).
2. UKF must be treated as an IMPLEMENTATION comparison, not automatically
   "more nonlinear = better." Sigma-point scaling (alpha=1e-3, beta=2,
   kappa=0), covariance repair (eigvalh + non-negative clamp + Cholesky
   regularization), and unit-vector residual handling (tangent-plane at
   weighted-mean LOS) all differ from the EKF/IEKF paths. These are
   documented in run_case's helpers and noted in the report subsection.

Experiment matrix
-----------------
filter_kind ∈ { ekf, iekf, ukf }  on identical truth/measurement/seeds
Baseline scenario: Moon-only, estimate-driven active pointing, σ_px = 1 px,
q_acc = 1e-14, t_c = 2.0, n_seeds matched across cells.

Metrics reported per cell
-------------------------
Mission-output (existing):
  miss_ekf median + p95
  pass rate at 1e-3 LU (~390 km) and 1e-4 LU (~39 km)
  pos_err_tc median
Estimator consistency:
  NIS median (chi^2(2) reference 2.0)
  NEES median + 95% band fraction (chi^2(6) reference 6.0; band [1.24,12.83])
  divergence count (NaN/inf miss)
Computational:
  t_trial_total median [s]
  t_predict_mean median [us]
  t_update_mean median [us]
  iters_used_mean median (1 for EKF, 3 for IEKF, 1 for UKF)

Usage
-----
Smoke:        python scripts/06s_estimator_ablation.py --n-seeds 4
Tier-2:       python scripts/06s_estimator_ablation.py --n-seeds 100 --n-workers -1 --out results/mc/phase_g_estimator_ablation_tier2
Production:   python scripts/06s_estimator_ablation.py --n-seeds 1000 --n-workers -1 --out results/mc/phase_g_estimator_ablation
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

from _analysis_common import (  # noqa: E402
    add_truth_arg,
    apply_dark_theme,
    apply_truth_suffix,
    AMBER, BG, BORDER, CYAN, GREEN, PANEL, RED, TEXT, VIOLET,
)
from _common import ensure_src_on_path, repo_path

ensure_src_on_path()
from utils.units import RunUnits  # noqa: E402

_KM_PER_LU = 384_400.0
_FILTER_KINDS = ("ekf", "iekf", "ukf")

# Mission-relevant terminal-miss thresholds (LU, converted to km in the
# aggregator). 1e-3 LU = ~390 km is the report's screening tolerance;
# 1e-4 LU = ~39 km is the tight reality-check tolerance from the
# existing terminal-tolerance reality-check section.
_THRESHOLDS_LU  = (1e-3, 1e-4)
_THRESHOLDS_KM  = tuple(t * _KM_PER_LU for t in _THRESHOLDS_LU)

# chi^2(6) 95% band — used for NEES consistency band-fraction reporting.
_NEES_BAND_LO   = float(chi2.ppf(0.025, df=6))
_NEES_BAND_HI   = float(chi2.ppf(0.975, df=6))


_EXTRACT_FIELDS = [
    ("miss_ekf",            "miss_ekf"),
    ("pos_err_tc",          "pos_err_tc"),
    ("valid_rate",          "valid_rate"),
    ("nis_mean",            "nis_mean"),
    ("nees_mean",           "nees_mean"),
    ("t_trial_total_s",     "t_trial_total_s"),
    ("t_predict_mean_us",   "t_predict_mean_us"),
    ("t_update_mean_us",    "t_update_mean_us"),
    ("iters_used_mean",     "iters_used_mean"),
    ("n_predict_calls",     "n_predict_calls"),
    ("n_update_calls",      "n_update_calls"),
]


def _run_filter(
    *, truth: str, filter_kind: str, n_seeds: int, base_seed: int,
    common_kwargs: dict, n_workers: int,
) -> list[dict]:
    from _parallel_seeds import run_seeds_parallel
    return run_seeds_parallel(
        truth=truth, n_seeds=int(n_seeds), base_seed=int(base_seed),
        n_workers=int(n_workers),
        kwargs_extra={**common_kwargs, "filter_kind": filter_kind},
        extract_fields=_EXTRACT_FIELDS,
        extra_row_fields={"filter_kind": filter_kind},
    )


def _aggregate(rows: list[dict], *, truth: str) -> dict:
    if not rows:
        return {"n": 0}
    miss = np.array([r["miss_ekf"] for r in rows], dtype=float)
    pos  = np.array([r["pos_err_tc"] for r in rows], dtype=float)
    nis  = np.array([r["nis_mean"] for r in rows], dtype=float)
    nees = np.array([r["nees_mean"] for r in rows], dtype=float)
    t_trial  = np.array([r["t_trial_total_s"] for r in rows], dtype=float)
    t_pred   = np.array([r["t_predict_mean_us"] for r in rows], dtype=float)
    t_upd    = np.array([r["t_update_mean_us"] for r in rows], dtype=float)
    iters    = np.array([r["iters_used_mean"] for r in rows], dtype=float)

    if truth == "cr3bp":
        miss_km = miss * _KM_PER_LU
        pos_km  = pos * _KM_PER_LU
    else:
        miss_km = miss
        pos_km  = pos

    finite_miss = miss_km[np.isfinite(miss_km)]
    n_div = int((~np.isfinite(miss_km)).sum())

    pass_rate = {}
    for thr_km in _THRESHOLDS_KM:
        if finite_miss.size:
            pass_rate[float(thr_km)] = float(
                (finite_miss < thr_km).mean()
            )
        else:
            pass_rate[float(thr_km)] = float("nan")

    nees_band_frac = float(
        ((nees >= _NEES_BAND_LO) & (nees <= _NEES_BAND_HI)).mean()
    ) if nees.size and np.any(np.isfinite(nees)) else float("nan")

    return {
        "n":               len(rows),
        "miss_med":        float(np.median(finite_miss)) if finite_miss.size else float("nan"),
        "miss_p95":        float(np.percentile(finite_miss, 95)) if finite_miss.size else float("nan"),
        "pos_med":         float(np.median(pos_km[np.isfinite(pos_km)])) if pos_km.size else float("nan"),
        "pass_rate_390km": pass_rate[_THRESHOLDS_KM[0]],
        "pass_rate_39km":  pass_rate[_THRESHOLDS_KM[1]],
        "nis_med":         float(np.nanmedian(nis)),
        "nees_med":        float(np.nanmedian(nees)),
        "nees_band_frac":  nees_band_frac,
        "n_diverged":      n_div,
        "t_trial_med_s":   float(np.nanmedian(t_trial)),
        "t_predict_med_us":float(np.nanmedian(t_pred)),
        "t_update_med_us": float(np.nanmedian(t_upd)),
        "iters_med":       float(np.nanmedian(iters)),
    }


def _write_summary(
    rows_by_filter: dict[str, list[dict]],
    *, truth: str, out_txt: Path,
) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Estimator Ablation (06s)",
        "=" * 130,
        f"truth = {truth}, NEES band = chi^2(6) 95% [{_NEES_BAND_LO:.2f}, {_NEES_BAND_HI:.2f}]",
        "",
        f"{'filter':>6}  {'n':>4}  "
        f"{'miss_med':>9}  {'miss_p95':>9}  "
        f"{'pass@390':>8}  {'pass@39':>8}  "
        f"{'pos_med':>8}  {'NIS_med':>7}  "
        f"{'NEES_med':>8}  {'NEES_band':>9}  "
        f"{'div':>4}  "
        f"{'iters':>5}  {'t_pred_us':>9}  {'t_upd_us':>8}  {'t_trial_s':>9}",
    ]
    for fk in _FILTER_KINDS:
        if fk not in rows_by_filter or not rows_by_filter[fk]:
            continue
        agg = _aggregate(rows_by_filter[fk], truth=truth)
        lines.append(
            f"{fk:>6}  {agg['n']:4d}  "
            f"{agg['miss_med']:9.2f}  {agg['miss_p95']:9.2f}  "
            f"{agg['pass_rate_390km']*100:7.2f}%  {agg['pass_rate_39km']*100:7.2f}%  "
            f"{agg['pos_med']:8.2f}  {agg['nis_med']:7.3f}  "
            f"{agg['nees_med']:8.2f}  {agg['nees_band_frac']*100:8.2f}%  "
            f"{agg['n_diverged']:4d}  "
            f"{agg['iters_med']:5.2f}  {agg['t_predict_med_us']:9.1f}  "
            f"{agg['t_update_med_us']:8.1f}  {agg['t_trial_med_s']:9.4f}"
        )
    out_txt.write_text("\n".join(lines))


def _plot(
    rows_by_filter: dict[str, list[dict]],
    *, truth: str, n_seeds: int, outpath: Path,
) -> None:
    apply_dark_theme()
    cells = [fk for fk in _FILTER_KINDS if fk in rows_by_filter and rows_by_filter[fk]]
    if not cells:
        return

    miss = []
    nis = []
    nees = []
    t_upd = []
    for fk in cells:
        rows = rows_by_filter[fk]
        m = np.array([r["miss_ekf"] for r in rows], dtype=float)
        if truth == "cr3bp":
            m = m * _KM_PER_LU
        miss.append(m[np.isfinite(m)])
        nis.append(np.array([r["nis_mean"] for r in rows
                             if np.isfinite(r["nis_mean"])], dtype=float))
        nees.append(np.array([r["nees_mean"] for r in rows
                              if np.isfinite(r["nees_mean"])], dtype=float))
        t_upd.append(np.array([r["t_update_mean_us"] for r in rows
                               if np.isfinite(r["t_update_mean_us"])], dtype=float))

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), constrained_layout=True)
    fig.patch.set_facecolor(BG)
    palette = [VIOLET, AMBER, CYAN]

    panels = [
        (axes[0], miss,  "Terminal Miss [km]",     True),
        (axes[1], nis,   "NIS (chi^2(2) ref=2)",   True),
        (axes[2], nees,  "NEES (chi^2(6) ref=6)",  True),
        (axes[3], t_upd, "Mean Update Time [us]",  True),
    ]
    for ax, data, title, log_y in panels:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.grid(True, color=BORDER, lw=0.3)
        if not all(d.size for d in data):
            ax.set_title(title + " (incomplete)", color=TEXT)
            continue
        bp = ax.boxplot(data, positions=range(len(cells)), widths=0.55,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color=AMBER, lw=1.8))
        for patch, c in zip(bp["boxes"], palette):
            patch.set_facecolor(c)
            patch.set_alpha(0.45)
            patch.set_edgecolor(c)
        if log_y:
            ax.set_yscale("log")
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels(cells, fontsize=10)
        ax.set_title(title, color=TEXT, fontsize=11)

    # NEES reference lines (chi^2(6) 95% band) on panel 2.
    axes[2].axhline(_NEES_BAND_LO, color=GREEN, lw=1.0, ls="--", alpha=0.6)
    axes[2].axhline(_NEES_BAND_HI, color=GREEN, lw=1.0, ls="--", alpha=0.6)
    axes[2].axhline(6.0, color=RED, lw=1.0, ls=":", alpha=0.6, label="ref=6")

    fig.suptitle(
        f"Estimator Ablation  ·  truth={truth}  n_seeds={n_seeds}  "
        f"(matched seeds, Moon-only, active-pointing baseline)",
        color=TEXT, fontsize=12,
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, facecolor=BG)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimator Ablation (06s) — EKF vs IEKF vs UKF, matched seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-seeds",   type=int, default=8)
    p.add_argument("--mu",        type=float, default=0.0121505856)
    p.add_argument("--t0",        type=float, default=0.0)
    p.add_argument("--tf",        type=float, default=6.0)
    p.add_argument("--tc",        type=float, default=2.0)
    p.add_argument("--dt-meas",   type=float, default=0.02)
    p.add_argument("--sigma-px",  type=float, default=1.0)
    p.add_argument("--q-acc",     type=float, default=1e-14)
    p.add_argument("--out",       type=str, default="results/mc/phase_g_estimator_ablation")
    p.add_argument("--base-seed", type=int, default=7)
    p.add_argument("--n-workers", type=int, default=-1,
                   help="Process-pool size; -1 = cpu_count(); 1 = serial.")
    add_truth_arg(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    apply_dark_theme()

    common_kwargs = dict(
        mu=args.mu, t0=args.t0, tf=args.tf, tc=args.tc,
        dt_meas=args.dt_meas, sigma_px=args.sigma_px,
        dropout_prob=0.0, q_acc=args.q_acc,
        camera_mode="estimate_tracking",
    )

    rows_by_filter: dict[str, list[dict]] = {}
    print(f"\n▸ 06s Estimator Ablation")
    print(f"  truth         = {args.truth}")
    print(f"  n_seeds       = {args.n_seeds}  (matched across filters)")
    print(f"  n_workers     = {args.n_workers}")
    print(f"  filters       = {' / '.join(_FILTER_KINDS)}")
    print(f"  scenario      = Moon-only, estimate-driven active pointing\n")

    for i, fk in enumerate(_FILTER_KINDS, 1):
        print(f"  [{i}/{len(_FILTER_KINDS)}] filter_kind = {fk}")
        rows_by_filter[fk] = _run_filter(
            truth=str(args.truth), filter_kind=fk,
            n_seeds=int(args.n_seeds), base_seed=int(args.base_seed),
            common_kwargs=common_kwargs,
            n_workers=int(args.n_workers),
        )

    out_dir = apply_truth_suffix(repo_path(args.out), args.truth)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "06s_estimator_ablation.txt"
    plot_path    = out_dir / "06s_estimator_ablation.png"
    csv_path     = out_dir / "06s_estimator_ablation.csv"

    _write_summary(rows_by_filter, truth=str(args.truth), out_txt=summary_path)
    _plot(rows_by_filter, truth=str(args.truth),
          n_seeds=int(args.n_seeds), outpath=plot_path)

    fieldnames = ["filter_kind", "trial_id", "seed"] + [
        out_key for out_key, _ in _EXTRACT_FIELDS
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for fk, rows in rows_by_filter.items():
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\n  Wrote:")
    print(f"    {summary_path}")
    print(f"    {plot_path}")
    print(f"    {csv_path}")


if __name__ == "__main__":
    main()
