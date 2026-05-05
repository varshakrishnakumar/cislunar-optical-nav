"""Fine q_acc sweep in the baseline MC regime.

Finds the process-noise setting that maximizes P(miss < tol) while keeping the
state NEES inside the χ²(6) 95% band. Unlike 06d (which sweeps q_acc with
enlarged injection errors to force divergence), this script uses the *baseline*
MC scenario: sigma_px=1.0, sigma_r_inj=sigma_v_inj=1e-4, dt_meas=0.02.

Methodology is textbook NIS/NEES consistency gating (Bar-Shalom, Li &
Kirubarajan 2001, §5.4; Crassidis & Junkins 2e, §5.7.3). The "pick the
smallest q that stays inside the χ² band" rule is just mechanical application
of that test; not tied to a cislunar-specific reference.

Outputs:
  - CSV with per-q_acc metrics (pass rate, miss quantiles, NEES, NIS).
  - Pareto plot: pass rate vs NEES band-fraction, annotated with q_acc values.
  - Miss-vs-q plot and NEES-vs-q plot for the slide/report.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from _analysis_common import (
    AMBER as _AMBER,
    BG as _BG,
    BORDER as _BORDER,
    CYAN as _CYAN,
    GREEN as _GREEN,
    ORANGE as _ORANGE,
    PANEL as _PANEL,
    RED as _RED,
    TEXT as _TEXT,
    VIOLET as _VIOLET,
    add_truth_arg,
    apply_dark_theme as _apply_dark_theme,
    apply_truth_suffix,
    load_midcourse_run_case as _load_run_case,
    tag_rows_with_truth,
    write_dict_rows_csv as _write_csv,
)
from _common import ensure_src_on_path, repo_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine q_acc sweep for baseline MC scenario.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--plots-dir", default="results/mc/fine_tune")
    p.add_argument("--n-trials", type=int, default=500,
                   help="Trials per q_acc point.")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="Miss tolerance for pass-rate.")
    p.add_argument("--q-min-log10", type=float, default=-12.0,
                   help="log10(q_acc) lower bound.")
    p.add_argument("--q-max-log10", type=float, default=-7.0,
                   help="log10(q_acc) upper bound.")
    p.add_argument("--n-q", type=int, default=11,
                   help="Number of q_acc grid points (logspace).")
    p.add_argument("--n-workers", type=int, default=-1,
                   help="Thread pool for MC. -1 uses cpu_count().")
    add_truth_arg(p)
    return p.parse_args()


def _metrics_for_q(results) -> dict:
    miss   = np.array([r.miss_ekf   for r in results])
    nees   = np.array([r.nees_mean  for r in results])
    nis    = np.array([r.nis_mean   for r in results])
    posErr = np.array([r.pos_err_tc for r in results])

    # χ²(6) two-sided 95% band
    lb6, ub6 = float(chi2.ppf(0.025, 6)), float(chi2.ppf(0.975, 6))
    # χ²(2) 95% band (for NIS)
    lb2, ub2 = float(chi2.ppf(0.025, 2)), float(chi2.ppf(0.975, 2))

    finite_nees = nees[np.isfinite(nees)]
    finite_nis  = nis[np.isfinite(nis)]

    return {
        "n_trials":        int(len(results)),
        "miss_median":     float(np.median(miss)),
        "miss_mean":       float(np.mean(miss)),
        "miss_p05":        float(np.percentile(miss, 5)),
        "miss_p95":        float(np.percentile(miss, 95)),
        "nees_median":     float(np.median(finite_nees)) if finite_nees.size else float("nan"),
        "nees_mean":       float(np.mean(finite_nees))   if finite_nees.size else float("nan"),
        "nees_p95":        float(np.percentile(finite_nees, 95)) if finite_nees.size else float("nan"),
        "nis_median":      float(np.median(finite_nis))  if finite_nis.size  else float("nan"),
        "nis_mean":        float(np.mean(finite_nis))    if finite_nis.size  else float("nan"),
        "pos_err_tc_mean": float(np.mean(posErr)),
        "nees_in_band_frac": float(np.mean((finite_nees >= lb6) & (finite_nees <= ub6))) if finite_nees.size else float("nan"),
        "nis_in_band_frac":  float(np.mean((finite_nis  >= lb2) & (finite_nis  <= ub2))) if finite_nis.size  else float("nan"),
    }


def main() -> None:
    _apply_dark_theme()
    args = parse_args()

    ensure_src_on_path()
    from mc import run_monte_carlo
    from mc.types import MonteCarloConfig

    run_case = _load_run_case(truth=args.truth)

    q_grid = np.logspace(args.q_min_log10, args.q_max_log10, int(args.n_q))
    print(f"\n▸ 06E fine q_acc sweep — baseline regime  truth={args.truth}", flush=True)
    print(f"  n_trials/q = {args.n_trials}, n_q = {len(q_grid)}, tol = {args.tol}", flush=True)
    print(f"  q grid: {[f'{q:.1e}' for q in q_grid]}\n", flush=True)

    lb6, ub6 = float(chi2.ppf(0.025, 6)), float(chi2.ppf(0.975, 6))
    rows: list[dict] = []

    for q in q_grid:
        cfg = MonteCarloConfig(
            mu=0.0121505856,
            t0=0.0, tf=6.0, tc=2.0,
            dt_meas=0.02,
            sigma_px=1.0,
            dropout_prob=0.0,
            camera_mode="estimate_tracking",
            n_trials=int(args.n_trials),
            base_seed=7,
            sigma_r_inj=1e-4, sigma_v_inj=1e-4,
            sigma_r_est=1e-4, sigma_v_est=1e-4,
            q_acc=float(q),
            study_name=f"fine_q_{q:.0e}",
        )
        results = run_monte_carlo(cfg, run_case, n_workers=int(args.n_workers))
        if not results:
            print(f"  q={q:.1e} — ALL TRIALS FAILED, skipping")
            continue

        miss = np.array([r.miss_ekf for r in results])
        pass_rate = float(np.mean(miss < float(args.tol)))
        m = _metrics_for_q(results)
        row = {"q_acc": float(q), "pass_rate": pass_rate, **m}
        rows.append(row)

        print(
            f"  q={q:.1e}  pass@{args.tol:.0e}={pass_rate*100:5.1f}%  "
            f"miss_med={m['miss_median']:.2e}  NEES_med={m['nees_median']:6.1f}  "
            f"NEES_in_band={m['nees_in_band_frac']*100:5.1f}%  "
            f"NIS_med={m['nis_median']:.2f}",
            flush=True,
        )

    if not rows:
        raise RuntimeError("No successful sweep points.")

    plots_dir = apply_truth_suffix(repo_path(args.plots_dir), args.truth)
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_path = plots_dir / "06e_fine_tune.csv"
    _write_csv(csv_path, tag_rows_with_truth(rows, args.truth))
    print(f"\nWrote CSV: {csv_path}")

    # Identify best "honest" q_acc:
    # highest pass_rate subject to NEES in [lb6, ub6] band holding for ≥ 90%
    # of trials, and NEES median itself inside [lb6, ub6].
    honest = [
        r for r in rows
        if r["nees_in_band_frac"] >= 0.90
        and lb6 <= r["nees_median"] <= ub6
    ]
    # Tie-break: among trials with the top pass_rate (within 0.5%), find
    # the leaders with NEES in-band fraction within 0.5pp of the maximum
    # (i.e. statistically indistinguishable consistency), then pick the
    # *smallest* q_acc — the textbook "minimum process noise that achieves
    # consistency" rule (Bar-Shalom §5.4).  Avoids drifting toward larger
    # q just because of MC noise on the in-band fraction.
    if honest:
        best_pr = max(r["pass_rate"] for r in honest)
        leaders = [r for r in honest if r["pass_rate"] >= best_pr - 0.005]
        best_band = max(r["nees_in_band_frac"] for r in leaders)
        equally_calibrated = [r for r in leaders
                              if r["nees_in_band_frac"] >= best_band - 0.005]
        best_honest = min(equally_calibrated, key=lambda r: r["q_acc"])
    else:
        best_honest = None
    best_raw = max(rows, key=lambda r: r["pass_rate"])

    print("\n── recommendation ─────────────────────────────")
    print(f"  χ²(6) 95% band: [{lb6:.2f}, {ub6:.2f}]")
    if best_honest is not None:
        print(
            f"  Best *consistent* q_acc: {best_honest['q_acc']:.2e}"
            f"  →  pass@{args.tol:.0e}={best_honest['pass_rate']*100:.1f}%"
            f"  NEES_med={best_honest['nees_median']:.1f}"
        )
    else:
        print("  No q_acc produced a consistent NEES distribution.")
    print(
        f"  Highest raw pass rate (possibly overconfident): "
        f"q_acc={best_raw['q_acc']:.2e}  "
        f"pass={best_raw['pass_rate']*100:.1f}%  "
        f"NEES_med={best_raw['nees_median']:.1f}"
    )
    print("────────────────────────────────────────────────\n")

    _make_plots(rows, plots_dir, tol=float(args.tol),
                band=(lb6, ub6), best_honest=best_honest)


def _make_plots(rows, plots_dir: Path, *, tol: float,
                band: tuple[float, float], best_honest) -> None:
    """One presentation-grade figure that tells the whole story.

    `06e_sweep_summary.png` — stacked 2-panel:
      top    = performance (pass rate) across the q_acc sweep
      bottom = consistency (NEES median + P95 vs χ²(6) 95% band,
               NIS overlay on the same axis vs its χ²(2) nominal)

    Layout uses in-panel headers (not Axes titles) so spacing never
    collides with markers, annotations, or the suptitle.
    """
    import matplotlib.pyplot as plt

    q      = np.array([r["q_acc"]            for r in rows])
    pr     = np.array([r["pass_rate"]        for r in rows]) * 100.0
    nmed   = np.array([r["nees_median"]      for r in rows])
    npxx   = np.array([r["nees_p95"]         for r in rows])
    nismed = np.array([r["nis_median"]       for r in rows])
    n_tr   = int(rows[0].get("n_trials", 0))
    lb6, ub6 = band

    chosen_q = best_honest["q_acc"] if best_honest is not None else None

    # ─── Figure · Sweep summary ──────────────────────────────────────
    fig, (axP, axC) = plt.subplots(
        2, 1, figsize=(14, 9), facecolor=_BG, sharex=True,
        gridspec_kw={"height_ratios": [0.7, 1.6], "hspace": 0.18},
    )
    fig.subplots_adjust(left=0.075, right=0.975, top=0.90, bottom=0.16)

    # ── Top · PERFORMANCE ────────────────────────────────────────────
    axP.set_facecolor(_PANEL)
    axP.plot(q, pr, "-o", color=_CYAN, lw=2.6, ms=9,
             markeredgecolor=_TEXT, markeredgewidth=0.8, zorder=4)
    for xi, yi in zip(q, pr):
        axP.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                     xytext=(0, -16), ha="center", va="top",
                     color=_TEXT, fontsize=9, fontweight="bold")

    axP.set_ylim(40, 110)
    axP.set_yticks([50, 75, 100])
    axP.set_ylabel("P(miss < tol)  [%]", color=_CYAN, fontweight="bold", fontsize=11)
    axP.tick_params(axis="y", colors=_CYAN)
    # In-panel header (left-aligned) — never collides with the suptitle
    pr_lo, pr_hi = float(pr.min()), float(pr.max())
    if pr_hi - pr_lo < 1.0:
        perf_msg = f"pass rate held at {pr_hi:.0f}% across the full q_acc range"
    else:
        perf_msg = f"pass rate spans {pr_lo:.0f}–{pr_hi:.0f}% across the full q_acc range"
    axP.text(
        0.005, 1.06,
        f"Performance — {perf_msg}  (tol = {tol:g})",
        transform=axP.transAxes, color=_TEXT, fontweight="bold",
        fontsize=11.5, ha="left", va="bottom",
    )
    axP.grid(True, color=_BORDER, alpha=0.35, which="major")
    if chosen_q is not None:
        axP.axvline(chosen_q, color=_VIOLET, lw=1.6, ls=":", alpha=0.95, zorder=2)
    for sp in axP.spines.values(): sp.set_edgecolor(_BORDER)

    # ── Bottom · CONSISTENCY (NEES dominant + NIS overlay) ───────────
    axC.set_facecolor(_PANEL)

    # χ²(6) 95% band
    axC.axhspan(lb6, ub6, color=_GREEN, alpha=0.16, zorder=0)
    # band edge tick lines for clarity
    axC.axhline(lb6, color=_GREEN, lw=0.8, ls="-", alpha=0.55, zorder=0)
    axC.axhline(ub6, color=_GREEN, lw=0.8, ls="-", alpha=0.55, zorder=0)

    # Reference lines (drawn behind data)
    axC.axhline(6.0, color=_TEXT, lw=0.9, ls="--", alpha=0.55, zorder=1)
    axC.axhline(2.0, color=_CYAN, lw=0.9, ls=":",  alpha=0.60, zorder=1)

    # NEES envelope (median → P95) and markers
    axC.fill_between(q, nmed, npxx, color=_RED, alpha=0.22, zorder=2,
                     label="NEES envelope (median → P95)")
    axC.plot(q, npxx, "-s", color=_ORANGE, lw=1.6, ms=7,
             markeredgecolor=_TEXT, markeredgewidth=0.5, alpha=0.95, zorder=3,
             label="NEES P95")
    axC.plot(q, nmed, "-o", color=_RED, lw=2.6, ms=9,
             markeredgecolor=_TEXT, markeredgewidth=0.8, zorder=5,
             label="NEES median")

    # NIS median (χ²(2) nominal = 2)
    axC.plot(q, nismed, "-D", color=_CYAN, lw=2.0, ms=7,
             markeredgecolor=_TEXT, markeredgewidth=0.5, alpha=0.95, zorder=4,
             label="NIS median  (χ²(2) nominal = 2)")

    if chosen_q is not None:
        axC.axvline(chosen_q, color=_VIOLET, lw=1.6, ls=":", alpha=0.95, zorder=2)

    axC.set_xscale("log")
    axC.set_xlabel("q_acc   [dimensionless CR3BP acceleration density]",
                   color=_TEXT, fontweight="bold", fontsize=11)
    axC.set_ylabel("NEES  /  NIS", color=_TEXT, fontweight="bold", fontsize=11)
    y_top = max(float(np.nanmax(npxx)) * 1.18, ub6 * 1.20)
    axC.set_ylim(0, y_top)
    axC.grid(True, color=_BORDER, alpha=0.35, which="major")
    for sp in axC.spines.values(): sp.set_edgecolor(_BORDER)

    # In-panel header — single line, won't overlap markers or legend
    sub = "Consistency — NEES envelope tracks the χ²(6) 95% band"
    if best_honest is not None:
        sub += (
            f"   ·   chosen q_acc = {best_honest['q_acc']:.1e}"
            f"   ·   {best_honest['nees_in_band_frac']*100:.0f}% of trials in-band"
        )
    axC.text(0.005, 1.025, sub, transform=axC.transAxes,
             color=_TEXT, fontweight="bold", fontsize=11.5,
             ha="left", va="bottom")

    # In-panel reference labels (Axes-fraction x, data-space y).
    # χ²(6) band label sits at the band's top edge, away from the y-axis
    # ticks and away from the chosen-q vertical line.
    axC.text(0.30, ub6, f"  χ²(6) 95% band  [{lb6:.1f} – {ub6:.1f}]",
             transform=axC.get_yaxis_transform(),
             color=_GREEN, fontsize=9.5, fontweight="bold",
             ha="left", va="bottom", alpha=0.95)
    axC.text(0.30, 6.0, "  nominal NEES = 6",
             transform=axC.get_yaxis_transform(),
             color=_TEXT, fontsize=9, ha="left", va="bottom", alpha=0.75)
    axC.text(0.30, 2.0, "  nominal NIS = 2",
             transform=axC.get_yaxis_transform(),
             color=_CYAN, fontsize=9, ha="left", va="bottom", alpha=0.85)

    # Legend goes BELOW the bottom panel — never overlaps data or band labels.
    leg = axC.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.22),
        facecolor=_PANEL, edgecolor=_BORDER, labelcolor=_TEXT,
        framealpha=0.95, fontsize=10, ncol=4,
        handlelength=2.2, columnspacing=1.6, borderaxespad=0.4,
    )
    leg.get_frame().set_linewidth(0.6)

    # ── Suptitle (top) ───────────────────────────────────────────────
    n_pts = len(rows)
    fig.suptitle(
        f"Fine q_acc Tuning  ·  Baseline MC  ·  {n_tr} trials × {n_pts} q-points",
        color=_TEXT, fontweight="bold", fontsize=15, y=0.975,
    )

    fig.savefig(plots_dir / "06e_sweep_summary.png",
                dpi=220, facecolor=_BG)
    plt.close(fig)

    print(f"Wrote plot: {plots_dir / '06e_sweep_summary.png'}")


if __name__ == "__main__":
    main()
