"""Phase 2 — multi-revolution EKF stability.

Extends the EKF observation arc to span 2–3 halo periods (the baseline
halo near L1 has period ≈ π ND time units, so tf ∈ {6, 9, 12}
correspond to roughly 2 / 3 / 4 cycles). For each configuration runs N
seeds, captures the position error and NEES histories, and plots the
median / IQR envelope so the question "does the filter remain stable
over multiple revolutions?" can be answered visually.

The terminal correction logic is left in place — the only change is
moving tc *back* near tf so EKF coverage spans the full arc, and tf is
extended to multiple periods.

Usage
-----
python scripts/06l_multi_revolution.py --n-seeds 12 --tf-list 6 9 12
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

from _paper_constants import KM_PER_LU as _KM_PER_LU
_HALO_PERIOD_ND = np.pi  # ≈ period of an L1 Earth-Moon halo, used for
                          #   "n_revs" axis tickmarks.


def _run_long_arc_seeds(
    *, run_case, tf: float, tc: float, n_seeds: int,
    base_seed: int, config_kwargs: dict,
) -> dict:
    """Run n_seeds long-arc trials; return aggregated pos-err and NEES."""
    from mc.sampler import (
        make_trial_rng,
        sample_estimation_error,
        sample_injection_error,
    )

    pos_err_curves: list[np.ndarray] = []
    nees_curves:    list[np.ndarray] = []
    t_grid_ref = None
    for trial_id in range(int(n_seeds)):
        rng = make_trial_rng(base_seed, trial_id)
        seed = int(rng.integers(0, 2**31 - 1))
        dx0 = sample_injection_error(rng, sigma_r=1e-4, sigma_v=1e-4,
                                     planar_only=False)
        est_err = sample_estimation_error(rng, sigma_r=1e-4, sigma_v=1e-4,
                                          planar_only=False)
        try:
            out = run_case(
                seed=seed, dx0=dx0, est_err=est_err,
                tf=float(tf), tc=float(tc),
                return_debug=True, accumulate_gramian=False,
                **config_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  seed={seed} failed: {exc}")
            continue
        dbg = out["debug"]
        t_meas  = np.asarray(dbg["t_meas"], dtype=float)
        k_tc    = int(dbg["k_tc"])
        pos_err = np.asarray(dbg["pos_err_hist"], dtype=float)
        nees    = np.asarray(dbg["nees_hist"], dtype=float)

        # pos_err is the EKF history for k=1..k_tc (length k_tc).
        t_used = t_meas[1: k_tc + 1]
        if t_grid_ref is None:
            t_grid_ref = t_used
        if t_used.shape[0] != t_grid_ref.shape[0]:
            # Re-interpolate to the reference grid.
            pos_err = np.interp(t_grid_ref, t_used, pos_err)
            nees    = np.interp(t_grid_ref, t_used, nees)
        pos_err_curves.append(pos_err)
        nees_curves.append(nees)

    if not pos_err_curves:
        raise RuntimeError(f"All seeds failed for tf={tf}")

    return dict(
        t_grid=t_grid_ref,
        pos_err=np.asarray(pos_err_curves),  # (n_seeds, T)
        nees=np.asarray(nees_curves),
    )


def _envelope_plot(
    ax: plt.Axes, t: np.ndarray, curves: np.ndarray, color: str,
    *, label: str, log_y: bool = False,
) -> None:
    if curves.size == 0:
        return
    med = np.nanmedian(curves, axis=0)
    p25 = np.nanpercentile(curves, 25, axis=0)
    p75 = np.nanpercentile(curves, 75, axis=0)
    ax.fill_between(t, p25, p75, color=color, alpha=0.18)
    ax.plot(t, med, color=color, lw=1.8, label=label)
    if log_y:
        ax.set_yscale("log")


def _plot_summary(
    by_tf: dict[float, dict],
    units: RunUnits,
    *,
    truth: str,
    n_seeds: int,
    outpath: Path,
) -> None:
    apply_dark_theme()
    tfs = sorted(by_tf.keys())
    palette = [CYAN, AMBER, VIOLET, GREEN, RED]

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True,
                             sharex=True)
    fig.patch.set_facecolor(BG)

    # ── pos error vs time ────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, lw=0.3)

    for i, tf in enumerate(tfs):
        d = by_tf[tf]
        if units.truth == "cr3bp":
            curves = d["pos_err"] * _KM_PER_LU
            len_lab = "km"
        else:
            curves = d["pos_err"]
            len_lab = "km"
        _envelope_plot(ax, d["t_grid"], curves, palette[i % len(palette)],
                       label=f"tf = {tf:g}  ({tf/_HALO_PERIOD_ND:.2f} halo periods)",
                       log_y=True)
    ax.set_ylabel(f"‖r̂ − r‖  [{len_lab}]", color=TEXT)
    ax.set_title(
        "EKF Position Error over Multi-Revolution Arc  ·  IQR envelope",
        color=TEXT, fontweight="bold",
    )
    ax.legend(fontsize=9)

    # ── NEES vs time ─────────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, lw=0.3)

    nees_lo = chi2.ppf(0.025, df=6)
    nees_hi = chi2.ppf(0.975, df=6)
    # Use the largest tf for the band x-extent:
    t_full = by_tf[tfs[-1]]["t_grid"]
    ax.fill_between(t_full, nees_lo, nees_hi, color=GREEN, alpha=0.10,
                    label=f"95% χ²(6): [{nees_lo:.1f}, {nees_hi:.1f}]")
    ax.axhline(6.0, color=GREEN, lw=0.8, ls="--", alpha=0.6)
    for i, tf in enumerate(tfs):
        d = by_tf[tf]
        _envelope_plot(ax, d["t_grid"], d["nees"], palette[i % len(palette)],
                       label=f"tf = {tf:g}", log_y=False)
    ax.set_ylim(0, max(40.0, float(nees_hi) * 2.0))
    ax.set_ylabel("NEES  (mean across seeds)", color=TEXT)
    ax.set_xlabel("t  [ND CR3BP time]", color=TEXT)
    ax.set_title("NEES Consistency over Multi-Revolution Arc", color=TEXT,
                 fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle(
        f"Multi-Revolution EKF Stability  ·  truth={truth}  n_seeds={n_seeds}",
        color=TEXT, fontsize=13,
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, facecolor=BG)
    plt.close(fig)


def _write_summary(
    by_tf: dict[float, dict],
    units: RunUnits,
    out_txt: Path,
) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Multi-revolution EKF stability — endpoint summary",
             "=" * 64,
             f"{'tf':>6}  {'n_revs':>7}  {'final pos_err [km]':>21}  "
             f"{'NEES_band%':>11}"]
    nees_lo = chi2.ppf(0.025, df=6)
    nees_hi = chi2.ppf(0.975, df=6)
    for tf in sorted(by_tf.keys()):
        d = by_tf[tf]
        if units.truth == "cr3bp":
            err_end_km = np.nanmedian(d["pos_err"][:, -1]) * _KM_PER_LU
        else:
            err_end_km = np.nanmedian(d["pos_err"][:, -1])
        nees_flat = d["nees"][np.isfinite(d["nees"])]
        in_band = float(np.mean(
            (nees_flat >= nees_lo) & (nees_flat <= nees_hi)
        )) if nees_flat.size else float("nan")
        lines.append(
            f"{tf:6g}  {tf/_HALO_PERIOD_ND:7.2f}  "
            f"{err_end_km:21.2f}  {in_band*100:10.1f}%"
        )
    out_txt.write_text("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-revolution EKF stability sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--tf-list", type=float, nargs="+", default=[6.0, 9.0, 12.0])
    p.add_argument("--mu",   type=float, default=0.0121505856)
    p.add_argument("--t0",   type=float, default=0.0)
    p.add_argument("--dt-meas", type=float, default=0.02)
    p.add_argument("--sigma-px", type=float, default=1.0)
    p.add_argument("--q-acc", type=float, default=1e-14)
    p.add_argument("--out", type=str, default="results/mc/multi_revolution")
    p.add_argument("--base-seed", type=int, default=7)
    add_truth_arg(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    apply_dark_theme()
    run_case = load_midcourse_run_case(truth=args.truth)
    units = RunUnits.for_truth(args.truth)

    config_kwargs = dict(
        mu=args.mu, t0=args.t0,
        dt_meas=args.dt_meas, sigma_px=args.sigma_px, dropout_prob=0.0,
        camera_mode="estimate_tracking", q_acc=args.q_acc,
    )

    by_tf: dict[float, dict] = {}
    for tf in args.tf_list:
        # Move tc to ~95% of tf so the EKF runs across (almost) the whole
        # arc — this is the long-arc consistency question.
        tc = float(tf) * 0.95
        print(f"\n▸ tf={tf:g}  tc={tc:.3f}  n_seeds={args.n_seeds}")
        by_tf[float(tf)] = _run_long_arc_seeds(
            run_case=run_case, tf=float(tf), tc=tc,
            n_seeds=int(args.n_seeds), base_seed=int(args.base_seed),
            config_kwargs=config_kwargs,
        )

    out_dir = apply_truth_suffix(repo_path(args.out), args.truth)
    _plot_summary(by_tf, units, truth=args.truth,
                  n_seeds=int(args.n_seeds),
                  outpath=out_dir / "06l_multi_revolution.png")
    _write_summary(by_tf, units, out_dir / "06l_multi_revolution.txt")
    print(f"\nWrote:")
    print(f"  {out_dir / '06l_multi_revolution.png'}")
    print(f"  {out_dir / '06l_multi_revolution.txt'}")


if __name__ == "__main__":
    main()
