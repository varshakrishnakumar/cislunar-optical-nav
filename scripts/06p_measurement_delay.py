"""Phase 4 — measurement-delay sensitivity.

Sweeps the camera-to-filter measurement-delay (in measurement steps) and
captures the dual-metric story the user already observed: terminal miss
inflates while filter consistency (NIS) can stay innocent. Specifically
shows that an EKF that *believes* its bearing was instantaneous but is
actually fed a delayed measurement gets bias-injected through the
state-transition mismatch — the filter has no way to detect it from
innovations alone, but the operational cost (miss) is real.

Usage
-----
python scripts/06p_measurement_delay.py --n-seeds 30
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


def _run_one_delay(
    *, run_case, delay_steps: int, n_seeds: int, base_seed: int,
    config_kwargs: dict,
) -> list[dict]:
    from mc.sampler import (
        make_trial_rng,
        sample_estimation_error,
        sample_injection_error,
    )

    rows: list[dict] = []
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
                meas_delay_steps=int(delay_steps),
                return_debug=False, accumulate_gramian=False,
                **config_kwargs,
            )
            rows.append({
                "trial_id":      trial_id, "seed": seed,
                "delay_steps":   int(delay_steps),
                "miss_ekf":      float(out["miss_ekf"]),
                "pos_err_tc":    float(out["pos_err_tc"]),
                "nis_mean":      float(out["nis_mean"]),
                "nees_mean":     float(out["nees_mean"]),
                "valid_rate":    float(out["valid_rate"]),
            })
        except Exception as exc:  # noqa: BLE001
            print(f"  delay={delay_steps} trial={trial_id} failed: {exc}")
    return rows


def _plot_delay(
    rows_by_delay: dict[int, list[dict]],
    units: RunUnits,
    *, dt_meas: float, truth: str, n_seeds: int, outpath: Path,
) -> None:
    apply_dark_theme()
    delays = sorted(rows_by_delay.keys())
    delays_dt = [d * float(dt_meas) for d in delays]  # ND time

    miss = [
        np.array([r["miss_ekf"] for r in rows_by_delay[d]
                  if np.isfinite(r["miss_ekf"])], dtype=float)
        for d in delays
    ]
    nis = [
        np.array([r["nis_mean"] for r in rows_by_delay[d]
                  if np.isfinite(r["nis_mean"])], dtype=float)
        for d in delays
    ]
    nees = [
        np.array([r["nees_mean"] for r in rows_by_delay[d]
                  if np.isfinite(r["nees_mean"])], dtype=float)
        for d in delays
    ]
    if units.truth == "cr3bp":
        miss = [m * _KM_PER_LU for m in miss]

    miss_med = np.array([np.median(m) if m.size else np.nan for m in miss])
    miss_p95 = np.array([np.percentile(m, 95) if m.size else np.nan for m in miss])
    nis_med  = np.array([np.median(n) if n.size else np.nan for n in nis])
    nees_med = np.array([np.median(n) if n.size else np.nan for n in nees])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, lw=0.3)
    ax.fill_between(delays, miss_med, miss_p95, color=VIOLET, alpha=0.18,
                    label="median–p95 band")
    ax.plot(delays, miss_med, "o-", color=VIOLET, lw=1.8, ms=6,
            label="median miss")
    ax.plot(delays, miss_p95, "s--", color=AMBER, lw=1.4, ms=5,
            label="p95 miss")
    ax.axhline(39.0, color=GREEN, lw=0.8, ls=":", alpha=0.7,
               label="tight tol (39 km)")
    ax.axhline(390.0, color=RED, lw=0.8, ls=":", alpha=0.7,
               label="paper tol (390 km)")
    ax.set_xlabel("measurement delay  [steps]", color=TEXT)
    ax.set_ylabel("miss_ekf  [km]", color=TEXT)
    ax.set_yscale("log")
    ax.set_title("Operational cost  (terminal miss)", color=TEXT,
                 fontweight="bold")
    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda x: x * float(dt_meas), lambda x: x / float(dt_meas)),
    )
    secax.set_xlabel("delay  [ND CR3BP time]", color=TEXT)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, lw=0.3)
    ax.plot(delays, nis_med, "o-", color=CYAN, lw=1.8, ms=6,
            label="median NIS")
    ax.plot(delays, nees_med / 3.0, "s--", color=AMBER, lw=1.4, ms=5,
            label="median NEES / 3")
    ax.axhline(2.0, color=GREEN, lw=0.8, ls="--", alpha=0.6,
               label="NIS ideal = 2")
    ax.set_xlabel("measurement delay  [steps]", color=TEXT)
    ax.set_ylabel("filter consistency", color=TEXT)
    ax.set_title("Innovation health  (looks fine while miss explodes)",
                 color=TEXT, fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle(
        f"Measurement-Delay Sensitivity  ·  truth={truth}  "
        f"n_seeds={n_seeds}  (dt_meas={dt_meas:g})",
        color=TEXT, fontsize=13,
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, facecolor=BG)
    plt.close(fig)


def _write_summary(
    rows_by_delay: dict[int, list[dict]],
    units: RunUnits,
    *, dt_meas: float, out_txt: Path,
) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    scale = _KM_PER_LU if units.truth == "cr3bp" else 1.0
    lines = [
        "Measurement-delay sweep — per-delay summary",
        "=" * 70,
        f"{'delay':>6}  {'Δt [ND]':>10}  {'miss_med [km]':>14}  "
        f"{'miss_p95 [km]':>14}  {'NIS_med':>9}  {'NEES_med':>10}",
    ]
    for d in sorted(rows_by_delay.keys()):
        rows = rows_by_delay[d]
        miss = np.array([r["miss_ekf"] for r in rows], dtype=float) * scale
        nis  = np.array([r["nis_mean"] for r in rows], dtype=float)
        nees = np.array([r["nees_mean"] for r in rows], dtype=float)
        miss = miss[np.isfinite(miss)]
        nis  = nis[np.isfinite(nis)]
        nees = nees[np.isfinite(nees)]
        lines.append(
            f"{d:6d}  {d * dt_meas:10.4f}  "
            f"{np.median(miss):14.2f}  {np.percentile(miss, 95):14.2f}  "
            f"{np.median(nis):9.2f}  {np.median(nees):10.2f}"
        )
    out_txt.write_text("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measurement-delay sensitivity sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--delay-steps-list", type=int, nargs="+",
                   default=[0, 1, 2, 5, 10, 20, 50])
    p.add_argument("--mu",  type=float, default=0.0121505856)
    p.add_argument("--t0",  type=float, default=0.0)
    p.add_argument("--tf",  type=float, default=6.0)
    p.add_argument("--tc",  type=float, default=2.0)
    p.add_argument("--dt-meas",  type=float, default=0.02)
    p.add_argument("--sigma-px", type=float, default=1.0)
    p.add_argument("--q-acc",  type=float, default=1e-14)
    p.add_argument("--out", type=str, default="results/mc/measurement_delay")
    p.add_argument("--base-seed", type=int, default=7)
    add_truth_arg(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    apply_dark_theme()
    run_case = load_midcourse_run_case(truth=args.truth)
    units = RunUnits.for_truth(args.truth)

    config_kwargs = dict(
        mu=args.mu, t0=args.t0, tf=args.tf, tc=args.tc,
        dt_meas=args.dt_meas, sigma_px=args.sigma_px, dropout_prob=0.0,
        camera_mode="estimate_tracking", q_acc=args.q_acc,
    )

    rows_by_delay: dict[int, list[dict]] = {}
    for d in args.delay_steps_list:
        print(f"\n▸ delay = {d} steps  (Δt = {d * args.dt_meas:.4f} ND)")
        rows_by_delay[int(d)] = _run_one_delay(
            run_case=run_case, delay_steps=int(d),
            n_seeds=int(args.n_seeds), base_seed=int(args.base_seed),
            config_kwargs=config_kwargs,
        )

    out_dir = apply_truth_suffix(repo_path(args.out), args.truth)
    _plot_delay(rows_by_delay, units, dt_meas=float(args.dt_meas),
                truth=args.truth, n_seeds=int(args.n_seeds),
                outpath=out_dir / "06p_measurement_delay.png")
    _write_summary(rows_by_delay, units, dt_meas=float(args.dt_meas),
                   out_txt=out_dir / "06p_measurement_delay.txt")
    print(f"\nWrote:")
    print(f"  {out_dir / '06p_measurement_delay.png'}")
    print(f"  {out_dir / '06p_measurement_delay.txt'}")


if __name__ == "__main__":
    main()
