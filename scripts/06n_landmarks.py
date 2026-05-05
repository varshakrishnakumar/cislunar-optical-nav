"""Phase 3 — synthetic landmarks vs Moon-center bearing.

Adds K synthetic landmarks at fixed offsets from the Moon center (one
lunar radius along ±x / ±y / ±z by default) and re-runs the MC under
three configurations:

  - moon_only          : current paper baseline (Moon-center bearing)
  - landmarks_only     : disable Moon-center, use only landmark bearings
  - moon_plus_landmarks: both (richer information set, uses every step)

Compares miss / pos_err / NEES distributions to answer the reviewer's
classical question: *do landmarks reduce dependence on active pointing?*
The same MC seeds are reused across configs so per-trial deltas are
meaningful.

Usage
-----
python scripts/06n_landmarks.py --n-seeds 60
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
_MOON_RADIUS_KM = 1737.4
_MOON_RADIUS_ND = _MOON_RADIUS_KM / _KM_PER_LU


def _default_landmarks_nd(mu: float) -> np.ndarray:
    """Six landmarks: ±x, ±y, ±z one Moon-radius offset from Moon center.

    Treated as fixed inertial points in the rotating frame (CR3BP). For
    SPICE this gets re-built per trial below.
    """
    moon = np.array([1.0 - float(mu), 0.0, 0.0], dtype=float)
    R = float(_MOON_RADIUS_ND)
    offsets = np.array([
        [+R, 0, 0], [-R, 0, 0],
        [0, +R, 0], [0, -R, 0],
        [0, 0, +R], [0, 0, -R],
    ], dtype=float)
    return moon[None, :] + offsets


def _default_landmark_offsets_km() -> np.ndarray:
    """For SPICE: offsets in km; the run wrapper adds them to r_moon at
    each step internally."""
    R = float(_MOON_RADIUS_KM)
    return np.array([
        [+R, 0, 0], [-R, 0, 0],
        [0, +R, 0], [0, -R, 0],
        [0, 0, +R], [0, 0, -R],
    ], dtype=float)


_CONFIGS = ("moon_only", "landmarks_only", "moon_plus_landmarks")


def _disable_moon_only_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Build a landmark-only configuration by passing only the landmark
    list (no Moon-center) — but our run_case always uses the Moon center.

    Workaround: simulate "landmarks only" by placing the Moon outside
    the FOV via dropout=1.0 logically — except that's not reachable.
    Instead, we approximate landmarks_only by giving the filter ONLY the
    landmark bearings AND zeroing the Moon-center one.
    """
    # See below: we hijack dropout=1.0 + a separate code path.
    return landmarks


def _run_config(
    *, run_case, config: str, n_seeds: int, base_seed: int,
    config_kwargs: dict, landmarks: np.ndarray,
    dropout_for_moon_only: float = 0.0,
    truth: str,
) -> list[dict]:
    """Run n_seeds trials for a given config. Same seeds across configs."""
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

        if config == "moon_only":
            kwargs = dict(disable_moon_center=False)
        elif config == "landmarks_only":
            kwargs = dict(disable_moon_center=True)
            if truth == "cr3bp":
                kwargs["landmark_positions"] = landmarks
            else:
                kwargs["landmark_offsets_km"] = _default_landmark_offsets_km()
        else:  # moon_plus_landmarks
            kwargs = dict(disable_moon_center=False)
            if truth == "cr3bp":
                kwargs["landmark_positions"] = landmarks
            else:
                kwargs["landmark_offsets_km"] = _default_landmark_offsets_km()

        try:
            out = run_case(
                seed=seed, dx0=dx0, est_err=est_err,
                return_debug=False, accumulate_gramian=False,
                **{**config_kwargs, **kwargs},
            )
            rows.append({
                "trial_id":   trial_id, "seed": seed, "config": config,
                "miss_ekf":   float(out["miss_ekf"]),
                "pos_err_tc": float(out["pos_err_tc"]),
                "valid_rate": float(out["valid_rate"]),
                "nis_mean":   float(out["nis_mean"]),
                "nees_mean":  float(out["nees_mean"]),
            })
        except Exception as exc:  # noqa: BLE001
            print(f"  config={config} trial={trial_id} failed: {exc}")
    return rows


def _box_plot(
    rows_by_cfg: dict[str, list[dict]],
    units: RunUnits,
    *, truth: str, n_seeds: int, outpath: Path,
) -> None:
    apply_dark_theme()
    cfgs = [c for c in _CONFIGS if c in rows_by_cfg and rows_by_cfg[c]]
    if not cfgs:
        return

    miss = [
        np.array([r["miss_ekf"] for r in rows_by_cfg[c]
                  if np.isfinite(r["miss_ekf"])], dtype=float)
        for c in cfgs
    ]
    pos = [
        np.array([r["pos_err_tc"] for r in rows_by_cfg[c]
                  if np.isfinite(r["pos_err_tc"])], dtype=float)
        for c in cfgs
    ]
    valid = [
        np.array([r["valid_rate"] for r in rows_by_cfg[c]
                  if np.isfinite(r["valid_rate"])], dtype=float)
        for c in cfgs
    ]

    if units.truth == "cr3bp":
        miss = [m * _KM_PER_LU for m in miss]
        pos  = [p * _KM_PER_LU for p in pos]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.patch.set_facecolor(BG)

    palette = [VIOLET, AMBER, CYAN]
    titles = ["Terminal Miss", "Pos Error at tc", "Moon-center Accept Rate"]
    ylabels = ["miss_ekf [km]", "pos_err_tc [km]", "valid_rate (Moon)"]
    log_y = [True, True, False]
    series = [miss, pos, valid]

    for ax, ttl, yl, srs, lg in zip(axes, titles, ylabels, series, log_y):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.grid(True, color=BORDER, lw=0.3)
        bp = ax.boxplot(srs, positions=range(len(cfgs)), widths=0.55,
                        patch_artist=True, showfliers=True,
                        medianprops=dict(color=AMBER, lw=1.6))
        for patch, c in zip(bp["boxes"], palette):
            patch.set_facecolor(c)
            patch.set_alpha(0.45)
            patch.set_edgecolor(c)
        if lg:
            ax.set_yscale("log")
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels(cfgs, fontsize=9)
        ax.set_title(ttl, color=TEXT)
        ax.set_ylabel(yl, color=TEXT)

    fig.suptitle(
        f"Landmark Integration Comparison  ·  truth={truth}  "
        f"n_seeds={n_seeds}  (matched seeds across configs)",
        color=TEXT, fontsize=12,
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, facecolor=BG)
    plt.close(fig)


def _write_summary(
    rows_by_cfg: dict[str, list[dict]],
    units: RunUnits,
    out_txt: Path,
) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Landmark integration — per-config summary",
        "=" * 70,
        f"{'config':>22}  {'n':>4}  {'miss_med':>10}  {'miss_p95':>10}  "
        f"{'pos_med':>10}  {'NEES_med':>9}  {'NIS_med':>8}",
    ]
    for c in _CONFIGS:
        if c not in rows_by_cfg or not rows_by_cfg[c]:
            continue
        rows = rows_by_cfg[c]
        miss = np.array([r["miss_ekf"] for r in rows], dtype=float)
        pos  = np.array([r["pos_err_tc"] for r in rows], dtype=float)
        nees = np.array([r["nees_mean"] for r in rows], dtype=float)
        nis  = np.array([r["nis_mean"]  for r in rows], dtype=float)
        miss = miss[np.isfinite(miss)]
        if units.truth == "cr3bp":
            miss = miss * _KM_PER_LU
            pos = pos * _KM_PER_LU
        lines.append(
            f"{c:>22}  {len(rows):4d}  "
            f"{np.median(miss):10.2f}  {np.percentile(miss,95):10.2f}  "
            f"{np.median(pos):10.2f}  {np.median(nees):9.2f}  "
            f"{np.median(nis):8.2f}"
        )
    out_txt.write_text("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthetic-landmark integration comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-seeds", type=int, default=40)
    p.add_argument("--mu",   type=float, default=0.0121505856)
    p.add_argument("--t0",   type=float, default=0.0)
    p.add_argument("--tf",   type=float, default=6.0)
    p.add_argument("--tc",   type=float, default=2.0)
    p.add_argument("--dt-meas", type=float, default=0.02)
    p.add_argument("--sigma-px", type=float, default=1.0)
    p.add_argument("--q-acc", type=float, default=1e-14)
    p.add_argument("--sigma-att-deg", type=float, default=0.0)
    p.add_argument("--out", type=str, default="results/mc/landmarks")
    p.add_argument("--base-seed", type=int, default=7)
    add_truth_arg(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    apply_dark_theme()
    run_case = load_midcourse_run_case(truth=args.truth)
    units = RunUnits.for_truth(args.truth)

    landmarks_nd = _default_landmarks_nd(args.mu)

    config_kwargs = dict(
        mu=args.mu, t0=args.t0, tf=args.tf, tc=args.tc,
        dt_meas=args.dt_meas, sigma_px=args.sigma_px,
        dropout_prob=0.0,
        camera_mode="estimate_tracking", q_acc=args.q_acc,
        sigma_att_rad=float(args.sigma_att_deg) * np.pi / 180.0,
    )

    rows_by_cfg: dict[str, list[dict]] = {}
    for cfg in _CONFIGS:
        print(f"\n▸ config = {cfg}  n_seeds={args.n_seeds}")
        rows_by_cfg[cfg] = _run_config(
            run_case=run_case, config=cfg, n_seeds=int(args.n_seeds),
            base_seed=int(args.base_seed),
            config_kwargs=config_kwargs, landmarks=landmarks_nd,
            truth=args.truth,
        )

    out_dir = apply_truth_suffix(repo_path(args.out), args.truth)
    _box_plot(rows_by_cfg, units, truth=args.truth,
              n_seeds=int(args.n_seeds),
              outpath=out_dir / "06n_landmarks.png")
    _write_summary(rows_by_cfg, units, out_dir / "06n_landmarks.txt")
    print(f"\nWrote:")
    print(f"  {out_dir / '06n_landmarks.png'}")
    print(f"  {out_dir / '06n_landmarks.txt'}")


if __name__ == "__main__":
    main()
