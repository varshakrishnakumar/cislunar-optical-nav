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

from _paper_constants import KM_PER_LU as _KM_PER_LU  # noqa: E402

_MOON_RADIUS_KM = 1737.4
_MOON_RADIUS_ND = _MOON_RADIUS_KM / _KM_PER_LU


def _unit_offsets(case: str) -> np.ndarray:
    """Return (N, 3) unit Moon-fixed offsets for the selected case."""
    if case == "synthetic_6":
        return np.array([
            [+1, 0, 0], [-1, 0, 0],
            [0, +1, 0], [0, -1, 0],
            [0, 0, +1], [0, 0, -1],
        ], dtype=float)
    if case == "synthetic_12":
        d = 1.0 / np.sqrt(2.0)
        return np.array([
            [+1, 0, 0], [-1, 0, 0],
            [0, +1, 0], [0, -1, 0],
            [0, 0, +1], [0, 0, -1],
            [+d, +d, 0], [-d, +d, 0], [+d, -d, 0], [-d, -d, 0],
            [+d, 0, +d], [0, +d, -d],
        ], dtype=float)
    if case in ("catalog_craters_6", "catalog_craters_12"):
        from cv.landmark_catalog import catalog_unit_offsets
        return catalog_unit_offsets(case)
    raise ValueError(f"Unknown landmark case: {case!r}")


def _landmarks_nd(mu: float, case: str) -> np.ndarray:
    """Build absolute landmark positions in the CR3BP rotating frame
    for the selected case."""
    moon = np.array([1.0 - float(mu), 0.0, 0.0], dtype=float)
    return moon[None, :] + _unit_offsets(case) * float(_MOON_RADIUS_ND)


def _landmark_offsets_km(case: str) -> np.ndarray:
    """Moon-fixed offsets in km for the SPICE arm."""
    return _unit_offsets(case) * float(_MOON_RADIUS_KM)


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
    *, truth: str, config: str, n_seeds: int, base_seed: int,
    config_kwargs: dict, landmarks_nd: np.ndarray,
    landmarks_km: np.ndarray, n_workers: int,
) -> list[dict]:
    """Run n_seeds trials for a given config. Same seeds across configs."""
    from _parallel_seeds import run_seeds_parallel

    if config == "moon_only":
        kw = {"disable_moon_center": False}
    elif config == "landmarks_only":
        kw = {"disable_moon_center": True}
        if truth == "cr3bp":
            kw["landmark_positions"] = landmarks_nd
        else:
            kw["landmark_offsets_km"] = landmarks_km
    else:  # moon_plus_landmarks
        kw = {"disable_moon_center": False}
        if truth == "cr3bp":
            kw["landmark_positions"] = landmarks_nd
        else:
            kw["landmark_offsets_km"] = landmarks_km

    rows = run_seeds_parallel(
        truth=truth, n_seeds=int(n_seeds), base_seed=int(base_seed),
        n_workers=int(n_workers),
        kwargs_extra={**config_kwargs, **kw},
        extract_fields=[
            ("miss_ekf",            "miss_ekf"),
            ("pos_err_tc",          "pos_err_tc"),
            ("valid_rate",          "valid_rate"),
            ("valid_rate_moon",     "valid_rate_moon"),
            ("valid_rate_landmarks","valid_rate_landmarks"),
            ("nis_mean",            "nis_mean"),
            ("nis_mean_all",        "nis_mean_all"),
            ("nis_mean_landmarks",  "nis_mean_landmarks"),
            ("nees_mean",           "nees_mean"),
        ],
        extra_row_fields={"config": config},
    )
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
    # accepted-update fraction per scheduled epoch (any source). Use
    # this instead of the legacy moon-only valid_rate so landmarks_only
    # isn't dragged to zero on a metric that doesn't apply.
    accepted = [
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
    titles = ["Terminal Miss", "Pos Error at tc", "Accepted-update fraction"]
    ylabels = ["miss_ekf [km]", "pos_err_tc [km]",
               "any update accepted / scheduled epoch"]
    log_y = [True, True, False]
    series = [miss, pos, accepted]

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
        "=" * 90,
        f"{'config':>22}  {'n':>4}  {'miss_med':>10}  {'miss_p95':>10}  "
        f"{'pos_med':>10}  {'NEES_med':>9}  "
        f"{'NIS_all_med':>11}  {'vr_any':>7}  {'vr_lmk':>7}",
    ]
    for c in _CONFIGS:
        if c not in rows_by_cfg or not rows_by_cfg[c]:
            continue
        rows = rows_by_cfg[c]
        miss = np.array([r["miss_ekf"] for r in rows], dtype=float)
        pos  = np.array([r["pos_err_tc"] for r in rows], dtype=float)
        nees = np.array([r["nees_mean"] for r in rows], dtype=float)
        nis_all = np.array([r["nis_mean_all"] for r in rows], dtype=float)
        vr_any = np.array([r["valid_rate"] for r in rows], dtype=float)
        vr_lmk = np.array([r["valid_rate_landmarks"] for r in rows],
                           dtype=float)
        miss = miss[np.isfinite(miss)]
        if units.truth == "cr3bp":
            miss = miss * _KM_PER_LU
            pos = pos * _KM_PER_LU
        lines.append(
            f"{c:>22}  {len(rows):4d}  "
            f"{np.median(miss):10.2f}  {np.percentile(miss,95):10.2f}  "
            f"{np.median(pos):10.2f}  {np.median(nees):9.2f}  "
            f"{np.nanmedian(nis_all):11.2f}  "
            f"{np.nanmedian(vr_any):7.3f}  {np.nanmedian(vr_lmk):7.3f}"
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
    p.add_argument("--n-workers", type=int, default=-1,
                   help="Process-pool size; -1 = cpu_count(); 1 = serial.")
    p.add_argument("--landmark-case", type=str, default="synthetic_6",
                   choices=("synthetic_6", "synthetic_12",
                            "catalog_craters_6", "catalog_craters_12"),
                   help="Which landmark set to use. 'synthetic_*' is the "
                        "geometric ±-axis set; 'catalog_craters_*' uses "
                        "real lunar-crater lat/lon coordinates with "
                        "identity assumed known.")
    add_truth_arg(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    apply_dark_theme()
    units = RunUnits.for_truth(args.truth)

    landmarks_nd = _landmarks_nd(args.mu, args.landmark_case)
    landmarks_km = _landmark_offsets_km(args.landmark_case)

    config_kwargs = dict(
        mu=args.mu, t0=args.t0, tf=args.tf, tc=args.tc,
        dt_meas=args.dt_meas, sigma_px=args.sigma_px,
        dropout_prob=0.0,
        camera_mode="estimate_tracking", q_acc=args.q_acc,
        sigma_att_rad=float(args.sigma_att_deg) * np.pi / 180.0,
    )

    rows_by_cfg: dict[str, list[dict]] = {}
    print(f"▸ landmark_case = {args.landmark_case}  "
          f"({landmarks_nd.shape[0]} landmarks)")
    for cfg in _CONFIGS:
        print(f"\n▸ config = {cfg}  n_seeds={args.n_seeds}  workers={args.n_workers}")
        rows_by_cfg[cfg] = _run_config(
            truth=str(args.truth), config=cfg, n_seeds=int(args.n_seeds),
            base_seed=int(args.base_seed),
            config_kwargs=config_kwargs,
            landmarks_nd=landmarks_nd, landmarks_km=landmarks_km,
            n_workers=int(args.n_workers),
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
