"""Phase 4 (post-thesis-lock) — Landmarks Under Pointing Degradation.

Question this experiment answers
--------------------------------
The locked thesis claims active pointing preserves measurement
availability and landmarks improve bearing geometry, but treats them as
distinct mechanisms.  The advisor's question is whether landmarks
*reduce reliance* on aggressive active pointing.  This driver probes
that by sweeping a landmark-config × pointing-degradation matrix on
matched seeds, with the explicit hypothesis that:

  Landmarks improve bearing geometry when visible, but active pointing
  remains the measurement-availability mechanism; landmark diversity
  reduces performance *sensitivity* to pointing degradation rather than
  replacing pointing.

Falsification criterion (revised, post-user-review)
---------------------------------------------------
The above tempered claim is falsified if fixed-pointing landmark
configurations achieve near-active terminal performance while
maintaining adequate accepted-update fractions, because that would show
that bearing diversity can substitute for active measurement-
availability control in this geometry.  In that case the paper's
narrower active-pointing-necessity claim must be replaced.

Experiment matrix
-----------------
landmark_config  ∈ { moon_only, landmarks_only_L2, moon_plus_landmarks_L2 }
pointing_mode    ∈ { fixed,
                     active_ideal,
                     active_biased       (bias 0.1° about y),
                     active_lagged       (lag 5 steps ≈ 0.1 ND ≈ 10.6 h),
                     active_attitude_noisy (σ_att = 0.05° per step) }

3 × 5 = 15 cells, n_seeds matched across cells.

Metrics reported per cell
-------------------------
Existing fields:
  miss_ekf, pos_err_tc, valid_rate, valid_rate_moon,
  valid_rate_landmarks, nis_mean, nis_mean_all, nis_mean_landmarks,
  nees_mean, parallax_cumulative_rad

Phase-4 additions (per-source angular offset from commanded boresight):
  moon_offset_rad_med, landmark_offset_rad_med

The angular-offset pair lets us distinguish two mechanisms:
  • visibility-substitution  : landmarks accepted while moon is not
  • geometry-improvement     : landmarks accepted *in addition* to moon,
                               with both at small offset from boresight

Usage
-----
Smoke test:    python scripts/06r_landmarks_under_pointing_degradation.py --n-seeds 4
Production:    python scripts/06r_landmarks_under_pointing_degradation.py --n-seeds 1000 --n-workers -1
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
    AMBER, BG, BORDER, CYAN, GREEN, PANEL, RED, TEXT, VIOLET,
)
from _common import ensure_src_on_path, repo_path

ensure_src_on_path()
from utils.units import RunUnits  # noqa: E402

_KM_PER_LU = 384_400.0
_MOON_RADIUS_KM = 1737.4
_MOON_RADIUS_ND = _MOON_RADIUS_KM / _KM_PER_LU


# ----------------------------------------------------------------------
# Landmark layout (L2 catalog: Tycho/Copernicus/Aristarchus/Plato/Kepler/
# Grimaldi). Reuses the cv.landmark_catalog module already used by 06n.
# ----------------------------------------------------------------------

def _landmarks_nd(mu: float) -> np.ndarray:
    from cv.landmark_catalog import catalog_unit_offsets
    moon = np.array([1.0 - float(mu), 0.0, 0.0], dtype=float)
    return moon[None, :] + catalog_unit_offsets("catalog_craters_6") * float(_MOON_RADIUS_ND)


def _landmarks_km() -> np.ndarray:
    from cv.landmark_catalog import catalog_unit_offsets
    return catalog_unit_offsets("catalog_craters_6") * float(_MOON_RADIUS_KM)


# ----------------------------------------------------------------------
# Cell definitions: (landmark_config, pointing_mode) → kwargs for
# run_case. matched-seed Monte Carlo means every cell gets exactly the
# same trial_id sequence, with kwargs differing only in the listed
# fields.
# ----------------------------------------------------------------------

_LANDMARK_CONFIGS = ("moon_only", "landmarks_only_L2", "moon_plus_landmarks_L2")
_POINTING_MODES   = (
    "fixed", "active_ideal", "active_biased",
    "active_lagged", "active_attitude_noisy",
)


def _landmark_kwargs(
    config: str, *, truth: str,
    landmarks_nd: np.ndarray, landmarks_km: np.ndarray,
) -> dict:
    if config == "moon_only":
        return {"disable_moon_center": False}
    if config == "landmarks_only_L2":
        kw = {"disable_moon_center": True}
    elif config == "moon_plus_landmarks_L2":
        kw = {"disable_moon_center": False}
    else:
        raise ValueError(f"unknown landmark config: {config!r}")
    if truth == "cr3bp":
        kw["landmark_positions"] = landmarks_nd
    else:
        kw["landmark_offsets_km"] = landmarks_km
    return kw


def _pointing_kwargs(mode: str) -> dict:
    if mode == "fixed":
        return {"camera_mode": "fixed"}
    if mode == "active_ideal":
        return {"camera_mode": "estimate_tracking"}
    if mode == "active_biased":
        # 0.1 deg constant bias about camera-frame y axis (unmodeled boresight
        # offset). Same magnitude used in the existing 06o bias sweep.
        return {
            "camera_mode": "estimate_tracking",
            "bias_att_rad": (0.0, float(np.deg2rad(0.1)), 0.0),
        }
    if mode == "active_lagged":
        # 5 measurement steps of pointing lag (≈ 0.1 ND ≈ ~10.6 h at the
        # Earth-Moon scale). Slightly more aggressive than 06o's smallest
        # nonzero lag value to ensure a clean signal.
        return {"camera_mode": "estimate_tracking", "pointing_lag_steps": 5}
    if mode == "active_attitude_noisy":
        # 0.05 deg per-step zero-mean attitude noise. Sits near the
        # graceful-degradation knee from the 06q production sweep.
        return {
            "camera_mode": "estimate_tracking",
            "sigma_att_rad": float(np.deg2rad(0.05)),
        }
    raise ValueError(f"unknown pointing mode: {mode!r}")


# ----------------------------------------------------------------------
# Cell run + metric extraction
# ----------------------------------------------------------------------

_EXTRACT_FIELDS = [
    ("miss_ekf",                 "miss_ekf"),
    ("pos_err_tc",               "pos_err_tc"),
    ("valid_rate",               "valid_rate"),
    ("valid_rate_moon",          "valid_rate_moon"),
    ("valid_rate_landmarks",     "valid_rate_landmarks"),
    ("nis_mean",                 "nis_mean"),
    ("nis_mean_all",             "nis_mean_all"),
    ("nis_mean_landmarks",       "nis_mean_landmarks"),
    ("nees_mean",                "nees_mean"),
    ("parallax_cumulative_rad",  "parallax_cumulative_rad"),
    ("moon_offset_rad_med",      "moon_offset_rad_med"),
    ("landmark_offset_rad_med",  "landmark_offset_rad_med"),
]


def _run_cell(
    *, truth: str, lm_config: str, pt_mode: str,
    n_seeds: int, base_seed: int, common_kwargs: dict,
    landmarks_nd: np.ndarray, landmarks_km: np.ndarray, n_workers: int,
) -> list[dict]:
    from _parallel_seeds import run_seeds_parallel
    kw_lmk = _landmark_kwargs(
        lm_config, truth=truth,
        landmarks_nd=landmarks_nd, landmarks_km=landmarks_km,
    )
    kw_pt = _pointing_kwargs(pt_mode)
    return run_seeds_parallel(
        truth=truth, n_seeds=int(n_seeds), base_seed=int(base_seed),
        n_workers=int(n_workers),
        kwargs_extra={**common_kwargs, **kw_lmk, **kw_pt},
        extract_fields=_EXTRACT_FIELDS,
        extra_row_fields={"lm_config": lm_config, "pt_mode": pt_mode},
    )


# ----------------------------------------------------------------------
# Summary table + per-cell aggregation
# ----------------------------------------------------------------------

def _aggregate_cell(rows: list[dict], *, truth: str) -> dict:
    if not rows:
        return {"n": 0}
    miss = np.array([r["miss_ekf"] for r in rows], dtype=float)
    pos  = np.array([r["pos_err_tc"] for r in rows], dtype=float)
    nees = np.array([r["nees_mean"] for r in rows], dtype=float)
    nis_all = np.array([r["nis_mean_all"] for r in rows], dtype=float)
    vr_any = np.array([r["valid_rate"] for r in rows], dtype=float)
    vr_moon = np.array([r["valid_rate_moon"] for r in rows], dtype=float)
    vr_lmk  = np.array([r["valid_rate_landmarks"] for r in rows], dtype=float)
    moon_off = np.array([r["moon_offset_rad_med"] for r in rows], dtype=float)
    lmk_off  = np.array([r["landmark_offset_rad_med"] for r in rows], dtype=float)

    if truth == "cr3bp":
        miss = miss * _KM_PER_LU
        pos  = pos * _KM_PER_LU

    finite_miss = miss[np.isfinite(miss)]
    finite_pos  = pos[np.isfinite(pos)]
    return {
        "n":                len(rows),
        "miss_med":         float(np.median(finite_miss)) if finite_miss.size else float("nan"),
        "miss_p95":         float(np.percentile(finite_miss, 95)) if finite_miss.size else float("nan"),
        "pos_med":          float(np.median(finite_pos)) if finite_pos.size else float("nan"),
        "nees_med":         float(np.nanmedian(nees)),
        "nis_all_med":      float(np.nanmedian(nis_all)),
        "vr_any_med":       float(np.nanmedian(vr_any)),
        "vr_moon_med":      float(np.nanmedian(vr_moon)),
        "vr_lmk_med":       float(np.nanmedian(vr_lmk)),
        "moon_off_deg_med": float(np.degrees(np.nanmedian(moon_off))),
        "lmk_off_deg_med":  float(np.degrees(np.nanmedian(lmk_off))),
    }


def _write_summary(
    rows_by_cell: dict[tuple[str, str], list[dict]],
    *, truth: str, out_txt: Path,
) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Landmarks Under Pointing Degradation (06r)",
        "=" * 110,
        f"truth = {truth}",
        "",
        f"{'lm_config':>22}  {'pt_mode':>22}  "
        f"{'n':>4}  {'miss_med':>9}  {'miss_p95':>9}  "
        f"{'pos_med':>8}  {'NEES':>7}  {'NIS_all':>7}  "
        f"{'vr_any':>6}  {'vr_M':>6}  {'vr_L':>6}  "
        f"{'M_off°':>7}  {'L_off°':>7}",
    ]
    for lm in _LANDMARK_CONFIGS:
        for pt in _POINTING_MODES:
            cell = rows_by_cell.get((lm, pt), [])
            agg = _aggregate_cell(cell, truth=truth)
            if not agg.get("n"):
                continue
            lines.append(
                f"{lm:>22}  {pt:>22}  {agg['n']:4d}  "
                f"{agg['miss_med']:9.2f}  {agg['miss_p95']:9.2f}  "
                f"{agg['pos_med']:8.2f}  {agg['nees_med']:7.2f}  "
                f"{agg['nis_all_med']:7.2f}  "
                f"{agg['vr_any_med']:6.3f}  {agg['vr_moon_med']:6.3f}  "
                f"{agg['vr_lmk_med']:6.3f}  "
                f"{agg['moon_off_deg_med']:7.3f}  {agg['lmk_off_deg_med']:7.3f}"
            )
    out_txt.write_text("\n".join(lines))


# ----------------------------------------------------------------------
# Plot: heatmap of median terminal miss across the 3×5 grid, with
# accepted-update fraction shown in a companion panel.
# ----------------------------------------------------------------------

def _heatmap(rows_by_cell, *, truth: str, n_seeds: int, outpath: Path) -> None:
    apply_dark_theme()
    miss_grid = np.full((len(_LANDMARK_CONFIGS), len(_POINTING_MODES)), np.nan)
    vr_grid   = np.full_like(miss_grid, np.nan)
    moon_off_grid = np.full_like(miss_grid, np.nan)
    lmk_off_grid  = np.full_like(miss_grid, np.nan)
    for i, lm in enumerate(_LANDMARK_CONFIGS):
        for j, pt in enumerate(_POINTING_MODES):
            agg = _aggregate_cell(rows_by_cell.get((lm, pt), []), truth=truth)
            if agg.get("n"):
                miss_grid[i, j]     = agg["miss_med"]
                vr_grid[i, j]       = agg["vr_any_med"]
                moon_off_grid[i, j] = agg["moon_off_deg_med"]
                lmk_off_grid[i, j]  = agg["lmk_off_deg_med"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    fig.patch.set_facecolor(BG)
    panels = [
        (axes[0, 0], miss_grid, "Median terminal miss [km]", "viridis", False, "{:.0f}"),
        (axes[0, 1], vr_grid,   "Accepted-update fraction (any source)", "magma", False, "{:.2f}"),
        (axes[1, 0], moon_off_grid, "Median moon-center offset [deg]", "cividis", False, "{:.1f}"),
        (axes[1, 1], lmk_off_grid,  "Median landmark offset [deg]",   "cividis", False, "{:.1f}"),
    ]
    for ax, grid, title, cmap, log_c, fmt in panels:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        finite_mask = np.isfinite(grid)
        if not finite_mask.any():
            ax.set_title(title + "  (no data)", color=TEXT)
            continue
        im = ax.imshow(grid, aspect="auto", cmap=cmap, origin="upper",
                       interpolation="nearest")
        ax.set_xticks(range(len(_POINTING_MODES)))
        ax.set_xticklabels(_POINTING_MODES, rotation=30, ha="right",
                           fontsize=8, color=TEXT)
        ax.set_yticks(range(len(_LANDMARK_CONFIGS)))
        ax.set_yticklabels(_LANDMARK_CONFIGS, fontsize=9, color=TEXT)
        ax.set_title(title, color=TEXT, fontsize=11)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, fmt.format(grid[i, j]),
                            ha="center", va="center", fontsize=8,
                            color="white", weight="bold")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(colors=TEXT)

    fig.suptitle(
        f"Landmarks Under Pointing Degradation  ·  truth={truth}  "
        f"n_seeds={n_seeds}  (matched seeds across all 15 cells)",
        color=TEXT, fontsize=12,
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, facecolor=BG)
    plt.close(fig)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Landmarks Under Pointing Degradation (06r).",
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
    p.add_argument("--out",       type=str, default="results/mc/landmarks_pointing_degradation")
    p.add_argument("--base-seed", type=int, default=7)
    p.add_argument("--n-workers", type=int, default=-1,
                   help="Process-pool size; -1 = cpu_count(); 1 = serial.")
    add_truth_arg(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    apply_dark_theme()

    landmarks_nd = _landmarks_nd(args.mu)
    landmarks_km = _landmarks_km()

    common_kwargs = dict(
        mu=args.mu, t0=args.t0, tf=args.tf, tc=args.tc,
        dt_meas=args.dt_meas, sigma_px=args.sigma_px,
        dropout_prob=0.0, q_acc=args.q_acc,
    )

    rows_by_cell: dict[tuple[str, str], list[dict]] = {}
    print(f"\n▸ 06r Landmarks Under Pointing Degradation")
    print(f"  truth         = {args.truth}")
    print(f"  n_seeds       = {args.n_seeds}  (matched across cells)")
    print(f"  n_workers     = {args.n_workers}")
    print(f"  landmarks     = catalog_craters_6 (Tycho/Copernicus/"
          f"Aristarchus/Plato/Kepler/Grimaldi)")
    print(f"  matrix        = {len(_LANDMARK_CONFIGS)} × {len(_POINTING_MODES)}"
          f" = {len(_LANDMARK_CONFIGS) * len(_POINTING_MODES)} cells\n")
    cell_idx = 0
    n_cells = len(_LANDMARK_CONFIGS) * len(_POINTING_MODES)
    for lm in _LANDMARK_CONFIGS:
        for pt in _POINTING_MODES:
            cell_idx += 1
            print(f"  [{cell_idx:2d}/{n_cells}] lm={lm:>22}  pt={pt:>22}")
            rows_by_cell[(lm, pt)] = _run_cell(
                truth=str(args.truth), lm_config=lm, pt_mode=pt,
                n_seeds=int(args.n_seeds), base_seed=int(args.base_seed),
                common_kwargs=common_kwargs,
                landmarks_nd=landmarks_nd, landmarks_km=landmarks_km,
                n_workers=int(args.n_workers),
            )

    out_dir = apply_truth_suffix(repo_path(args.out), args.truth)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "06r_landmarks_under_pointing_degradation.txt"
    plot_path    = out_dir / "06r_landmarks_under_pointing_degradation.png"

    _write_summary(rows_by_cell, truth=str(args.truth), out_txt=summary_path)
    _heatmap(rows_by_cell, truth=str(args.truth),
             n_seeds=int(args.n_seeds), outpath=plot_path)

    # Save raw rows as CSV for later analysis / paper inclusion.
    import csv
    csv_path = out_dir / "06r_landmarks_under_pointing_degradation.csv"
    fieldnames = ["lm_config", "pt_mode", "trial_id", "seed"] + [
        out_key for out_key, _ in _EXTRACT_FIELDS
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for (lm, pt), rows in rows_by_cell.items():
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\n  Wrote:")
    print(f"    {summary_path}")
    print(f"    {plot_path}")
    print(f"    {csv_path}")


if __name__ == "__main__":
    main()
