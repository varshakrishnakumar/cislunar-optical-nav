"""Phase 6 — Success-rate-vs-threshold central figure (presentation only).

This is a presentation/reframing script, not a new physics experiment.
It pulls existing n=1000 production CSVs (06_monte_carlo headline at
phase_d_production, the SPICE point-mass truth comparison, the 06r
landmarks-under-pointing-degradation grid) and renders the paper's
honest "truth in advertising" central figure: a terminal-miss ECDF and a
success-rate-vs-threshold curve, both spanning 1 km to 1000 km, with
markers at the screening tolerance (390 km), the precision-arrival
reality-check tolerance (39 km), and a tighter 25 km point.

Curves rendered (4 total, no clutter):
  C1  Baseline CR3BP truth, Moon-only, active estimate-driven pointing
  C2  SPICE point-mass truth, Moon-only, active estimate-driven pointing
  C3  CR3BP truth, Moon + L2 catalog landmarks, active estimate-driven
  C4  Uncorrected baseline (gray reference, miss_uncorrected from C1's
      production CSV — the same trials with no midcourse correction)

Outputs:
  06t_success_ecdf_central.png  — 2-panel figure (ECDF + success-vs-thr)
  06t_success_ecdf_central.txt  — pass rates at 25, 39, 390 km per curve
  06t_success_ecdf_central.csv  — interpolated success curves (LaTeX-friendly)

Usage:
  python scripts/06t_success_ecdf_central.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _analysis_common import (  # noqa: E402
    apply_dark_theme, AMBER, BG, BORDER, CYAN, GREEN, PANEL,
    RED, TEXT, VIOLET, ORANGE,
)
from _common import ensure_src_on_path, repo_path

ensure_src_on_path()


_KM_PER_LU = 384_400.0
_THRESHOLDS_KM_REPORT = (25.0, 39.0, 390.0)
_THRESHOLDS_KM_GRID   = np.logspace(0.0, 3.0, 121)  # 1 → 1000 km, 121 points


def _load_phase_d_csv(path: Path, *, miss_col: str, scale_to_km: bool) -> np.ndarray:
    """Load miss values (km) from phase_d-style CSV with one row per trial."""
    misses = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                v = float(row[miss_col])
            except (KeyError, ValueError):
                continue
            if not np.isfinite(v):
                continue
            misses.append(v * _KM_PER_LU if scale_to_km else v)
    return np.array(misses, dtype=float)


def _load_06r_filtered(
    path: Path, *, lm_config: str, pt_mode: str,
) -> np.ndarray:
    """Load miss values (km) from 06r grid CSV, filtered to one cell."""
    misses = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("lm_config") != lm_config or row.get("pt_mode") != pt_mode:
                continue
            try:
                v = float(row["miss_ekf"])
            except (KeyError, ValueError):
                continue
            if not np.isfinite(v):
                continue
            # 06r runs CR3BP truth, miss_ekf is in LU; convert to km.
            misses.append(v * _KM_PER_LU)
    return np.array(misses, dtype=float)


def _ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_x, F(x)) where F is the empirical CDF."""
    x = np.sort(x[np.isfinite(x)])
    n = x.size
    return x, np.arange(1, n + 1, dtype=float) / n


def _success_curve(x: np.ndarray, thr_grid_km: np.ndarray) -> np.ndarray:
    """Pass rate (fraction of trials with miss < thr) at each threshold."""
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.full_like(thr_grid_km, np.nan)
    return np.array([float((x < t).mean()) for t in thr_grid_km])


def _pass_at(x: np.ndarray, thr_km: float) -> float:
    x = x[np.isfinite(x)]
    return float((x < thr_km).mean()) if x.size else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/mc/phase_h_central_ecdf")
    args = ap.parse_args()

    out_dir = repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Source CSVs (all n=1000 except where noted).
    cr3bp_csv     = repo_path("results/mc/phase_d_production/06c_baseline_results.csv")
    spice_csv     = repo_path("results/mc/phase_d_production_spice/06c_baseline_results.csv")
    landmarks_csv = repo_path("results/mc/phase_f_landmarks_pointing/06r_landmarks_under_pointing_degradation.csv")

    for p in (cr3bp_csv, spice_csv, landmarks_csv):
        if not p.exists():
            raise SystemExit(f"missing source CSV: {p}")

    # ----- Load curves -----
    c1 = _load_phase_d_csv(cr3bp_csv, miss_col="miss_ekf", scale_to_km=True)
    c2 = _load_phase_d_csv(spice_csv, miss_col="miss_ekf", scale_to_km=False)
    c3 = _load_06r_filtered(
        landmarks_csv, lm_config="moon_plus_landmarks_L2", pt_mode="active_ideal",
    )
    c4 = _load_phase_d_csv(cr3bp_csv, miss_col="miss_uncorrected", scale_to_km=True)

    curves = [
        ("C1 Baseline CR3BP, Moon-only, active",       c1, CYAN,   "-",  2.0),
        ("C2 SPICE truth, Moon-only, active",          c2, AMBER,  "-",  2.0),
        ("C3 CR3BP, Moon + L2 catalog landmarks",      c3, GREEN,  "-",  2.0),
        ("C4 Uncorrected (no midcourse burn)",         c4, "#888888", "--", 1.4),
    ]

    # ----- Pass-rate table -----
    txt = out_dir / "06t_success_ecdf_central.txt"
    txt.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Success-rate-vs-threshold central figure (06t)",
        "=" * 100,
        "",
        f"{'curve':<48}  {'n':>4}  "
        + "  ".join(f"{'pass<' + str(int(t)) + 'km':>12}" for t in _THRESHOLDS_KM_REPORT)
        + f"  {'med [km]':>9}  {'p95 [km]':>9}",
    ]
    for label, vals, *_ in curves:
        n = int(np.isfinite(vals).sum())
        if n == 0:
            continue
        med = float(np.median(vals[np.isfinite(vals)]))
        p95 = float(np.percentile(vals[np.isfinite(vals)], 95))
        cells = "  ".join(
            f"{_pass_at(vals, t)*100:11.2f}%" for t in _THRESHOLDS_KM_REPORT
        )
        lines.append(
            f"{label:<48}  {n:4d}  {cells}  {med:9.2f}  {p95:9.2f}"
        )
    txt.write_text("\n".join(lines))

    # ----- CSV of interpolated success curves (one column per curve) -----
    csv_path = out_dir / "06t_success_ecdf_central.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["threshold_km"] + [c[0].split(" ", 1)[0] for c in curves]
        w.writerow(header)
        rates = [_success_curve(c[1], _THRESHOLDS_KM_GRID) for c in curves]
        for i, t in enumerate(_THRESHOLDS_KM_GRID):
            w.writerow([f"{t:.4f}"] + [f"{r[i]:.6f}" for r in rates])

    # ----- Figure: 2-panel (ECDF top, success-vs-threshold bottom) -----
    apply_dark_theme()
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True,
                             sharex=True)
    fig.patch.set_facecolor(BG)

    # Top panel: ECDF
    ax_e = axes[0]
    ax_e.set_facecolor(PANEL)
    for sp in ax_e.spines.values():
        sp.set_edgecolor(BORDER)
    ax_e.grid(True, color=BORDER, lw=0.3, which="both")
    for label, vals, color, ls, lw in curves:
        if vals.size == 0:
            continue
        x, F = _ecdf(vals)
        # Add (1e-3, 0) sentinel so log-x ECDF starts at the left edge.
        x = np.concatenate([[1e-3], x])
        F = np.concatenate([[0.0], F])
        ax_e.step(x, F, where="post", color=color, ls=ls, lw=lw, label=label)
    ax_e.set_xscale("log")
    ax_e.set_xlim(1.0, 1000.0)
    ax_e.set_ylim(0.0, 1.0)
    ax_e.set_ylabel("Empirical CDF  F(miss < x)", color=TEXT)
    ax_e.set_title(
        "Terminal-miss ECDF — n=1000 matched-seed Monte Carlo (production runs)",
        color=TEXT, fontsize=11,
    )
    for t in _THRESHOLDS_KM_REPORT:
        ax_e.axvline(t, color=RED, lw=0.8, ls=":", alpha=0.6)
        ax_e.text(t, 1.02, f"{int(t)} km", color=RED, fontsize=8,
                  ha="center", va="bottom")
    ax_e.legend(loc="lower right", fontsize=9, framealpha=0.85)

    # Bottom panel: success-rate vs threshold
    ax_s = axes[1]
    ax_s.set_facecolor(PANEL)
    for sp in ax_s.spines.values():
        sp.set_edgecolor(BORDER)
    ax_s.grid(True, color=BORDER, lw=0.3, which="both")
    for label, vals, color, ls, lw in curves:
        if vals.size == 0:
            continue
        rates = _success_curve(vals, _THRESHOLDS_KM_GRID)
        ax_s.plot(_THRESHOLDS_KM_GRID, rates, color=color, ls=ls, lw=lw,
                  label=label)
    ax_s.set_xscale("log")
    ax_s.set_xlim(1.0, 1000.0)
    ax_s.set_ylim(0.0, 1.0)
    ax_s.set_xlabel("Terminal-miss tolerance [km]  (log scale)", color=TEXT)
    ax_s.set_ylabel("Pass rate  P(miss < tolerance)", color=TEXT)
    ax_s.set_title(
        "Success rate vs mission tolerance — read mission-specific accuracy "
        "directly from the curve",
        color=TEXT, fontsize=11,
    )
    for t in _THRESHOLDS_KM_REPORT:
        ax_s.axvline(t, color=RED, lw=0.8, ls=":", alpha=0.6)
    # Annotate the marker thresholds with their CR3BP-baseline pass rate
    # (most informative single number per threshold).
    for t in _THRESHOLDS_KM_REPORT:
        rate = _pass_at(c1, t)
        ax_s.text(t, rate + 0.03,
                  f"{int(t)} km: {rate*100:.0f}%",
                  color=CYAN, fontsize=8, ha="center", va="bottom",
                  bbox=dict(facecolor=BG, edgecolor=CYAN, lw=0.5,
                            alpha=0.85, boxstyle="round,pad=0.2"))

    fig.suptitle(
        "Cislunar Bearing-Only OpNav: Terminal-Miss ECDF and Success-vs-Threshold",
        color=TEXT, fontsize=13,
    )

    png_path = out_dir / "06t_success_ecdf_central.png"
    fig.savefig(png_path, dpi=200, facecolor=BG)
    plt.close(fig)

    print(f"\nWrote:")
    print(f"  {txt}")
    print(f"  {csv_path}")
    print(f"  {png_path}")
    print(f"\nPass-rate table:")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
