"""Phase B validation — direct comparison of CR3BP vs SPICE Monte Carlo runs.

Reads two ``06c_*_results.csv`` files written by 06_monte_carlo.py (one per
truth mode), reports key statistics, and renders side-by-side distribution
plots. The point of this report is to surface model mismatch where it
shows up first — in NEES (overconfident filter under unmodeled dynamics)
and in the tails of the miss/burn distributions.

Usage:
    python3 scripts/06f_compare_truth_modes.py \
        --cr3bp-csv results/mc/phase_b_baseline/06c_baseline_results.csv \
        --spice-csv results/mc/phase_b_baseline_spice/06c_baseline_results.csv \
        --out-dir   results/mc/phase_b_baseline_compare

The script does NOT rerun MC — it consumes already-written CSVs. Re-run
06_monte_carlo.py twice (once per --truth) before invoking this.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2

from _analysis_common import (
    AMBER as _AMBER,
    BG as _BG,
    BORDER as _BORDER,
    CYAN as _CYAN,
    GREEN as _GREEN,
    PANEL as _PANEL,
    RED as _RED,
    TEXT as _TEXT,
    VIOLET as _VIOLET,
    apply_dark_theme as _apply_dark_theme,
)
from _common import ensure_src_on_path, repo_path

ensure_src_on_path()
from utils.units import RunUnits  # noqa: E402

# χ² gates on filter-consistency metrics (6D EKF state, 2D bearing innovation).
_LB6, _UB6 = float(chi2.ppf(0.025, 6)), float(chi2.ppf(0.975, 6))
_LB2, _UB2 = float(chi2.ppf(0.025, 2)), float(chi2.ppf(0.975, 2))

# Single source of truth for the CR3BP-ND ↔ km conversion factor.
_LU_KM = RunUnits.for_truth("cr3bp").length_km_per_nd


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open() as f:
        return list(csv.DictReader(f))


def _floats(rows: list[dict], key: str) -> np.ndarray:
    out: list[float] = []
    for r in rows:
        try:
            out.append(float(r.get(key, "nan")))
        except (TypeError, ValueError):
            out.append(float("nan"))
    arr = np.array(out, dtype=float)
    return arr[np.isfinite(arr)]


def _summary(rows: list[dict], truth: str) -> dict:
    miss     = _floats(rows, "miss_ekf")
    nees     = _floats(rows, "nees_mean")
    nis      = _floats(rows, "nis_mean")
    dv_delta = _floats(rows, "dv_delta_mag")
    dv_infl  = _floats(rows, "dv_inflation_pct") * 100.0
    pos_err  = _floats(rows, "pos_err_tc")

    # Unit-aligned miss in km for cross-mode comparison. CR3BP miss is in
    # dimensionless lengths (multiply by _LU_KM); SPICE miss is already km.
    miss_km = miss * _LU_KM if truth == "cr3bp" else miss

    def _stat(arr: np.ndarray, q: float) -> float:
        return float(np.percentile(arr, q)) if arr.size else float("nan")

    return {
        "truth":            truth,
        "n":                len(rows),
        "miss_med":         _stat(miss, 50),
        "miss_p95":         _stat(miss, 95),
        "miss_p05":         _stat(miss, 5),
        "miss_med_km":      _stat(miss_km, 50),
        "miss_p95_km":      _stat(miss_km, 95),
        "nees_med":         _stat(nees, 50),
        "nees_p95":         _stat(nees, 95),
        "nees_in_band":     float(np.mean((nees >= _LB6) & (nees <= _UB6))) if nees.size else float("nan"),
        "nis_med":          _stat(nis, 50),
        "nis_p95":          _stat(nis, 95),
        "nis_in_band":      float(np.mean((nis  >= _LB2) & (nis  <= _UB2))) if nis.size  else float("nan"),
        "dv_delta_med":     _stat(dv_delta, 50),
        "dv_delta_p95":     _stat(dv_delta, 95),
        "dv_infl_med":      _stat(dv_infl, 50),
        "pos_err_tc_med":   _stat(pos_err, 50),
        "_arrays": {
            "miss": miss, "miss_km": miss_km,
            "nees": nees, "nis": nis,
            "dv_delta": dv_delta, "dv_infl": dv_infl, "pos_err": pos_err,
        },
    }


def _print_table(s_cr3bp: dict, s_spice: dict, *, miss_unit_cr3bp: str,
                 miss_unit_spice: str) -> str:
    """Side-by-side text table; returns the rendered string for both
    stdout and a saved .txt artifact."""
    def _fmt(v: float, fmt: str) -> str:
        if not np.isfinite(v):
            return "    n/a"
        return format(v, fmt)

    def _ratio(a: float, b: float) -> str:
        if not (np.isfinite(a) and np.isfinite(b)) or b == 0.0:
            return "—"
        return f"{a / b:5.2f}×"

    rows: list[tuple[str, str, str, str]] = [
        ("Final miss   median  (native units)",
         _fmt(s_cr3bp['miss_med'], '10.3e') + f"  [{miss_unit_cr3bp}]",
         _fmt(s_spice['miss_med'], '10.3e') + f"  [{miss_unit_spice}]",
         "—  (units differ)"),
        ("Final miss   median  [km, unit-aligned]",
         _fmt(s_cr3bp['miss_med_km'], '10.1f'),
         _fmt(s_spice['miss_med_km'], '10.1f'),
         _ratio(s_spice['miss_med_km'], s_cr3bp['miss_med_km'])),
        ("Final miss   p95     [km, unit-aligned]",
         _fmt(s_cr3bp['miss_p95_km'], '10.1f'),
         _fmt(s_spice['miss_p95_km'], '10.1f'),
         _ratio(s_spice['miss_p95_km'], s_cr3bp['miss_p95_km'])),
        ("NEES median",
         _fmt(s_cr3bp['nees_med'], '10.2f'),
         _fmt(s_spice['nees_med'], '10.2f'),
         _ratio(s_spice['nees_med'], s_cr3bp['nees_med'])),
        ("NEES p95",
         _fmt(s_cr3bp['nees_p95'], '10.2f'),
         _fmt(s_spice['nees_p95'], '10.2f'),
         _ratio(s_spice['nees_p95'], s_cr3bp['nees_p95'])),
        (f"NEES in χ²(6) band [{_LB6:.2f}, {_UB6:.2f}]",
         _fmt(s_cr3bp['nees_in_band'] * 100, '9.1f') + " %",
         _fmt(s_spice['nees_in_band'] * 100, '9.1f') + " %",
         "—"),
        ("NIS  median",
         _fmt(s_cr3bp['nis_med'], '10.2f'),
         _fmt(s_spice['nis_med'], '10.2f'),
         _ratio(s_spice['nis_med'], s_cr3bp['nis_med'])),
        (f"NIS  in χ²(2) band [{_LB2:.2f}, {_UB2:.2f}]",
         _fmt(s_cr3bp['nis_in_band'] * 100, '9.1f') + " %",
         _fmt(s_spice['nis_in_band'] * 100, '9.1f') + " %",
         "—"),
        ("|Δv_EKF − Δv_perfect|  median  (native)",
         _fmt(s_cr3bp['dv_delta_med'], '10.3e'),
         _fmt(s_spice['dv_delta_med'], '10.3e'),
         "—"),
    ]

    lines = []
    lines.append("═" * 90)
    lines.append(f"  Phase B validation — CR3BP vs SPICE   "
                 f"(n_cr3bp={s_cr3bp['n']}, n_spice={s_spice['n']})")
    lines.append("═" * 90)
    lines.append(f"  {'metric':<38} {'CR3BP':>20} {'SPICE':>20} {'ratio':>9}")
    lines.append("  " + "─" * 88)
    for name, c, s, r in rows:
        lines.append(f"  {name:<38} {c:>20} {s:>20} {r:>9}")
    lines.append("═" * 90)

    # Verbal interpretation — distinguishes "filter is overconfident" (NEES
    # inflates) from "filter absorbs mismatch but pays in miss" (consistency
    # holds, operational cost shows up downstream).
    lines.append("")
    lines.append("  Interpretation")
    lines.append("  ──────────────")
    nees_ratio = s_spice['nees_med'] / s_cr3bp['nees_med'] if s_cr3bp['nees_med'] else float('nan')
    miss_ratio = (s_spice['miss_med_km'] / s_cr3bp['miss_med_km']
                  if s_cr3bp['miss_med_km'] else float('nan'))

    nees_inflates = np.isfinite(nees_ratio) and nees_ratio > 1.5
    miss_inflates = np.isfinite(miss_ratio) and miss_ratio > 1.3

    if nees_inflates and miss_inflates:
        lines.append(f"  • Both NEES ({nees_ratio:.1f}×) and miss ({miss_ratio:.1f}×)")
        lines.append(f"    inflate under SPICE — classic model-mismatch signature:")
        lines.append(f"    filter is overconfident AND pays in operational cost.")
    elif miss_inflates and not nees_inflates:
        lines.append(f"  • Miss distance inflates {miss_ratio:.1f}× under SPICE while")
        lines.append(f"    NEES stays within band (ratio {nees_ratio:.2f}×). The filter")
        lines.append(f"    absorbs ephemeris-driven residuals via its grown covariance")
        lines.append(f"    (consistent in the χ² sense) but pays for it in mission")
        lines.append(f"    accuracy. This is the mismatch surfacing as an OPERATIONAL")
        lines.append(f"    cost rather than a CONSISTENCY violation. Phase D headline.")
    elif nees_inflates and not miss_inflates:
        lines.append(f"  • NEES inflates ({nees_ratio:.1f}×) but miss is comparable —")
        lines.append(f"    filter is statistically overconfident but operationally")
        lines.append(f"    fine. Likely q_acc undertuned for SPICE truth. Re-tune.")
    else:
        lines.append(f"  • NEES ratio = {nees_ratio:.2f}×, miss ratio = {miss_ratio:.2f}×.")
        lines.append(f"    Mismatch is mild at this regime. Try a longer arc, larger")
        lines.append(f"    σ_inj, or smaller q_acc to surface differences.")

    nis_band_drop = s_cr3bp['nis_in_band'] - s_spice['nis_in_band']
    if np.isfinite(nis_band_drop) and nis_band_drop > 0.05:
        lines.append(f"  • NIS in-band drops {nis_band_drop * 100:.1f} pp under SPICE —")
        lines.append(f"    individual innovations falling outside their predicted")
        lines.append(f"    covariance. Worth digging into per-step innovations.")

    # Bar-Shalom band-membership commentary
    if np.isfinite(s_spice['nees_in_band']):
        if s_spice['nees_in_band'] >= 0.90:
            lines.append(f"  • SPICE NEES in-band fraction {s_spice['nees_in_band']*100:.0f}%")
            lines.append(f"    ≥ 90% (Bar-Shalom textbook gate) — filter is consistent")
            lines.append(f"    even under high-fidelity truth.")
        elif s_spice['nees_in_band'] < 0.50:
            lines.append(f"  • SPICE NEES in-band fraction {s_spice['nees_in_band']*100:.0f}%")
            lines.append(f"    < 50% — filter consistency seriously degraded by SPICE")
            lines.append(f"    truth. Investigate q_acc tuning or filter dynamics.")

    lines.append("")
    lines.append("  Phase C: align units, rescale thresholds, update axis labels.")
    lines.append("  Phase D: the deltas above become the paper's headline plots.")
    return "\n".join(lines)


def _theme_axes(ax) -> None:
    ax.set_facecolor(_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(_BORDER); sp.set_linewidth(0.7)
    ax.tick_params(colors=_TEXT, labelsize=8.5)
    ax.xaxis.label.set_color(_TEXT); ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    ax.grid(True, color=_BORDER, lw=0.4)


def _hist_pair(ax, vals_cr3bp: np.ndarray, vals_spice: np.ndarray,
               *, title: str, xlabel: str,
               band_lb: float | None = None, band_ub: float | None = None,
               band_label: str = "") -> None:
    """Side-by-side histograms with optional χ² band shaded on the axis."""
    _theme_axes(ax)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("trials", fontsize=9)

    # Shared bin edges so the two distributions are directly comparable
    finite = np.concatenate([vals_cr3bp[np.isfinite(vals_cr3bp)],
                             vals_spice[np.isfinite(vals_spice)]])
    if finite.size == 0:
        ax.text(0.5, 0.5, "no data", color=_TEXT, ha="center", va="center",
                transform=ax.transAxes)
        return
    bins = np.linspace(finite.min(), finite.max(), 22)

    if band_lb is not None and band_ub is not None:
        ax.axvspan(band_lb, band_ub, color=_GREEN, alpha=0.10, zorder=0,
                   label=band_label)

    ax.hist(vals_cr3bp, bins=bins, color=_CYAN,   alpha=0.55,
            label=f"CR3BP (n={vals_cr3bp.size})", edgecolor=_BG, linewidth=0.4)
    ax.hist(vals_spice, bins=bins, color=_AMBER,  alpha=0.55,
            label=f"SPICE (n={vals_spice.size})", edgecolor=_BG, linewidth=0.4)

    # Median markers
    if vals_cr3bp.size:
        ax.axvline(np.median(vals_cr3bp), color=_CYAN,  lw=1.4, ls="--", alpha=0.95)
    if vals_spice.size:
        ax.axvline(np.median(vals_spice), color=_AMBER, lw=1.4, ls="--", alpha=0.95)

    ax.legend(fontsize=7.5, facecolor=_PANEL, edgecolor=_BORDER,
              labelcolor=_TEXT, loc="upper right")


def _render_figure(s_cr3bp: dict, s_spice: dict, out_path: Path,
                   *, miss_unit_cr3bp: str, miss_unit_spice: str) -> None:
    a_c = s_cr3bp["_arrays"]
    a_s = s_spice["_arrays"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=_BG,
                             layout="constrained")

    _hist_pair(
        axes[0, 0], a_c["nees"], a_s["nees"],
        title="NEES (per-trial mean) — model-mismatch signature",
        xlabel="NEES",
        band_lb=_LB6, band_ub=_UB6,
        band_label=f"χ²(6) 95% band  [{_LB6:.2f}, {_UB6:.2f}]",
    )
    _hist_pair(
        axes[0, 1], a_c["nis"], a_s["nis"],
        title="NIS (per-trial mean) — innovation consistency",
        xlabel="NIS",
        band_lb=_LB2, band_ub=_UB2,
        band_label=f"χ²(2) 95% band  [{_LB2:.2f}, {_UB2:.2f}]",
    )
    _hist_pair(
        axes[1, 0], a_c["miss_km"], a_s["miss_km"],
        title="Final miss to target — unit-aligned [km]",
        xlabel="miss_ekf   [km]   (CR3BP scaled by 1 LU = 384,400 km)",
    )
    _hist_pair(
        axes[1, 1], a_c["dv_delta"], a_s["dv_delta"],
        title="‖Δv_EKF − Δv_perfect‖   (different units)",
        xlabel=f"|Δv_EKF − Δv_perfect|   [CR3BP: ND, SPICE: km/s]",
    )

    fig.suptitle(
        f"Phase B Validation — CR3BP vs SPICE   "
        f"(n_cr3bp={s_cr3bp['n']}, n_spice={s_spice['n']})",
        color=_TEXT, fontsize=13, fontweight="bold",
    )
    fig.savefig(out_path, dpi=180, facecolor=_BG)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cr3bp-csv", type=Path, required=True,
                   help="Path to CR3BP-truth MC results CSV")
    p.add_argument("--spice-csv", type=Path, required=True,
                   help="Path to SPICE-truth MC results CSV")
    p.add_argument("--out-dir",   type=Path, required=True,
                   help="Where to write comparison report (.txt + .png)")
    p.add_argument("--miss-unit-cr3bp", default="ND",
                   help="Units label for CR3BP miss values (default ND).")
    p.add_argument("--miss-unit-spice", default="km",
                   help="Units label for SPICE miss values (default km).")
    return p.parse_args()


def main() -> None:
    _apply_dark_theme()
    args = parse_args()

    cr3bp_rows = _read_csv(repo_path(args.cr3bp_csv))
    spice_rows = _read_csv(repo_path(args.spice_csv))

    # Sanity-check the truth_mode column matches the file the user supplied.
    def _expected(rows: Sequence[dict], expected: str, label: str) -> None:
        observed = {r.get("truth_mode", "missing") for r in rows}
        if observed != {expected}:
            print(f"⚠ {label} CSV has truth_mode={observed} (expected {expected!r})")

    _expected(cr3bp_rows, "cr3bp", "cr3bp")
    _expected(spice_rows, "spice", "spice")

    s_cr3bp = _summary(cr3bp_rows, "cr3bp")
    s_spice = _summary(spice_rows, "spice")

    table = _print_table(s_cr3bp, s_spice,
                         miss_unit_cr3bp=args.miss_unit_cr3bp,
                         miss_unit_spice=args.miss_unit_spice)
    print(table)

    out_dir = repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / "06f_phase_b_summary.txt"
    txt_path.write_text(table + "\n")
    print(f"\nWrote: {txt_path}")

    fig_path = out_dir / "06f_phase_b_distributions.png"
    _render_figure(s_cr3bp, s_spice, fig_path,
                   miss_unit_cr3bp=args.miss_unit_cr3bp,
                   miss_unit_spice=args.miss_unit_spice)
    print(f"Wrote: {fig_path}")


if __name__ == "__main__":
    main()
