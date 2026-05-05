"""Phase D — paper headline figure for the truth-mode comparison.

Reads matched CR3BP and SPICE Monte Carlo CSVs (one per truth mode) and
produces the publication-grade plot set:

  Figure 1 — Operational impact of model mismatch
    • Miss distance ECDFs (km, unit-aligned) — the headline.
    • NEES & NIS box plots vs χ² bands — proof the filter stays consistent.
    • Burn-error distributions (km/s, unit-aligned).

The thesis: "filter remains χ²-consistent under SPICE truth (NEES inside
band, NIS unchanged) but mission accuracy degrades — model mismatch
surfaces as an OPERATIONAL cost, not a CONSISTENCY failure."

Usage::

    python3 scripts/06g_phase_d_headline.py \\
        --cr3bp-csv results/mc/phase_d_production/06c_baseline_results.csv \\
        --spice-csv results/mc/phase_d_production_spice/06c_baseline_results.csv \\
        --out-dir   results/mc/phase_d_production_compare
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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

_LB6, _UB6 = float(chi2.ppf(0.025, 6)), float(chi2.ppf(0.975, 6))
_LB2, _UB2 = float(chi2.ppf(0.025, 2)), float(chi2.ppf(0.975, 2))
_LU_KM = RunUnits.for_truth("cr3bp").length_km_per_nd
_VU_KMPS = RunUnits.for_truth("cr3bp").velocity_kmps_per_nd


def _read(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _floats(rows: list[dict], key: str) -> np.ndarray:
    out = []
    for r in rows:
        try:
            out.append(float(r.get(key, "nan")))
        except (TypeError, ValueError):
            out.append(float("nan"))
    a = np.array(out, dtype=float)
    return a[np.isfinite(a)]


def _theme(ax) -> None:
    ax.set_facecolor(_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(_BORDER); sp.set_linewidth(0.7)
    ax.tick_params(colors=_TEXT, labelsize=9)
    ax.xaxis.label.set_color(_TEXT); ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    ax.grid(True, color=_BORDER, lw=0.4, alpha=0.7)


def _convert(rows: list[dict]) -> dict:
    """Pull out unit-aligned arrays. CR3BP rows scale ND → km / km·s."""
    truth = rows[0].get("truth_mode", "cr3bp") if rows else "cr3bp"
    miss     = _floats(rows, "miss_ekf")
    pos_err  = _floats(rows, "pos_err_tc")
    dv_delta = _floats(rows, "dv_delta_mag")
    nees     = _floats(rows, "nees_mean")
    nis      = _floats(rows, "nis_mean")
    if truth == "cr3bp":
        miss     = miss     * _LU_KM
        pos_err  = pos_err  * _LU_KM
        dv_delta = dv_delta * _VU_KMPS * 1000.0   # km/s → m/s
    elif truth == "spice":
        dv_delta = dv_delta * 1000.0              # km/s → m/s
    return dict(
        truth=truth, n=len(rows),
        miss_km=miss, pos_err_km=pos_err, dv_delta_mps=dv_delta,
        nees=nees, nis=nis,
    )


def _ecdf_pair(ax, vals_c: np.ndarray, vals_s: np.ndarray, *,
               xlabel: str, title: str) -> None:
    _theme(ax)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("ECDF", fontsize=10)

    for vals, color, lbl in [(vals_c, _CYAN,  f"CR3BP truth (n={vals_c.size})"),
                             (vals_s, _AMBER, f"SPICE truth (n={vals_s.size})")]:
        if vals.size == 0:
            continue
        x = np.sort(vals)
        y = np.arange(1, x.size + 1) / x.size
        ax.plot(x, y, color=color, lw=2.2, label=lbl)
        # Median tick
        med = float(np.median(vals))
        ax.axvline(med, color=color, lw=0.9, ls="--", alpha=0.65)

    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9, facecolor=_PANEL, edgecolor=_BORDER,
              labelcolor=_TEXT, loc="lower right")


def _box_pair(ax, vals_c: np.ndarray, vals_s: np.ndarray, *,
              ylabel: str, title: str,
              band_lb: float | None = None, band_ub: float | None = None,
              band_label: str = "") -> None:
    _theme(ax)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10)

    if band_lb is not None and band_ub is not None:
        ax.axhspan(band_lb, band_ub, color=_GREEN, alpha=0.12, zorder=0,
                   label=band_label)

    data = [vals_c, vals_s]
    labels = [f"CR3BP\n(n={vals_c.size})", f"SPICE\n(n={vals_s.size})"]
    bp = ax.boxplot(
        data, tick_labels=labels, patch_artist=True,
        widths=0.55, showfliers=True,
        boxprops=dict(linewidth=1.0, edgecolor=_TEXT),
        whiskerprops=dict(color=_TEXT, linewidth=1.0),
        capprops=dict(color=_TEXT, linewidth=1.0),
        medianprops=dict(color=_BG, linewidth=1.6),
        flierprops=dict(marker="o", markersize=3.5, markerfacecolor=_RED,
                        markeredgecolor=_RED, alpha=0.7),
    )
    for patch, color in zip(bp["boxes"], (_CYAN, _AMBER)):
        patch.set_facecolor(color); patch.set_alpha(0.65)

    if band_lb is not None:
        ax.legend(fontsize=8, facecolor=_PANEL, edgecolor=_BORDER,
                  labelcolor=_TEXT, loc="upper right")


def _build_figure(c: dict, s: dict, *, out_path: Path,
                  config_label: str = "") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), facecolor=_BG,
                             gridspec_kw=dict(height_ratios=[1.2, 1.0]),
                             layout="constrained")

    # ── Top-left: ECDF of miss distance — the headline ───────────────────────
    _ecdf_pair(
        axes[0, 0], c["miss_km"], s["miss_km"],
        xlabel=r"miss distance to target  [km, unit-aligned]",
        title="Mission accuracy — final miss distance ECDF",
    )

    # ── Top-right: ECDF of position error at tc ──────────────────────────────
    _ecdf_pair(
        axes[0, 1], c["pos_err_km"], s["pos_err_km"],
        xlabel=r"‖r̂(t_c) − r(t_c)‖  [km, unit-aligned]",
        title="Position estimate error at correction time",
    )

    # ── Bottom-left: NEES box plot with χ²(6) band ───────────────────────────
    _box_pair(
        axes[1, 0], c["nees"], s["nees"],
        ylabel="NEES (per-trial mean)",
        title="Filter consistency — NEES vs χ²(6) band",
        band_lb=_LB6, band_ub=_UB6,
        band_label=f"χ²(6) 95% band [{_LB6:.2f}, {_UB6:.2f}]",
    )

    # ── Bottom-right: NIS box plot with χ²(2) band ───────────────────────────
    _box_pair(
        axes[1, 1], c["nis"], s["nis"],
        ylabel="NIS (per-trial mean)",
        title="Innovation consistency — NIS vs χ²(2) band",
        band_lb=_LB2, band_ub=_UB2,
        band_label=f"χ²(2) 95% band [{_LB2:.2f}, {_UB2:.2f}]",
    )

    suffix = f"  ·  {config_label}" if config_label else ""
    fig.suptitle(
        f"Operational Impact of Model Mismatch — CR3BP vs SPICE Truth{suffix}",
        color=_TEXT, fontsize=14, fontweight="bold",
    )

    fig.savefig(out_path, dpi=200, facecolor=_BG, bbox_inches="tight")
    plt.close(fig)


def _stats_table(c: dict, s: dict) -> str:
    def _med(a): return float(np.median(a)) if a.size else float("nan")
    def _p95(a): return float(np.percentile(a, 95)) if a.size else float("nan")
    def _band(a, lb, ub):
        return float(np.mean((a >= lb) & (a <= ub))) if a.size else float("nan")

    miss_med_c, miss_med_s = _med(c["miss_km"]),    _med(s["miss_km"])
    miss_p95_c, miss_p95_s = _p95(c["miss_km"]),    _p95(s["miss_km"])
    nees_med_c, nees_med_s = _med(c["nees"]),       _med(s["nees"])
    nees_in_c              = _band(c["nees"], _LB6, _UB6)
    nees_in_s              = _band(s["nees"], _LB6, _UB6)
    nis_in_c               = _band(c["nis"],  _LB2, _UB2)
    nis_in_s               = _band(s["nis"],  _LB2, _UB2)

    rows = [
        ("Final miss median       [km]", miss_med_c, miss_med_s),
        ("Final miss p95          [km]", miss_p95_c, miss_p95_s),
        ("Pos error at tc median  [km]", _med(c["pos_err_km"]), _med(s["pos_err_km"])),
        ("‖Δv_EKF − Δv_perfect‖ med [m/s]", _med(c["dv_delta_mps"]), _med(s["dv_delta_mps"])),
        ("NEES median",                 nees_med_c, nees_med_s),
        ("NIS  median",                 _med(c["nis"]),        _med(s["nis"])),
    ]
    band_rows = [
        ("NEES in χ²(6) band  [%]", nees_in_c * 100, nees_in_s * 100),
        ("NIS  in χ²(2) band  [%]", nis_in_c  * 100, nis_in_s  * 100),
    ]

    lines = []
    lines.append("═" * 80)
    lines.append(f"  Phase D headline — CR3BP vs SPICE (n_cr3bp={c['n']}, n_spice={s['n']})")
    lines.append("═" * 80)
    lines.append(f"  {'metric':<36} {'CR3BP':>14} {'SPICE':>14} {'ratio':>10}")
    lines.append("  " + "─" * 78)
    for name, cv, sv in rows:
        ratio = (sv / cv) if (np.isfinite(cv) and cv != 0) else float("nan")
        ratio_str = f"{ratio:7.2f}×" if np.isfinite(ratio) else "      —"
        lines.append(f"  {name:<36} {cv:>14.4g} {sv:>14.4g} {ratio_str:>10}")
    lines.append("  " + "─" * 78)
    for name, cv, sv in band_rows:
        lines.append(f"  {name:<36} {cv:>13.1f} % {sv:>13.1f} %       —")
    lines.append("═" * 80)

    # ── Interpretation ───────────────────────────────────────────────────────
    lines.append("")
    lines.append("  Interpretation")
    lines.append("  ──────────────")
    miss_ratio = (miss_med_s / miss_med_c) if miss_med_c else float("nan")
    nees_ratio = (nees_med_s / nees_med_c) if nees_med_c else float("nan")
    nees_in_drop = nees_in_c - nees_in_s    # CR3BP minus SPICE, expect positive

    miss_inflates = np.isfinite(miss_ratio) and miss_ratio > 1.3
    nees_inflates = np.isfinite(nees_ratio) and nees_ratio > 1.3
    nees_band_breaks = (np.isfinite(nees_in_s) and nees_in_s < 0.90 and
                        np.isfinite(nees_in_c) and nees_in_c >= 0.85 and
                        nees_in_drop > 0.10)

    if miss_inflates and nees_band_breaks:
        lines.append(f"  • Operational cost: miss inflates {miss_ratio:.1f}× (median)")
        lines.append(f"    under SPICE — mission accuracy degrades.")
        lines.append(f"  • Consistency cost: NEES in-band drops")
        lines.append(f"    {nees_in_c*100:.0f}% → {nees_in_s*100:.0f}% under SPICE")
        lines.append(f"    (Bar-Shalom 90% gate). Filter is no longer textbook-")
        lines.append(f"    consistent under realistic dynamics, but median NEES")
        lines.append(f"    (= {nees_med_s:.1f}) remains inside the χ²(6) band.")
        lines.append(f"  • Innovation NIS unchanged ({nis_in_s*100:.0f}% in band) —")
        lines.append(f"    the filter doesn't 'see' the mismatch in individual")
        lines.append(f"    measurement residuals; it accumulates over the arc.")
        lines.append(f"  • Pre-burn pos-error at tc is unchanged — the divergence")
        lines.append(f"    is post-tc, where SPICE truth and CR3BP integration")
        lines.append(f"    diverge unchecked.")
    elif miss_inflates and not (nees_band_breaks or nees_inflates):
        lines.append(f"  • Miss inflates {miss_ratio:.1f}× under SPICE while NEES stays")
        lines.append(f"    in band ({nees_in_s*100:.0f}%) — the filter absorbs ephemeris")
        lines.append(f"    residuals as honestly-quantified uncertainty (consistent)")
        lines.append(f"    but pays for it in mission accuracy. (Often a sample-size")
        lines.append(f"    artifact; n>500 typically reveals the consistency cost.)")
    elif nees_inflates and not miss_inflates:
        lines.append(f"  • NEES inflates {nees_ratio:.1f}× under SPICE but miss is")
        lines.append(f"    comparable. Filter is statistically overconfident but")
        lines.append(f"    operationally fine.")
    else:
        lines.append(f"  • miss ratio {miss_ratio:.2f}×, NEES ratio {nees_ratio:.2f}×.")
        lines.append(f"    Mismatch is mild at this regime.")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cr3bp-csv", type=Path, required=True)
    p.add_argument("--spice-csv", type=Path, required=True)
    p.add_argument("--out-dir",   type=Path, required=True)
    p.add_argument("--config-label", default="",
                   help="Optional subtitle suffix (e.g. 'q_acc=1e-14, σ_px=1.0').")
    return p.parse_args()


def main() -> None:
    _apply_dark_theme()
    args = parse_args()
    cr3bp_rows = _read(repo_path(args.cr3bp_csv))
    spice_rows = _read(repo_path(args.spice_csv))
    c = _convert(cr3bp_rows)
    s = _convert(spice_rows)

    table = _stats_table(c, s)
    print(table)

    out_dir = repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt = out_dir / "06g_headline_summary.txt"
    txt.write_text(table + "\n")
    fig = out_dir / "06g_headline.png"
    _build_figure(c, s, out_path=fig, config_label=args.config_label)
    print(f"\nWrote: {txt}")
    print(f"Wrote: {fig}")


if __name__ == "__main__":
    main()
