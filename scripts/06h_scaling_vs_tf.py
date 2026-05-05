"""Scaling of miss-inflation with arc length.

Reads matched CR3BP/SPICE MC CSVs across multiple tf values and plots
how the model-mismatch operational cost grows with arc length. Phase B+
supporting figure for the paper: complements 06g's headline by showing
that the 2.1× headline number isn't a regime-specific accident — it
grows monotonically with arc length while filter consistency stays
within band the whole time.

Usage::

    python3 scripts/06h_scaling_vs_tf.py \\
        --pair tf=4:results/mc/phase_b_tf4 \\
        --pair tf=6:results/mc/phase_b_baseline \\
        --pair tf=8:results/mc/phase_b_tf8 \\
        --pair tf=10:results/mc/phase_b_tf10 \\
        --out-dir results/mc/phase_b_scaling

Each ``--pair label:dir`` references a directory that contains *both*
the CR3BP CSV (e.g. ``dir/06c_baseline_results.csv``) and the SPICE
CSV under the auto-suffixed sibling (``dir_spice/06c_baseline_results.csv``).
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
_LU_KM = RunUnits.for_truth("cr3bp").length_km_per_nd


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
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


def _ingest(label: str, base_dir: Path) -> dict | None:
    cr3bp_path = base_dir / "06c_baseline_results.csv"
    spice_path = base_dir.with_name(base_dir.name + "_spice") / "06c_baseline_results.csv"
    cr3bp_rows = _read(cr3bp_path)
    spice_rows = _read(spice_path)
    if not cr3bp_rows or not spice_rows:
        print(f"⚠ {label}: missing data ({cr3bp_path.exists()=}, {spice_path.exists()=})")
        return None

    miss_c_km = _floats(cr3bp_rows, "miss_ekf") * _LU_KM
    miss_s_km = _floats(spice_rows, "miss_ekf")
    nees_c    = _floats(cr3bp_rows, "nees_mean")
    nees_s    = _floats(spice_rows, "nees_mean")

    # Try to extract tf from label (`tf=8` → 8.0); fall back to NaN.
    tf = float("nan")
    if "=" in label:
        try:
            tf = float(label.split("=", 1)[1])
        except ValueError:
            pass

    return dict(
        label=label, tf=tf,
        n_c=miss_c_km.size, n_s=miss_s_km.size,
        miss_c_med=float(np.median(miss_c_km)),
        miss_c_p95=float(np.percentile(miss_c_km, 95)),
        miss_s_med=float(np.median(miss_s_km)),
        miss_s_p95=float(np.percentile(miss_s_km, 95)),
        nees_c_med=float(np.median(nees_c)) if nees_c.size else float("nan"),
        nees_s_med=float(np.median(nees_s)) if nees_s.size else float("nan"),
        nees_c_in=float(np.mean((nees_c >= _LB6) & (nees_c <= _UB6))) if nees_c.size else float("nan"),
        nees_s_in=float(np.mean((nees_s >= _LB6) & (nees_s <= _UB6))) if nees_s.size else float("nan"),
    )


def _theme(ax) -> None:
    ax.set_facecolor(_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(_BORDER); sp.set_linewidth(0.7)
    ax.tick_params(colors=_TEXT, labelsize=9)
    ax.xaxis.label.set_color(_TEXT); ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    ax.grid(True, color=_BORDER, lw=0.4, alpha=0.7)


def _build_figure(rows: list[dict], out_path: Path) -> None:
    rows_sorted = sorted([r for r in rows if np.isfinite(r["tf"])], key=lambda r: r["tf"])
    if not rows_sorted:
        raise RuntimeError("No usable rows for scaling plot")

    tf      = np.array([r["tf"] for r in rows_sorted])
    miss_c  = np.array([r["miss_c_med"] for r in rows_sorted])
    miss_s  = np.array([r["miss_s_med"] for r in rows_sorted])
    miss_cp = np.array([r["miss_c_p95"] for r in rows_sorted])
    miss_sp = np.array([r["miss_s_p95"] for r in rows_sorted])
    ratio_med = miss_s / miss_c
    ratio_p95 = miss_sp / miss_cp
    nees_c_in = np.array([r["nees_c_in"] for r in rows_sorted]) * 100
    nees_s_in = np.array([r["nees_s_in"] for r in rows_sorted]) * 100

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor=_BG,
                             layout="constrained")

    # Panel A: absolute miss medians (km) vs tf
    ax = axes[0]; _theme(ax)
    ax.plot(tf, miss_c, "o-", color=_CYAN,  lw=2, ms=8,
            label="CR3BP truth (median)")
    ax.plot(tf, miss_s, "o-", color=_AMBER, lw=2, ms=8,
            label="SPICE truth (median)")
    ax.fill_between(tf, miss_c, miss_cp, color=_CYAN,  alpha=0.15)
    ax.fill_between(tf, miss_s, miss_sp, color=_AMBER, alpha=0.15)
    ax.set_xlabel("Arc length tf  [days]", fontsize=10)
    ax.set_ylabel("Miss distance  [km]", fontsize=10)
    ax.set_title("Operational cost grows with arc length",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, facecolor=_PANEL, edgecolor=_BORDER, labelcolor=_TEXT)

    # Panel B: ratio (SPICE / CR3BP) vs tf
    ax = axes[1]; _theme(ax)
    ax.axhline(1.0, color=_TEXT, lw=0.6, alpha=0.4)
    ax.plot(tf, ratio_med, "o-", color=_VIOLET, lw=2, ms=8,
            label="median ratio")
    ax.plot(tf, ratio_p95, "s--", color=_RED, lw=1.6, ms=7,
            label="p95 ratio")
    ax.set_xlabel("Arc length tf  [days]", fontsize=10)
    ax.set_ylabel("miss_SPICE / miss_CR3BP  [×]", fontsize=10)
    ax.set_title("Mismatch amplification",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, facecolor=_PANEL, edgecolor=_BORDER, labelcolor=_TEXT)
    ymin = min(0.9, float(np.nanmin([ratio_med.min(), ratio_p95.min()])) * 0.9)
    ymax = max(2.0, float(np.nanmax([ratio_med.max(), ratio_p95.max()])) * 1.10)
    ax.set_ylim(ymin, ymax)

    # Panel C: NEES in-band fraction vs tf
    ax = axes[2]; _theme(ax)
    ax.axhspan(90, 100, color=_GREEN, alpha=0.10, zorder=0,
               label="≥ 90% — Bar-Shalom consistent")
    ax.plot(tf, nees_c_in, "o-", color=_CYAN,  lw=2, ms=8,
            label="CR3BP truth")
    ax.plot(tf, nees_s_in, "o-", color=_AMBER, lw=2, ms=8,
            label="SPICE truth")
    ax.set_xlabel("Arc length tf  [days]", fontsize=10)
    ax.set_ylabel("NEES in χ²(6) band  [%]", fontsize=10)
    ax.set_title("Filter stays consistent at every arc length",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, facecolor=_PANEL, edgecolor=_BORDER, labelcolor=_TEXT)
    ax.set_ylim(0, 105)

    fig.suptitle(
        "Model-mismatch scaling — miss inflates while consistency holds",
        color=_TEXT, fontsize=14, fontweight="bold",
    )
    fig.savefig(out_path, dpi=200, facecolor=_BG, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", action="append", required=True,
                   help="label:base_dir; repeat for each tf point.")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    _apply_dark_theme()
    args = parse_args()

    rows: list[dict] = []
    for spec in args.pair:
        if ":" not in spec:
            print(f"⚠ skipping malformed --pair {spec!r}")
            continue
        label, base = spec.split(":", 1)
        ingest = _ingest(label, repo_path(base))
        if ingest is not None:
            rows.append(ingest)

    if not rows:
        raise SystemExit("No usable --pair inputs found")

    out_dir = repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n  tf [days]  CR3BP miss [km]    SPICE miss [km]    ratio    NEES_in  ")
    print("  ─────────  ─────────────────  ─────────────────  ───────  ─────────")
    for r in sorted(rows, key=lambda r: r["tf"]):
        ratio = r["miss_s_med"] / r["miss_c_med"] if r["miss_c_med"] else float("nan")
        print(
            f"  {r['tf']:>7.1f}  "
            f"{r['miss_c_med']:>7.1f} ± {r['miss_c_p95']:>5.0f}  "
            f"{r['miss_s_med']:>7.1f} ± {r['miss_s_p95']:>5.0f}  "
            f"{ratio:>5.2f}×  "
            f"C:{r['nees_c_in']*100:>4.0f}% S:{r['nees_s_in']*100:>4.0f}%"
        )

    fig_path = out_dir / "06h_scaling_vs_tf.png"
    _build_figure(rows, fig_path)
    print(f"\nWrote: {fig_path}")


if __name__ == "__main__":
    main()
