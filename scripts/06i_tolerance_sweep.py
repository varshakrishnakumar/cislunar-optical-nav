"""Phase 1 — terminal-tolerance sensitivity.

Sweeps success rate over a range of miss tolerances (390 km → 7.8 km in
ND, plus the 1 km / 100 m headline). Operates on existing Monte-Carlo
result CSVs; rescales miss values from native units to km using the
SPICE epoch length unit so CR3BP-ND and SPICE-km runs sit on the same
axis.

Usage
-----
python scripts/06i_tolerance_sweep.py \
    --csv results/mc/phase_d_production/06c_baseline_results.csv \
    --csv results/mc/phase_d_production_spice/06c_baseline_results.csv \
    --label "CR3BP truth" --label "SPICE truth" \
    --out  results/mc/phase_d_tolerance_sweep
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Sequence

from _analysis_common import apply_dark_theme  # noqa: E402

import numpy as np
import matplotlib.pyplot as plt

from _common import ensure_src_on_path, repo_path

ensure_src_on_path()
from utils.units import RunUnits  # noqa: E402

_LU_KM = 384_400.0


def _load_miss_km(csv_path: Path) -> tuple[np.ndarray, str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    miss_vals: list[float] = []
    truth_modes: set[str] = set()
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                v = float(row.get("miss_ekf", "nan"))
            except (TypeError, ValueError):
                v = float("nan")
            miss_vals.append(v)
            tm = row.get("truth_mode", "cr3bp")
            truth_modes.add(tm or "cr3bp")
    if not miss_vals:
        raise ValueError(f"No rows in {csv_path}")
    arr = np.asarray(miss_vals, dtype=float)
    arr = arr[np.isfinite(arr)]

    # Heuristic: if median miss is < 1.0 we're in CR3BP-ND units; rescale.
    truth = next(iter(truth_modes)) if len(truth_modes) == 1 else "cr3bp"
    units = RunUnits.for_truth(truth)
    if units.truth == "cr3bp":
        miss_km = arr * _LU_KM
    else:
        miss_km = arr  # already in km
    return miss_km, units.truth


_DEFAULT_TOL_KM = np.array([
    1.0, 5.0, 10.0, 25.0, 39.0, 100.0, 195.0, 390.0, 1000.0
])


def _success_rate(miss_km: np.ndarray, tol_km: float) -> float:
    if miss_km.size == 0:
        return float("nan")
    return float(np.mean(miss_km < tol_km))


def _plot_tolerance_sweep(
    miss_km_list: Sequence[np.ndarray],
    labels: Sequence[str],
    tol_km: np.ndarray,
    outpath: Path,
) -> None:
    apply_dark_theme()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("#080B14")
    ax.set_facecolor("#0D1117")
    ax.grid(True, color="#1C2340", lw=0.4)

    colors = ["#22D3EE", "#A78BFA", "#F59E0B", "#10B981", "#EC4899"]
    for i, (miss_km, lab) in enumerate(zip(miss_km_list, labels)):
        srs = np.array([_success_rate(miss_km, t) for t in tol_km])
        ax.semilogx(tol_km, srs * 100.0, "-o",
                    color=colors[i % len(colors)], lw=2.0, ms=6,
                    label=f"{lab}  (n={miss_km.size})")

    # Reference markers — current paper threshold and the new tighter one.
    ax.axvline(390.0, color="#10B981", lw=0.8, ls=":", alpha=0.8,
               label="paper tol  (1e-3 LU = 390 km)")
    ax.axvline(39.0, color="#F43F5E", lw=0.8, ls=":", alpha=0.8,
               label="tight tol   (1e-4 LU = 39 km)")
    ax.set_xlabel("miss tolerance  [km]", color="#DCE0EC")
    ax.set_ylabel("success rate  [%]", color="#DCE0EC")
    ax.set_ylim(-2, 102)
    ax.set_title(
        "Terminal-Tolerance Sensitivity  ·  P(miss < tol)",
        color="#DCE0EC", fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, facecolor="#080B14")
    plt.close(fig)


def _summarize(
    miss_km_list: Sequence[np.ndarray],
    labels: Sequence[str],
    tol_km: np.ndarray,
    out_txt: Path,
) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("Tolerance sweep — success rate P(miss < tol)")
    lines.append("=" * 64)
    header = f"{'tol [km]':>10} | " + " | ".join(f"{lab[:18]:>18}" for lab in labels)
    lines.append(header)
    lines.append("-" * len(header))
    for t in tol_km:
        cells = [f"{_success_rate(m, t) * 100.0:17.1f}%" for m in miss_km_list]
        lines.append(f"{t:10.1f} | " + " | ".join(cells))
    lines.append("")
    for m, lab in zip(miss_km_list, labels):
        lines.append(
            f"{lab}: n={m.size}  median={np.median(m):.1f} km  "
            f"p95={np.percentile(m, 95):.1f} km  "
            f"max={m.max():.1f} km"
        )
    out_txt.write_text("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Terminal-tolerance sweep across MC result CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", action="append", required=True,
                   help="Repeatable. Path to a 06c_*_results.csv file.")
    p.add_argument("--label", action="append", default=None,
                   help="Repeatable. Per-CSV label. Defaults to CSV stem.")
    p.add_argument("--out", type=str, default="results/mc/tolerance_sweep",
                   help="Output directory for figure + summary.")
    p.add_argument("--tol-km", type=float, nargs="+", default=None,
                   help="Override default tolerance grid (km).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    csvs = [repo_path(c) for c in args.csv]
    labels = list(args.label) if args.label else [c.stem for c in csvs]
    if len(labels) != len(csvs):
        raise SystemExit(
            f"--label given {len(labels)} times but --csv given {len(csvs)} times"
        )
    tol_km = (
        np.asarray(args.tol_km, dtype=float)
        if args.tol_km is not None
        else _DEFAULT_TOL_KM
    )

    miss_lists: list[np.ndarray] = []
    truth_modes: list[str] = []
    for c in csvs:
        miss_km, truth = _load_miss_km(c)
        miss_lists.append(miss_km)
        truth_modes.append(truth)

    out_dir = repo_path(args.out)
    _plot_tolerance_sweep(miss_lists, labels, tol_km,
                          out_dir / "06i_tolerance_sweep.png")
    _summarize(miss_lists, labels, tol_km,
               out_dir / "06i_tolerance_sweep.txt")
    print("Wrote:")
    print(f"  {out_dir / '06i_tolerance_sweep.png'}")
    print(f"  {out_dir / '06i_tolerance_sweep.txt'}")
    for lab, t in zip(labels, truth_modes):
        print(f"  ({lab}: truth={t})")


if __name__ == "__main__":
    main()
