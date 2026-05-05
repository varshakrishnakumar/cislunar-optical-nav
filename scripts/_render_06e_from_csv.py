"""Re-render 06e_sweep_summary.png from the cached CSV (no MC re-run).

Used to iterate on the plot layout without paying for fresh MC.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

from scipy.stats import chi2

from _common import repo_path


def _load_make_plots():
    spec = importlib.util.spec_from_file_location(
        "fine_tune_06e", repo_path("scripts/06e_fine_tune.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fine_tune_06e"] = mod
    spec.loader.exec_module(mod)
    return mod._make_plots, mod._apply_dark_theme


def _read_rows(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: float(v) if v not in ("", None) else float("nan")
                         for k, v in r.items()})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/mc/fine_tune/06e_fine_tune.csv")
    p.add_argument("--plots-dir", default="results/mc/fine_tune")
    p.add_argument("--tol", type=float, default=1e-3)
    args = p.parse_args()

    make_plots, apply_theme = _load_make_plots()
    apply_theme()

    rows = _read_rows(repo_path(args.csv))
    lb6, ub6 = float(chi2.ppf(0.025, 6)), float(chi2.ppf(0.975, 6))

    honest = [
        r for r in rows
        if r["nees_in_band_frac"] >= 0.90
        and lb6 <= r["nees_median"] <= ub6
    ]
    if honest:
        best_pr = max(r["pass_rate"] for r in honest)
        leaders = [r for r in honest if r["pass_rate"] >= best_pr - 0.005]
        best_band = max(r["nees_in_band_frac"] for r in leaders)
        equally_calibrated = [r for r in leaders
                              if r["nees_in_band_frac"] >= best_band - 0.005]
        best_honest = min(equally_calibrated, key=lambda r: r["q_acc"])
    else:
        best_honest = None

    out = repo_path(args.plots_dir)
    out.mkdir(parents=True, exist_ok=True)
    make_plots(rows, out, tol=args.tol, band=(lb6, ub6), best_honest=best_honest)


if __name__ == "__main__":
    main()
