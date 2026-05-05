"""Print the slide-13 fault scenarios table from cached EKF diagnostics.

Source of truth: results/diagnostics/06_ekf/<case>/summary.json, written by
scripts/06_ekf_diagnostics.py. Re-run that script to refresh the cache.

Usage:
    python scripts/_slide13_fault_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

from _common import repo_path


CASES = [
    ("estimate_tracking_baseline",   "Nominal"),
    ("estimate_tracking_dropout",    "5% dropout"),
    ("estimate_tracking_outliers",   "Pixel outliers (3% × 12σ)"),
    ("estimate_tracking_loose_debug","Loose χ² gate (95%)"),
    ("estimate_tracking_delay",      "1-step meas delay"),
]


def main() -> None:
    root = repo_path("results/diagnostics/06_ekf")

    print(f"\n{'SCENARIO':<28} {'VALID':>7} {'NIS_mean':>9} {'‖r‖_final':>12} {'OK':>5}")
    print("-" * 64)

    for case, label in CASES:
        path: Path = root / case / "summary.json"
        if not path.exists():
            print(f"{label:<28} (missing — re-run scripts/06_ekf_diagnostics.py)")
            continue
        with path.open() as f:
            s = json.load(f)["summary"]

        valid    = float(s["valid_rate"])
        nis_mean = float(s["nis_mean"])
        r_final  = float(s["final_pos_err"])

        # Pass = NIS within χ²(2) 95% band [0.05, 7.38] AND r_final < 1e-2
        ok = (0.05 <= nis_mean <= 7.38) and (r_final < 1e-2)
        flag = "✓" if ok else "✗"

        print(f"{label:<28} {valid:>7.2f} {nis_mean:>9.2f} {r_final:>12.2e} {flag:>5}")

    print("\nSource: scripts/06_ekf_diagnostics.py → results/diagnostics/06_ekf/")


if __name__ == "__main__":
    main()
