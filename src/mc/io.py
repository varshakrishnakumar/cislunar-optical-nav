from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import List

from .types import TrialResult


def save_results_csv(results: List[TrialResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        path.write_text("")
        return path

    rows = [asdict(r) for r in results]
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return path


def load_results_csv(path: str | Path) -> List[TrialResult]:
    path = Path(path)
    if not path.exists():
        return []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        out: List[TrialResult] = []
        for row in reader:
            out.append(
                TrialResult(
                    trial_id=int(row["trial_id"]),
                    seed=int(row["seed"]),
                    tc=float(row["tc"]),
                    sigma_px=float(row["sigma_px"]),
                    dropout_prob=float(row["dropout_prob"]),
                    camera_mode=str(row["camera_mode"]),
                    dv_perfect_mag=float(row["dv_perfect_mag"]),
                    dv_ekf_mag=float(row["dv_ekf_mag"]),
                    dv_delta_mag=float(row["dv_delta_mag"]),
                    dv_inflation=float(row["dv_inflation"]),
                    dv_inflation_pct=float(row["dv_inflation_pct"]),
                    miss_uncorrected=float(row["miss_uncorrected"]),
                    miss_perfect=float(row["miss_perfect"]),
                    miss_ekf=float(row["miss_ekf"]),
                    pos_err_tc=float(row["pos_err_tc"]),
                    tracePpos_tc=float(row["tracePpos_tc"]),
                    nis_mean=float(row["nis_mean"]),
                    valid_rate=float(row["valid_rate"]),
                    dx0_norm_r=float(row["dx0_norm_r"]),
                    dx0_norm_v=float(row["dx0_norm_v"]),
                    notes=row.get("notes") or None,
                )
            )
    return out
