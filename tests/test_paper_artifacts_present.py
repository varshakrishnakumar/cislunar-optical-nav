"""Verify the paper_artifacts/ canonical bundle is present and self-consistent.

This is a reviewer-facing guard: a fresh clone of the repository should
be able to regenerate the central figure without re-running any Monte
Carlo. If this test fails, the repository has drifted from journal-ready
state.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "paper_artifacts"

# Per `paper_artifacts/README.md` — the five canonical CSVs that back
# every km-valued table and figure in the journal manuscript.
CANONICAL_CSVS = (
    "phase_d_production__06c_baseline_results.csv",
    "phase_d_production_spice__06c_baseline_results.csv",
    "phase_f_landmarks_pointing__06r_landmarks_under_pointing_degradation.csv",
    "phase_g_estimator_ablation__06s_estimator_ablation.csv",
    "phase_h_central_ecdf__06t_success_ecdf_central.csv",
)


@pytest.mark.parametrize("name", CANONICAL_CSVS)
def test_csv_exists(name):
    p = ARTIFACTS / "csv" / name
    assert p.exists(), f"missing canonical CSV: {p}"
    # Reject empty-or-near-empty files (i.e. trivial regressions).
    assert p.stat().st_size > 1024, f"CSV too small to be production data: {p}"


def test_baseline_median_matches_manuscript():
    # 61.64 km should match Table 3 (median terminal miss [km]) exactly.
    p = ARTIFACTS / "csv" / "phase_d_production__06c_baseline_results.csv"
    rows = list(csv.DictReader(p.open()))
    miss_lu = np.array([float(r["miss_ekf"]) for r in rows], dtype=float)
    KM_PER_LU = 389703.2648
    median_km = float(np.median(miss_lu)) * KM_PER_LU
    # Allow a 0.1 km tolerance for floating-point representation.
    assert abs(median_km - 61.64) < 0.1, \
        f"baseline median {median_km:.3f} km drifted from manuscript value 61.64 km"


def test_landmark_grid_has_15_cells():
    # 06r runs a 3×5 matrix; each cell is 1000 rows at production.
    p = ARTIFACTS / "csv" / "phase_f_landmarks_pointing__06r_landmarks_under_pointing_degradation.csv"
    rows = list(csv.DictReader(p.open()))
    cells = {(r["lm_config"], r["pt_mode"]) for r in rows}
    assert len(cells) == 15, f"expected 15 (lm_config, pt_mode) cells, got {len(cells)}"
    assert len(rows) == 15_000, f"expected 15000 production trials, got {len(rows)}"


def test_estimator_ablation_has_3_filters():
    p = ARTIFACTS / "csv" / "phase_g_estimator_ablation__06s_estimator_ablation.csv"
    rows = list(csv.DictReader(p.open()))
    filters = {r["filter_kind"] for r in rows}
    assert filters == {"ekf", "iekf", "ukf"}, filters
    assert len(rows) == 3000
