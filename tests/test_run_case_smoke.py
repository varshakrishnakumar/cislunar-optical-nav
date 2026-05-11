"""End-to-end smoke test for the three filter_kind branches of run_case.

Runs one deterministic-seed CR3BP trial per filter (ekf/iekf/ukf) and
checks that:
  - The trial completes without raising.
  - All three filters produce a finite terminal miss.
  - The result dict has the public fields the report leans on.
  - The default IEKF behavior is bit-stable: a re-run with the same
    seed gives an identical miss to within float tolerance.

This is intentionally CR3BP-only so the test does not require SPICE
kernels (which are not in version control).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from importlib import import_module
run_case = import_module("06_midcourse_ekf_correction").run_case


COMMON = dict(
    mu=0.0121505856, t0=0.0, tf=6.0, tc=2.0, dt_meas=0.02,
    sigma_px=1.0, dropout_prob=0.0, seed=42,
    dx0=np.array([1e-4] * 6), est_err=np.array([1e-4] * 6),
    camera_mode="estimate_tracking",
    return_debug=False, accumulate_gramian=False,
)


def _result_has_required_fields(out):
    required = {
        "miss_ekf", "pos_err_tc", "valid_rate",
        "nis_mean", "nees_mean",
        "filter_kind", "iters_used_mean",
        "t_predict_mean_us", "t_update_mean_us", "t_trial_total_s",
        "moon_offset_rad_med", "landmark_offset_rad_med",
    }
    assert required.issubset(out.keys()), \
        f"missing keys: {required - set(out.keys())}"


def test_iekf_smoke():
    out = run_case(filter_kind="iekf", **COMMON)
    _result_has_required_fields(out)
    assert np.isfinite(out["miss_ekf"]) and out["miss_ekf"] > 0
    assert out["filter_kind"] == "iekf"
    assert out["iters_used_mean"] > 1.0  # IEKF iterates


def test_ekf_smoke():
    out = run_case(filter_kind="ekf", **COMMON)
    _result_has_required_fields(out)
    assert np.isfinite(out["miss_ekf"]) and out["miss_ekf"] > 0
    assert out["filter_kind"] == "ekf"
    assert out["iters_used_mean"] == 1.0  # EKF is max_iter=1


def test_ukf_smoke():
    out = run_case(filter_kind="ukf", **COMMON)
    _result_has_required_fields(out)
    assert np.isfinite(out["miss_ekf"]) and out["miss_ekf"] > 0
    assert out["filter_kind"] == "ukf"
    # UKF should be more conservative (lower NEES) than EKF/IEKF at the
    # headline operating point; encoded here as a smoke check rather than
    # a tight tolerance.
    assert np.isfinite(out["nees_mean"])


def test_default_filter_is_iekf_and_deterministic():
    out_a = run_case(**COMMON)
    out_b = run_case(**COMMON)
    assert out_a["filter_kind"] == "iekf"
    assert out_a["miss_ekf"] == out_b["miss_ekf"]
