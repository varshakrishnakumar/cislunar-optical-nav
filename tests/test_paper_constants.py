"""Guard the paper-facing constants against accidental drift.

Every km-valued table in the manuscript depends on KM_PER_LU. Changing it
re-prices every report number, so we lock the value and the q_a operating
point with a regression test.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _paper_constants import (
    KM_PER_LU,
    Q_ACC_CANONICAL,
    THRESHOLD_PRECISION_KM,
    THRESHOLD_SCREENING_KM,
    THRESHOLD_TIGHT_KM,
)


def test_km_per_lu_is_jpl_catalog_value():
    # JPL periodic-orbits Earth-Moon catalog characteristic length, in km,
    # for mu = 1.215058560962404e-2. This is the value reported in the
    # Mission Scenario section of the manuscript.
    assert KM_PER_LU == 389703.2648


def test_q_acc_canonical_is_realism_operating_point():
    # Adopted for n=1000 production runs (Table 3, Tables 9/13, Figure 12).
    # The earlier q_a=1e-9 tuning-sweep recommendation was superseded;
    # see Section 11.2 of the manuscript for the justification.
    assert Q_ACC_CANONICAL == 1.0e-14


def test_threshold_screening_is_one_milli_lu():
    # The paper's 10^-3 LU screening tolerance, in km.
    assert abs(THRESHOLD_SCREENING_KM - 1e-3 * KM_PER_LU) < 1.0


def test_threshold_precision_is_one_tenth_milli_lu():
    # The 10^-4 LU precision-arrival reality-check tolerance, in km.
    assert abs(THRESHOLD_PRECISION_KM - 1e-4 * KM_PER_LU) < 1.0


def test_threshold_tight_is_below_precision():
    assert THRESHOLD_TIGHT_KM < THRESHOLD_PRECISION_KM < THRESHOLD_SCREENING_KM
