"""Canonical constants for paper-facing scripts.

Single source of truth for LU↔km and the canonical process-noise level
used by the n=1000 production runs that back the journal manuscript.
Every paper-facing driver (06n, 06r, 06s, 06t, ...) imports from here
so a re-render is a one-line change, and the report cannot drift from
the code.

History note (2026-05-08): An earlier generation of analysis scripts
hard-coded `_KM_PER_LU = 384_400.0` (the real-world Earth–Moon mean
separation). That value disagrees with the JPL periodic-orbits catalog
characteristic length used to seed the headline CR3BP scenario
(μ=1.215058560962404×10⁻², 1 LU = 389703.2648 km, reported in the
Mission Scenario section of the manuscript). The two differ by 1.4%
but propagate through every km-valued table in the paper, which is a
reviewer-grade inconsistency. This module fixes the LU to the
catalog value, and any new paper-facing driver should import from
here rather than redeclaring the constant locally.
"""

from __future__ import annotations


# JPL periodic-orbits Earth-Moon catalog characteristic length, in km,
# corresponding to μ = 1.215058560962404e-2. This is the value reported
# in the Mission Scenario section of the manuscript.
KM_PER_LU: float = 389_703.2648

# Earth-Moon CR3BP nondimensional time unit, in seconds.
S_PER_TU:  float = 382_981.2891

# Process-noise spectral density used for the headline n=1000 production
# baseline (Section 11.1) and for every realism extension at n=1000
# (Sections 13.x, 14.5). The earlier `q_a=1e-9` "tuned" recommendation
# from the fine sweep (Section 11.2, Table 4) was superseded for the
# production runs by `q_a=1e-14` to expose more covariance dynamics in
# the realism sweeps; the headline pass rate at the 390 km screening
# tolerance is unchanged (99.1%).
Q_ACC_CANONICAL: float = 1.0e-14


# Mission-tolerance thresholds (km) used in the success-vs-threshold
# central figure and across the paper.
THRESHOLD_SCREENING_KM:  float = 390.0   # 10⁻³ LU
THRESHOLD_PRECISION_KM:  float =  39.0   # 10⁻⁴ LU (reality check)
THRESHOLD_TIGHT_KM:      float =  25.0


__all__ = [
    "KM_PER_LU", "S_PER_TU", "Q_ACC_CANONICAL",
    "THRESHOLD_SCREENING_KM", "THRESHOLD_PRECISION_KM", "THRESHOLD_TIGHT_KM",
]
