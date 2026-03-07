from __future__ import annotations

import numpy as np
from numpy.random import Generator
from typing import Tuple


def make_trial_rng(base_seed: int, trial_id: int) -> Generator:
    """Create a deterministic RNG per trial."""
    # Simple reproducible mix; stable across python versions.
    seed = (int(base_seed) * 1000003 + int(trial_id) * 9176) % (2**32 - 1)
    return np.random.default_rng(seed)


def sample_injection_error(
    rng: Generator,
    sigma_r: float,
    sigma_v: float,
    planar_only: bool = False,
) -> np.ndarray:
    """Sample a 6D injection error dx0 = [dr, dv]."""
    dr = rng.normal(0.0, sigma_r, size=3)
    dv = rng.normal(0.0, sigma_v, size=3)
    if planar_only:
        dr[2] = 0.0
        dv[2] = 0.0
    return np.hstack([dr, dv]).astype(float)


def sample_estimation_error(
    rng: Generator,
    sigma_r: float,
    sigma_v: float,
    planar_only: bool = False,
) -> np.ndarray:
    """Sample a 6D initial filter estimation error."""
    er = rng.normal(0.0, sigma_r, size=3)
    ev = rng.normal(0.0, sigma_v, size=3)
    if planar_only:
        er[2] = 0.0
        ev[2] = 0.0
    return np.hstack([er, ev]).astype(float)
