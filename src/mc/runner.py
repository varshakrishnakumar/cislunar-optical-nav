from __future__ import annotations

from typing import Callable, List, Dict, Any
import numpy as np

from .types import MonteCarloConfig, TrialResult
from .sampler import make_trial_rng, sample_injection_error, sample_estimation_error
from .metrics import trial_result_from_run_case


CaseFn = Callable[..., Dict[str, Any]]


def run_monte_carlo(config: MonteCarloConfig, case_fn: CaseFn) -> List[TrialResult]:
    """Run Monte Carlo trials using a provided single-case function (e.g., 06A run_case).

    The case_fn is expected to accept:
      mu, t0, tf, tc, dt_meas, sigma_px, dropout_prob, seed, dx0, est_err
    and optionally:
      fixed_camera_pointing (bool)

    It must return a dict containing the scalar metrics consumed by `trial_result_from_run_case`.
    """

    results: List[TrialResult] = []

    fixed_camera_pointing = not bool(config.tracking_attitude)

    for trial_id in range(int(config.n_trials)):
        rng = make_trial_rng(config.base_seed, trial_id)
        seed = int(rng.integers(0, 2**31 - 1))

        dx0 = sample_injection_error(
            rng,
            sigma_r=float(config.sigma_r_inj),
            sigma_v=float(config.sigma_v_inj),
            planar_only=bool(config.planar_only),
        )
        est_err = sample_estimation_error(
            rng,
            sigma_r=float(config.sigma_r_est),
            sigma_v=float(config.sigma_v_est),
            planar_only=bool(config.planar_only),
        )

        # Call the single-case pipeline
        out = case_fn(
            config.mu,
            config.t0,
            config.tf,
            config.tc,
            config.dt_meas,
            config.sigma_px,
            config.dropout_prob,
            seed,
            dx0,
            est_err,
            fixed_camera_pointing=fixed_camera_pointing,
        )

        res = trial_result_from_run_case(
            trial_id=trial_id,
            seed=seed,
            tc=config.tc,
            sigma_px=config.sigma_px,
            dropout_prob=config.dropout_prob,
            tracking_attitude=config.tracking_attitude,
            dx0=dx0,
            out=out,
        )
        results.append(res)

    return results
