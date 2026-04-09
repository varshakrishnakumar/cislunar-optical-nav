from __future__ import annotations

import inspect
import traceback
import warnings
from typing import Any, Callable, Dict, List, Optional

from .metrics import trial_result_from_run_case
from .sampler import make_trial_rng, sample_estimation_error, sample_injection_error
from .types import MonteCarloConfig, TrialResult


CaseFn = Callable[..., Dict[str, Any]]


def _filter_kwargs(fn: CaseFn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    params = inspect.signature(fn).parameters
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if accepts_var_kw:
        return kwargs
    filtered = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        warnings.warn(
            f"run_monte_carlo: case_fn does not accept {dropped}; "
            "these kwargs will be dropped each trial. "
            "Update run_case() to accept them when ready.",
            UserWarning,
            stacklevel=4,
        )
    return filtered


def run_monte_carlo(
    config: MonteCarloConfig,
    case_fn: CaseFn,
    *,
    on_trial_error: str = "warn",
) -> List[TrialResult]:
    if on_trial_error not in ("warn", "raise", "skip"):
        raise ValueError(f"on_trial_error must be 'warn', 'raise', or 'skip'")

    results: List[TrialResult] = []
    n_failed = 0
    _warned_drop = False

    for trial_id in range(int(config.n_trials)):
        rng  = make_trial_rng(config.base_seed, trial_id)
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

        all_kwargs: Dict[str, Any] = dict(
            mu=config.mu,
            t0=config.t0,
            tf=config.tf,
            tc=config.tc,
            dt_meas=config.dt_meas,
            sigma_px=config.sigma_px,
            dropout_prob=config.dropout_prob,
            seed=seed,
            dx0=dx0,
            est_err=est_err,
            camera_mode=config.camera_mode,
        )

        if not _warned_drop:
            kwargs = _filter_kwargs(case_fn, all_kwargs)
            _warned_drop = True
        else:
            params = inspect.signature(case_fn).parameters
            accepts_var = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            kwargs = all_kwargs if accepts_var else {
                k: v for k, v in all_kwargs.items() if k in params
            }

        try:
            out = case_fn(**kwargs)
        except Exception as exc:
            n_failed += 1
            if on_trial_error == "raise":
                raise
            msg = (
                f"[MC trial {trial_id}] case_fn raised {type(exc).__name__}: {exc}\n"
                + traceback.format_exc()
            )
            if on_trial_error == "warn":
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            continue

        res = trial_result_from_run_case(
            trial_id=trial_id,
            seed=seed,
            tc=config.tc,
            sigma_px=config.sigma_px,
            dropout_prob=config.dropout_prob,
            camera_mode=config.camera_mode,
            dx0=dx0,
            out=out,
        )
        results.append(res)

    if n_failed > 0 and on_trial_error != "raise":
        warnings.warn(
            f"Monte Carlo run '{config.study_name}' completed with {n_failed} / "
            f"{config.n_trials} failed trials.",
            RuntimeWarning,
            stacklevel=2,
        )

    return results
