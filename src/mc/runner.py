from __future__ import annotations

import inspect
import os
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    n_workers: int = 1,
) -> List[TrialResult]:
    """Run a Monte Carlo study.

    Parameters
    ----------
    n_workers:
        Number of worker threads. ``1`` (default) runs sequentially.
        ``-1`` uses ``os.cpu_count()``. Any positive integer limits the pool
        to that many threads. Trials complete in trial-id order regardless.

        For meaningful speedup, install numba (``pip install numba``) so the
        CR3BP integrator releases the GIL during solve_ivp; otherwise thread
        overhead may exceed gains for small trial counts.
    """
    if on_trial_error not in ("warn", "raise", "skip"):
        raise ValueError(f"on_trial_error must be 'warn', 'raise', or 'skip'")

    # ------------------------------------------------------------------
    # Pre-compute every trial's kwargs in the main thread.
    # This keeps sampling deterministic and fires the "dropped kwargs"
    # warning exactly once before any parallel work begins.
    # ------------------------------------------------------------------
    trial_data: List[tuple] = []
    accepted_keys: Optional[set] = None

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

        if accepted_keys is None:
            kwargs = _filter_kwargs(case_fn, all_kwargs)   # may warn once
            accepted_keys = set(kwargs)
        else:
            kwargs = {k: v for k, v in all_kwargs.items() if k in accepted_keys}

        trial_data.append((trial_id, seed, dx0, kwargs))

    # ------------------------------------------------------------------
    # Per-trial worker — captures case_fn and config from enclosing scope.
    # ------------------------------------------------------------------
    def _run_one(trial_id: int, seed: int, dx0, kwargs: Dict[str, Any]) -> TrialResult:
        out = case_fn(**kwargs)
        return trial_result_from_run_case(
            trial_id=trial_id,
            seed=seed,
            tc=config.tc,
            sigma_px=config.sigma_px,
            dropout_prob=config.dropout_prob,
            camera_mode=config.camera_mode,
            dx0=dx0,
            out=out,
        )

    # ------------------------------------------------------------------
    # Execute — sequential or threaded.
    # ------------------------------------------------------------------
    results: List[TrialResult] = []
    failed_ids: List[int] = []

    if n_workers == 1:
        for trial_id, seed, dx0, kwargs in trial_data:
            try:
                results.append(_run_one(trial_id, seed, dx0, kwargs))
            except Exception as exc:
                failed_ids.append(trial_id)
                if on_trial_error == "raise":
                    raise
                if on_trial_error == "warn":
                    warnings.warn(
                        f"[MC trial {trial_id}] case_fn raised "
                        f"{type(exc).__name__}: {exc}\n" + traceback.format_exc(),
                        RuntimeWarning,
                        stacklevel=2,
                    )
    else:
        workers = (os.cpu_count() or 1) if n_workers < 0 else n_workers
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_run_one, *td): td[0] for td in trial_data}
            for future in as_completed(futures):
                trial_id = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    failed_ids.append(trial_id)
                    if on_trial_error == "raise":
                        raise
                    if on_trial_error == "warn":
                        warnings.warn(
                            f"[MC trial {trial_id}] case_fn raised "
                            f"{type(exc).__name__}: {exc}\n" + traceback.format_exc(),
                            RuntimeWarning,
                            stacklevel=2,
                        )
        results.sort(key=lambda r: r.trial_id)

    if failed_ids and on_trial_error != "raise":
        id_str = (
            str(failed_ids)
            if len(failed_ids) <= 10
            else f"{failed_ids[:10]} ... ({len(failed_ids)} total)"
        )
        warnings.warn(
            f"Monte Carlo run '{config.study_name}' completed with "
            f"{len(failed_ids)} / {config.n_trials} failed trials. "
            f"Failed trial IDs: {id_str}",
            RuntimeWarning,
            stacklevel=2,
        )

    return results
