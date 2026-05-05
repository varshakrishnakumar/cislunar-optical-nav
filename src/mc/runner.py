from __future__ import annotations

import inspect
import os
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .metrics import trial_result_from_run_case
from .sampler import make_trial_rng, sample_estimation_error, sample_injection_error
from .types import MonteCarloConfig, TrialResult


_KM_PER_LU       = 384_400.0
_MOON_RADIUS_KM  = 1737.4
_MOON_RADIUS_ND  = _MOON_RADIUS_KM / _KM_PER_LU


def _landmark_offsets_for_case(case: str) -> Optional[np.ndarray]:
    """Return Moon-fixed offsets (in CR3BP-ND or km — caller scales)
    for a named landmark case, or ``None`` if the case is "none".
    Each row is a unit-radius offset; multiply by the lunar radius in
    your unit system to get the actual landmark position relative to
    Moon center."""
    if case == "none" or not case:
        return None
    if case == "synthetic_6":
        return np.array([
            [+1, 0, 0], [-1, 0, 0],
            [0, +1, 0], [0, -1, 0],
            [0, 0, +1], [0, 0, -1],
        ], dtype=float)
    if case == "synthetic_12":
        # Six axes plus six edge-midpoint mixes.
        d = 1.0 / np.sqrt(2.0)
        return np.array([
            [+1, 0, 0], [-1, 0, 0],
            [0, +1, 0], [0, -1, 0],
            [0, 0, +1], [0, 0, -1],
            [+d, +d, 0], [-d, +d, 0], [+d, -d, 0], [-d, -d, 0],
            [+d, 0, +d], [0, +d, -d],
        ], dtype=float)
    if case in ("catalog_craters_6", "catalog_craters_12"):
        # Lat/lon-converted nearside crater catalog. Identity is
        # assumed known; image recognition / matching is not modeled.
        from cv.landmark_catalog import catalog_unit_offsets
        return catalog_unit_offsets(case)
    raise ValueError(f"Unknown landmark_case: {case!r}")


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
            q_acc=float(getattr(config, "q_acc", 1e-14)),
            sigma_att_rad=float(getattr(config, "sigma_att_rad", 0.0)),
            pointing_lag_steps=int(getattr(config, "pointing_lag_steps", 0)),
            meas_delay_steps=int(getattr(config, "meas_delay_steps", 0)),
            P0_scale=float(getattr(config, "P0_scale", 1.0)),
            disable_moon_center=bool(getattr(config, "disable_moon_center", False)),
            return_debug=False,
            accumulate_gramian=False,
        )

        bias = getattr(config, "bias_att_rad", None)
        if bias is not None:
            all_kwargs["bias_att_rad"] = np.asarray(bias, dtype=float)

        # Landmark case → positions array. CR3BP `run_case` accepts
        # ``landmark_positions`` (absolute, in ND); SPICE
        # ``run_case_spice`` accepts ``landmark_offsets_km`` (relative
        # to current Moon ephemeris position, in km). The dispatcher
        # detects which kwarg the case_fn supports and feeds the right
        # one — using the unit appropriate to the truth model.
        case_name = getattr(config, "landmark_case", "none")
        unit_offsets = _landmark_offsets_for_case(case_name)
        if unit_offsets is not None:
            sig_params = set(inspect.signature(case_fn).parameters)
            if "landmark_offsets_km" in sig_params:
                all_kwargs["landmark_offsets_km"] = (
                    unit_offsets * float(_MOON_RADIUS_KM)
                )
            elif "landmark_positions" in sig_params:
                moon_nd = np.array(
                    [1.0 - float(config.mu), 0.0, 0.0], dtype=float
                )
                all_kwargs["landmark_positions"] = (
                    moon_nd[None, :] + unit_offsets * float(_MOON_RADIUS_ND)
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
