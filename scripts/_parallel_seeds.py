"""Process-pool wrapper used by the new analysis drivers.

The new 06i–06q drivers loop over seeds serially, which at ~1.1 s/trial
single-threaded CR3BP would require many hours for the 1000-trial
production runs.  Threads don't help (GIL on solve_ivp under SciPy
without numba); a process pool does.

This module exposes a single helper, :func:`run_seeds_parallel`, that
maps a deterministic-seed Monte-Carlo loop across worker processes and
collects per-trial dict rows.  The worker imports
``scripts/06_midcourse_ekf_correction.py`` once per process via the
truth-mode dispatcher in ``_analysis_common.load_midcourse_run_case``
so we don't pay the import cost per trial.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


_SRC = Path(__file__).resolve().parent.parent / "src"
_SCRIPTS = Path(__file__).resolve().parent
for p in (str(_SRC), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)


_RUN_CASE_CACHE: dict[str, Callable[..., Any]] = {}


def _get_run_case(truth: str) -> Callable[..., Any]:
    if truth not in _RUN_CASE_CACHE:
        from _analysis_common import load_midcourse_run_case
        _RUN_CASE_CACHE[truth] = load_midcourse_run_case(truth=truth)
    return _RUN_CASE_CACHE[truth]


def _worker_run_one(args: tuple) -> Optional[dict]:
    """Pickleable worker: one trial → one row dict, or ``None`` on failure.

    Args is a tuple ``(truth, trial_id, base_seed, kwargs_extra,
    extract_fields, sigma_r_inj, sigma_v_inj, sigma_r_est, sigma_v_est,
    planar_only)``.  ``kwargs_extra`` is forwarded into ``run_case``;
    ``extract_fields`` is a list of (output_key, run_case_key) pairs.
    """
    import numpy as np  # local for the worker
    truth, trial_id, base_seed, kw_extra, extract_fields, \
        sr_inj, sv_inj, sr_est, sv_est, planar_only = args

    try:
        from mc.sampler import (
            make_trial_rng,
            sample_estimation_error,
            sample_injection_error,
        )
        rng  = make_trial_rng(int(base_seed), int(trial_id))
        seed = int(rng.integers(0, 2**31 - 1))
        dx0  = sample_injection_error(
            rng, sigma_r=float(sr_inj), sigma_v=float(sv_inj),
            planar_only=bool(planar_only),
        )
        est  = sample_estimation_error(
            rng, sigma_r=float(sr_est), sigma_v=float(sv_est),
            planar_only=bool(planar_only),
        )
        run_case = _get_run_case(truth)
        out = run_case(
            seed=seed, dx0=dx0, est_err=est,
            return_debug=False, accumulate_gramian=False,
            **kw_extra,
        )
        row = {"trial_id": int(trial_id), "seed": int(seed)}
        for out_key, run_key in extract_fields:
            try:
                row[out_key] = float(out[run_key])
            except (KeyError, TypeError, ValueError):
                row[out_key] = float("nan")
        return row
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"[parallel-worker trial {trial_id}] failed: {exc}\n"
        )
        return None


def run_seeds_parallel(
    *,
    truth: str,
    n_seeds: int,
    base_seed: int,
    kwargs_extra: Dict[str, Any],
    extract_fields: List[tuple[str, str]],
    n_workers: int = -1,
    sigma_r_inj: float = 1e-4,
    sigma_v_inj: float = 1e-4,
    sigma_r_est: float = 1e-4,
    sigma_v_est: float = 1e-4,
    planar_only: bool = False,
    extra_row_fields: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """Run ``n_seeds`` trials in parallel, return one row dict per trial.

    ``extract_fields`` controls the columns. Example:
    ``[("miss_ekf", "miss_ekf"), ("nis_mean_all", "nis_mean_all")]``.

    Use ``extra_row_fields`` to attach static values (e.g. config name,
    sweep parameter value) to every row.
    """
    if n_workers < 0:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, int(n_workers))
    extra_row_fields = dict(extra_row_fields or {})

    args_list = [
        (
            truth, int(trial_id), int(base_seed),
            dict(kwargs_extra), list(extract_fields),
            float(sigma_r_inj), float(sigma_v_inj),
            float(sigma_r_est), float(sigma_v_est),
            bool(planar_only),
        )
        for trial_id in range(int(n_seeds))
    ]

    rows: List[dict] = []
    if n_workers == 1:
        # Useful escape hatch for SPICE truth (not thread/process-safe).
        for a in args_list:
            r = _worker_run_one(a)
            if r is not None:
                r.update(extra_row_fields)
                rows.append(r)
        return rows

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for r in ex.map(_worker_run_one, args_list, chunksize=4):
            if r is not None:
                r.update(extra_row_fields)
                rows.append(r)
    rows.sort(key=lambda d: d["trial_id"])
    return rows


__all__ = ["run_seeds_parallel"]
