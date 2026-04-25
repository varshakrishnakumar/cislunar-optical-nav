"""Monte Carlo runner that drives ``run_trial`` over N seeded trials."""

from __future__ import annotations

import os
import time
import traceback
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ExperimentCfg, patch_cfg
from ..runner.trial import run_trial
from .storage import (
    failed_trial_row,
    trial_row,
    write_run_meta,
    write_trials,
)
from .types import MCSweepCfg, make_trial_seed


@dataclass
class MCResult:
    run_id: str
    out_dir: Path
    rows: list[dict[str, Any]]
    n_failed: int

    @property
    def n_trials(self) -> int:
        return len(self.rows)


def _run_one(
    *,
    base_cfg: ExperimentCfg,
    overrides: dict[str, Any],
    run_id: str,
    combo_id: str,
    trial_id: int,
    seed: int,
) -> dict[str, Any]:
    full_overrides = {**overrides, "trial.seed": int(seed)}
    cfg_i = patch_cfg(base_cfg, full_overrides)
    artifact = run_trial(cfg_i)
    return trial_row(
        artifact,
        run_id=run_id,
        combo_id=combo_id,
        trial_id=trial_id,
    )


def run_mc(
    sweep: MCSweepCfg,
    *,
    out_dir: str | Path,
    on_trial_error: str = "warn",
    overrides: dict[str, Any] | None = None,
    combo_id: str = "",
    run_id: str | None = None,
    write_outputs: bool = True,
) -> MCResult:
    if on_trial_error not in ("warn", "raise", "skip"):
        raise ValueError("on_trial_error must be 'warn', 'raise', or 'skip'")

    overrides = dict(overrides or {})
    run_id = run_id or uuid.uuid4().hex[:12]
    out = Path(out_dir).expanduser()
    if write_outputs:
        out.mkdir(parents=True, exist_ok=True)

    trial_ids = list(range(int(sweep.n_trials)))
    seeds = [make_trial_seed(sweep.base_seed, i) for i in trial_ids]

    rows: list[dict[str, Any]] = []
    failed_ids: list[int] = []

    def _record_failure(trial_id: int, seed: int, exc: BaseException) -> None:
        failed_ids.append(trial_id)
        if on_trial_error == "raise":
            raise exc
        if on_trial_error == "warn":
            warnings.warn(
                f"[MC trial {trial_id}] {type(exc).__name__}: {exc}\n"
                + traceback.format_exc(),
                RuntimeWarning,
                stacklevel=3,
            )
        rows.append(
            failed_trial_row(
                run_id=run_id,
                combo_id=combo_id,
                trial_id=trial_id,
                seed=seed,
                error_message=f"{type(exc).__name__}: {exc}",
            )
        )

    t0 = time.perf_counter()

    if sweep.n_workers == 1:
        for trial_id, seed in zip(trial_ids, seeds):
            try:
                rows.append(
                    _run_one(
                        base_cfg=sweep.base_cfg,
                        overrides=overrides,
                        run_id=run_id,
                        combo_id=combo_id,
                        trial_id=trial_id,
                        seed=seed,
                    )
                )
            except Exception as exc:
                _record_failure(trial_id, seed, exc)
    else:
        workers = (os.cpu_count() or 1) if sweep.n_workers < 0 else sweep.n_workers
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    _run_one,
                    base_cfg=sweep.base_cfg,
                    overrides=overrides,
                    run_id=run_id,
                    combo_id=combo_id,
                    trial_id=trial_id,
                    seed=seed,
                ): (trial_id, seed)
                for trial_id, seed in zip(trial_ids, seeds)
            }
            for fut in as_completed(futures):
                tid, seed = futures[fut]
                try:
                    rows.append(fut.result())
                except Exception as exc:
                    _record_failure(tid, seed, exc)
        rows.sort(key=lambda r: r["trial_id"])

    elapsed = time.perf_counter() - t0

    if write_outputs:
        write_trials(rows, out / "trials.parquet")
        write_run_meta(
            out,
            run_id=run_id,
            base_cfg=sweep.base_cfg,
            extra={
                "kind": "monte_carlo",
                "n_trials": int(sweep.n_trials),
                "base_seed": int(sweep.base_seed),
                "n_workers": int(sweep.n_workers),
                "n_failed": len(failed_ids),
                "elapsed_s": float(elapsed),
                "overrides": dict(overrides),
                "combo_id": str(combo_id),
            },
        )

    return MCResult(run_id=run_id, out_dir=out, rows=rows, n_failed=len(failed_ids))
