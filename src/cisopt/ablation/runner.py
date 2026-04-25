"""Ablation runner: drives ``run_mc`` per combo and aggregates results."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..sweeps.runner import run_mc
from ..sweeps.storage import write_run_meta, write_trials
from ..sweeps.types import MCSweepCfg
from .engine import combos_from_axes
from .types import AblationCfg


@dataclass
class AblationResult:
    run_id: str
    out_dir: Path
    rows: list[dict[str, Any]]
    combos: list[dict[str, Any]]
    n_failed: int

    @property
    def n_trials(self) -> int:
        return len(self.rows)


def run_ablation(
    cfg: AblationCfg,
    *,
    out_dir: str | Path,
    on_trial_error: str = "warn",
) -> AblationResult:
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex[:12]
    combos = combos_from_axes(cfg.axes)

    all_rows: list[dict[str, Any]] = []
    combo_meta: list[dict[str, Any]] = []
    n_failed = 0

    t0 = time.perf_counter()

    for combo_id, overrides in combos:
        sweep = MCSweepCfg(
            base_cfg=cfg.base_cfg,
            n_trials=cfg.n_trials_per_combo,
            base_seed=cfg.base_seed,
            n_workers=cfg.n_workers,
        )
        # Inner MC sweeps don't write per-combo files; the ablation runner
        # owns the consolidated trials.parquet at the run root.
        result = run_mc(
            sweep,
            out_dir=out,
            on_trial_error=on_trial_error,
            overrides=overrides,
            combo_id=combo_id,
            run_id=run_id,
            write_outputs=False,
        )
        all_rows.extend(result.rows)
        n_failed += result.n_failed
        combo_meta.append({"combo_id": combo_id, "overrides": overrides})

    elapsed = time.perf_counter() - t0

    write_trials(all_rows, out / "trials.parquet")
    (out / "combos.json").write_text(json.dumps(combo_meta, indent=2, default=str))
    write_run_meta(
        out,
        run_id=run_id,
        base_cfg=cfg.base_cfg,
        extra={
            "kind": "ablation",
            "n_combos": len(combos),
            "n_trials_per_combo": int(cfg.n_trials_per_combo),
            "n_trials_total": len(all_rows),
            "n_failed": int(n_failed),
            "base_seed": int(cfg.base_seed),
            "n_workers": int(cfg.n_workers),
            "axes": dict(cfg.axes),
            "elapsed_s": float(elapsed),
        },
    )

    return AblationResult(
        run_id=run_id,
        out_dir=out,
        rows=all_rows,
        combos=combo_meta,
        n_failed=n_failed,
    )
