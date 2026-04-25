"""Parquet-backed storage for Monte Carlo trial rows.

Schema is intentionally fixed and metric-flat — one row per trial, with
``run_id`` + ``combo_id`` columns so MC and ablation runs share the same
table layout. Axis definitions live in ``run_meta.json`` next to the table.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ..config import ExperimentCfg, to_dict
from ..results.artifact import TrialArtifact


TRIAL_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("combo_id", pa.string()),
    ("trial_id", pa.int32()),
    ("seed", pa.int64()),
    ("config_hash", pa.string()),
    ("dv_perfect_mag", pa.float64()),
    ("dv_ekf_mag", pa.float64()),
    ("dv_delta_mag", pa.float64()),
    ("dv_inflation_pct", pa.float64()),
    ("miss_uncorrected", pa.float64()),
    ("miss_perfect", pa.float64()),
    ("miss_ekf", pa.float64()),
    ("pos_err_tc", pa.float64()),
    ("trace_P_pos_tc", pa.float64()),
    ("nis_mean", pa.float64()),
    ("nees_mean", pa.float64()),
    ("valid_rate", pa.float64()),
    ("error_message", pa.string()),
])


def trial_row(
    artifact: TrialArtifact,
    *,
    run_id: str,
    combo_id: str = "",
    trial_id: int,
    error_message: str | None = None,
) -> dict[str, Any]:
    m = artifact.metrics
    return {
        "run_id": str(run_id),
        "combo_id": str(combo_id),
        "trial_id": int(trial_id),
        "seed": int(artifact.seed),
        "config_hash": str(artifact.config_hash),
        "dv_perfect_mag": float(m.dv_perfect_mag),
        "dv_ekf_mag": float(m.dv_ekf_mag),
        "dv_delta_mag": float(m.dv_delta_mag),
        "dv_inflation_pct": float(m.dv_inflation_pct),
        "miss_uncorrected": float(m.miss_uncorrected),
        "miss_perfect": float(m.miss_perfect),
        "miss_ekf": float(m.miss_ekf),
        "pos_err_tc": float(m.pos_err_tc),
        "trace_P_pos_tc": float(m.trace_P_pos_tc),
        "nis_mean": float(m.nis_mean),
        "nees_mean": float(m.nees_mean),
        "valid_rate": float(m.valid_rate),
        "error_message": "" if error_message is None else str(error_message),
    }


def failed_trial_row(
    *,
    run_id: str,
    combo_id: str,
    trial_id: int,
    seed: int,
    error_message: str,
) -> dict[str, Any]:
    nan = float("nan")
    return {
        "run_id": str(run_id),
        "combo_id": str(combo_id),
        "trial_id": int(trial_id),
        "seed": int(seed),
        "config_hash": "",
        "dv_perfect_mag": nan,
        "dv_ekf_mag": nan,
        "dv_delta_mag": nan,
        "dv_inflation_pct": nan,
        "miss_uncorrected": nan,
        "miss_perfect": nan,
        "miss_ekf": nan,
        "pos_err_tc": nan,
        "trace_P_pos_tc": nan,
        "nis_mean": nan,
        "nees_mean": nan,
        "valid_rate": nan,
        "error_message": error_message,
    }


def write_trials(rows: list[dict[str, Any]], path: str | Path) -> Path:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=TRIAL_SCHEMA)
    pq.write_table(table, out, compression="zstd")
    return out


def read_trials(path: str | Path) -> list[dict[str, Any]]:
    return pq.read_table(Path(path).expanduser()).to_pylist()


def write_run_meta(
    out_dir: str | Path,
    *,
    run_id: str,
    base_cfg: ExperimentCfg,
    extra: dict[str, Any] | None = None,
) -> Path:
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "run_id": str(run_id),
        "base_config": to_dict(base_cfg),
    }
    if extra is not None:
        payload.update(extra)
    path = out / "run_meta.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


def _json_default(obj: Any) -> Any:
    try:
        return asdict(obj)
    except TypeError:
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
