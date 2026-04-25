"""Minimal JSON+npz artifact store for Phase A.

Phase B will swap in a Parquet-backed results database with config-hash
indexing and cross-run query helpers. For now we keep the on-disk layout
simple and readable so individual trials can be inspected by hand.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .artifact import TrialArtifact, TrialMetrics


def _scalar_summary(art: TrialArtifact) -> dict[str, Any]:
    return {
        "config": art.config,
        "config_hash": art.config_hash,
        "seed": art.seed,
        "metrics": asdict(art.metrics),
        "units": art.units,
        "notes": art.notes,
    }


def save_artifact(art: TrialArtifact, out_dir: str | Path) -> Path:
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(_scalar_summary(art), indent=2, default=_json_default))

    if art.timeseries:
        ts_arrays = {k: np.asarray(v) for k, v in art.timeseries.items()}
        np.savez_compressed(out / "timeseries.npz", **ts_arrays)

    return out


def load_artifact(in_dir: str | Path) -> TrialArtifact:
    in_path = Path(in_dir).expanduser()
    summary = json.loads((in_path / "summary.json").read_text())

    ts: dict[str, np.ndarray] = {}
    npz_path = in_path / "timeseries.npz"
    if npz_path.exists():
        with np.load(npz_path) as data:
            ts = {k: np.asarray(data[k]) for k in data.files}

    return TrialArtifact(
        config=summary["config"],
        config_hash=summary["config_hash"],
        seed=int(summary["seed"]),
        metrics=TrialMetrics(**summary["metrics"]),
        units=dict(summary["units"]),
        timeseries=ts,
        notes=dict(summary.get("notes", {})),
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
