from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class TrialMetrics:
    dv_perfect_mag: float
    dv_ekf_mag: float
    dv_delta_mag: float
    dv_inflation_pct: float

    miss_uncorrected: float
    miss_perfect: float
    miss_ekf: float

    pos_err_tc: float
    trace_P_pos_tc: float
    nis_mean: float
    nees_mean: float
    valid_rate: float


@dataclass
class TrialArtifact:
    config: dict[str, Any]
    config_hash: str
    seed: int
    metrics: TrialMetrics
    units: dict[str, str]
    timeseries: dict[str, Array] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)
