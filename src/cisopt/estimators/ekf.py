"""Plain (non-iterated) EKF estimator.

The codebase's bearing update is iterated by default; setting
max_iterations=1 makes it a standard EKF. We keep this as a separate
registered estimator so config-driven ablations can compare EKF / IEKF / UKF
without changing scenario or sensor wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..protocols import Scenario
from .iekf import IEKFEstimator


@dataclass
class EKFEstimator(IEKFEstimator):
    max_iterations: int = 1
    name: str = "ekf"


def build_ekf(params: dict[str, Any], scenario: Scenario) -> EKFEstimator:
    return EKFEstimator(scenario=scenario, **params)
