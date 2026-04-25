"""Single-impulse position-targeting wrapping guidance.targeting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dynamics.integrators import propagate
from guidance.targeting import solve_single_impulse_position_target

from ..protocols import Scenario, StateEstimate


Array = np.ndarray


@dataclass
class SingleImpulseGuidance:
    scenario: Scenario
    max_iter: int = 10
    tol: float = 1e-10
    name: str = "single_impulse"

    def solve(self, est: StateEstimate, scenario: Scenario) -> tuple[Array, dict[str, Any]]:
        result = solve_single_impulse_position_target(
            propagate=propagate,
            dynamics=scenario.dynamics,
            x0=est.x,
            t0=float(est.t_s),
            tc=float(est.t_s),
            tf=float(scenario.tf_s),
            r_target=scenario.target_position(),
            max_iter=int(self.max_iter),
            tol=float(self.tol),
        )
        info: dict[str, Any] = {
            "converged": bool(result.converged),
            "iterations": int(result.iterations),
            "final_pos_error": np.asarray(result.final_pos_error, dtype=float),
        }
        return np.asarray(result.dv, dtype=float), info


def build_single_impulse(params: dict[str, Any], scenario: Scenario) -> SingleImpulseGuidance:
    return SingleImpulseGuidance(scenario=scenario, **params)
