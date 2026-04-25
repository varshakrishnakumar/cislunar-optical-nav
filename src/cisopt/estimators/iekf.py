"""Iterated EKF estimator wrapping the existing nav/ machinery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nav.ekf import ekf_propagate_stm
from nav.measurements.bearing import bearing_update_tangent

from ..protocols import Measurement, Scenario, StateEstimate


Array = np.ndarray


@dataclass
class IEKFEstimator:
    scenario: Scenario
    q_acc: float = 1e-9
    rtol: float = 1e-10
    atol: float = 1e-12
    max_iterations: int = 3
    gating_enabled: bool = False
    gate_probability: float = 0.9973
    name: str = "iekf"

    def predict(
        self, t1: float, est: StateEstimate,
    ) -> tuple[StateEstimate, dict[str, Any]]:
        x_pred, P_pred, Phi = ekf_propagate_stm(
            dynamics=self.scenario.dynamics,
            x=est.x,
            P=est.P,
            t0=float(est.t_s),
            t1=float(t1),
            q_acc=float(self.q_acc),
            rtol=float(self.rtol),
            atol=float(self.atol),
        )
        new_est = StateEstimate(t_s=float(t1), x=x_pred, P=P_pred)
        return new_est, {"Phi_step": Phi}

    def update(
        self,
        est: StateEstimate,
        meas: Measurement,
    ) -> tuple[StateEstimate, dict[str, Any]]:
        if not meas.valid or meas.payload is None:
            return est.copy(), {"accepted": False, "nis": float("nan"), "reason": "invalid"}

        if meas.payload.get("kind") != "bearing":
            raise ValueError(
                f"IEKFEstimator only handles 'bearing' payloads, got {meas.payload.get('kind')!r}"
            )

        upd = bearing_update_tangent(
            est.x,
            est.P,
            meas.payload["u_global"],
            meas.payload["r_body"],
            float(meas.payload["sigma_theta"]),
            gating_enabled=bool(self.gating_enabled),
            gate_probability=float(self.gate_probability),
            max_iterations=int(self.max_iterations),
        )

        if upd.accepted:
            new_est = StateEstimate(t_s=float(est.t_s), x=upd.x_upd, P=upd.P_upd)
        else:
            new_est = est.copy()

        info: dict[str, Any] = {
            "accepted": bool(upd.accepted),
            "nis": float(upd.nis),
            "iterations": int(upd.iterations),
            "converged": bool(upd.converged),
            "H": np.asarray(upd.H, dtype=float),
        }
        if upd.final_innovation is not None:
            info["innovation"] = np.asarray(upd.final_innovation, dtype=float)
        return new_est, info


def build_iekf(params: dict[str, Any], scenario: Scenario) -> IEKFEstimator:
    return IEKFEstimator(scenario=scenario, **params)
