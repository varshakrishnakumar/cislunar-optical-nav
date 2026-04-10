from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .ephemeris import PointMassBody


Array = np.ndarray


@dataclass(frozen=True)
class PointMassDynamics:
    """Dimensional inertial point-mass dynamics with ephemeris-driven bodies."""

    bodies: Sequence[PointMassBody]
    min_distance_km: float = 1.0e-9
    name: str = "PointMass"

    def acceleration_km_s2(self, t_s: float, r_sc_km: Array) -> Array:
        r_sc = np.asarray(r_sc_km, dtype=float).reshape(3)
        acc = np.zeros(3, dtype=float)
        for body in self.bodies:
            rho = np.asarray(body.position_km(float(t_s)), dtype=float).reshape(3) - r_sc
            dist = float(np.linalg.norm(rho))
            if dist <= self.min_distance_km:
                raise RuntimeError(
                    f"Spacecraft is too close to {body.name}: distance={dist:.3e} km"
                )
            acc += float(body.gm_km3_s2) * rho / dist**3
        return acc

    def gravity_gradient_s2(self, t_s: float, r_sc_km: Array) -> Array:
        r_sc = np.asarray(r_sc_km, dtype=float).reshape(3)
        grad = np.zeros((3, 3), dtype=float)
        I3 = np.eye(3, dtype=float)
        for body in self.bodies:
            rho = np.asarray(body.position_km(float(t_s)), dtype=float).reshape(3) - r_sc
            dist = float(np.linalg.norm(rho))
            if dist <= self.min_distance_km:
                raise RuntimeError(
                    f"Spacecraft is too close to {body.name}: distance={dist:.3e} km"
                )
            grad += float(body.gm_km3_s2) * (
                -I3 / dist**3 + 3.0 * np.outer(rho, rho) / dist**5
            )
        return grad

    def eom(self, t_s: float, x: Array) -> Array:
        state = np.asarray(x, dtype=float).reshape(6)
        out = np.empty(6, dtype=float)
        out[:3] = state[3:6]
        out[3:6] = self.acceleration_km_s2(float(t_s), state[:3])
        return out

    def eom_with_stm(self, t_s: float, z: Array) -> Array:
        state_and_stm = np.asarray(z, dtype=float).reshape(42)
        x = state_and_stm[:6]
        phi = state_and_stm[6:].reshape((6, 6), order="F")

        A = np.zeros((6, 6), dtype=float)
        A[:3, 3:6] = np.eye(3, dtype=float)
        A[3:6, :3] = self.gravity_gradient_s2(float(t_s), x[:3])

        out = np.empty(42, dtype=float)
        out[:6] = self.eom(float(t_s), x)
        out[6:] = (A @ phi).reshape(-1, order="F")
        return out
