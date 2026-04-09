from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from diagnostics.health import (
    GateDecision,
    decide_gate,
    joseph_update,
    normalized_innovation_squared,
    symmetrize,
)

Array = np.ndarray


@dataclass(frozen=True)
class BearingMeasurementModel:
    u_pred: Array
    u_meas: Array
    residual_3d: Array
    residual_2d: Array
    E: Array
    H: Array
    R: Array
    range_to_body: float


@dataclass(frozen=True)
class BearingUpdateResult:
    x_upd: Array
    P_upd: Array
    innovation: Array
    nis: float
    H: Array
    R: Array
    S: Array
    K: Array
    gate: GateDecision
    accepted: bool


def _unit(v: Array, eps: float = 1e-12) -> Array:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= eps:
        raise ValueError(f"Cannot normalize vector with norm <= {eps}: {v}")
    return v / n


def tangent_basis(u: Array) -> tuple[Array, Array]:
    u = _unit(u)

    a = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(a, u))) > 0.9:
        a = np.array([0.0, 1.0, 0.0], dtype=float)

    e1 = np.cross(u, a)
    e1 = _unit(e1)
    e2 = np.cross(u, e1)
    e2 = _unit(e2)
    return e1, e2


def los_unit(r_body: Array, r_sc: Array) -> tuple[Array, float]:
    rho = np.asarray(r_body, dtype=float).reshape(3) - np.asarray(r_sc, dtype=float).reshape(3)
    rng = float(np.linalg.norm(rho))
    if not np.isfinite(rng) or rng <= 1e-12:
        raise ValueError("Range to body is too small or non-finite.")
    u = rho / rng
    return u, rng


def bearing_measurement_model(
    x: Array,
    u_meas: Array,
    r_body: Array,
    sigma_theta: float,
) -> BearingMeasurementModel:
    x = np.asarray(x, dtype=float).reshape(6)
    r_body = np.asarray(r_body, dtype=float).reshape(3)

    if not np.isfinite(sigma_theta) or sigma_theta <= 0.0:
        raise ValueError(f"sigma_theta must be finite and > 0, got {sigma_theta}")

    r_sc = x[:3]
    u_pred, rng = los_unit(r_body, r_sc)

    u_meas = _unit(u_meas)

    e1, e2 = tangent_basis(u_pred)
    E = np.vstack([e1, e2])

    residual_3d = u_meas - u_pred
    residual_2d = E @ residual_3d

    I3 = np.eye(3, dtype=float)
    du_drsc = -(I3 - np.outer(u_pred, u_pred)) / rng

    H = np.zeros((2, 6), dtype=float)
    H[:, :3] = E @ du_drsc

    R = (float(sigma_theta) ** 2) * np.eye(2, dtype=float)

    return BearingMeasurementModel(
        u_pred=u_pred,
        u_meas=u_meas,
        residual_3d=residual_3d,
        residual_2d=residual_2d,
        E=E,
        H=H,
        R=R,
        range_to_body=rng,
    )


def bearing_predict_measurement(
    x: Array,
    r_body: Array,
) -> Array:
    x = np.asarray(x, dtype=float).reshape(6)
    u_pred, _ = los_unit(np.asarray(r_body, dtype=float).reshape(3), x[:3])
    return u_pred


def bearing_update_tangent(
    x: Array,
    P: Array,
    u_meas: Array,
    r_body: Array,
    sigma_theta: float,
    *,
    gating_enabled: bool = False,
    gate_probability: float = 0.9973,
    gate_dof: int = 2,
) -> BearingUpdateResult:
    x = np.asarray(x, dtype=float).reshape(6)
    P = symmetrize(np.asarray(P, dtype=float).reshape(6, 6))

    model = bearing_measurement_model(
        x=x,
        u_meas=u_meas,
        r_body=r_body,
        sigma_theta=sigma_theta,
    )

    y = model.residual_2d
    H = model.H
    R = model.R

    S = symmetrize(H @ P @ H.T + R)
    nis = normalized_innovation_squared(y, S, regularize=True)

    gate = decide_gate(
        nis,
        probability=float(gate_probability),
        dof=int(gate_dof),
        enabled=bool(gating_enabled),
    )

    if not gate.accepted:
        return BearingUpdateResult(
            x_upd=x.copy(),
            P_upd=P.copy(),
            innovation=y,
            nis=nis,
            H=H,
            R=R,
            S=S,
            K=np.zeros((6, 2), dtype=float),
            gate=gate,
            accepted=False,
        )

    x_upd, P_upd, K, _ = joseph_update(
        x=x,
        P=P,
        H=H,
        R=R,
        y=y,
    )

    return BearingUpdateResult(
        x_upd=x_upd,
        P_upd=P_upd,
        innovation=y,
        nis=nis,
        H=H,
        R=R,
        S=S,
        K=K,
        gate=gate,
        accepted=True,
    )
