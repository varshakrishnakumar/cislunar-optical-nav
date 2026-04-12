from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from diagnostics.health import (
    GateDecision,
    chol_solve_spd,
    decide_gate,
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
    final_innovation: Array | None = None
    iterations: int = 0
    converged: bool = False


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
    max_iterations: int = 3,
    step_tolerance: float = 1e-12,
) -> BearingUpdateResult:
    return bearing_update_tangent_iekf(
        x=x,
        P=P,
        u_meas=u_meas,
        r_body=r_body,
        sigma_theta=sigma_theta,
        gating_enabled=gating_enabled,
        gate_probability=gate_probability,
        gate_dof=gate_dof,
        max_iterations=max_iterations,
        step_tolerance=step_tolerance,
    )


def _innovation_gain(P: Array, H: Array, R: Array) -> tuple[Array, Array]:
    S = symmetrize(H @ P @ H.T + R)
    PHt = P @ H.T
    K = chol_solve_spd(S, PHt.T).T
    return S, K


def _joseph_covariance_update(P: Array, H: Array, R: Array, K: Array) -> Array:
    I = np.eye(P.shape[0], dtype=float)
    P_upd = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
    return symmetrize(P_upd)


def bearing_update_tangent_iekf(
    x: Array,
    P: Array,
    u_meas: Array,
    r_body: Array,
    sigma_theta: float,
    *,
    gating_enabled: bool = False,
    gate_probability: float = 0.9973,
    gate_dof: int = 2,
    max_iterations: int = 3,
    step_tolerance: float = 1e-12,
) -> BearingUpdateResult:
    x = np.asarray(x, dtype=float).reshape(6)
    P = symmetrize(np.asarray(P, dtype=float).reshape(6, 6))

    max_iterations = int(max_iterations)
    if max_iterations <= 0:
        raise ValueError("max_iterations must be > 0")
    if not np.isfinite(step_tolerance) or step_tolerance < 0.0:
        raise ValueError("step_tolerance must be finite and >= 0")

    model = bearing_measurement_model(
        x=x,
        u_meas=u_meas,
        r_body=r_body,
        sigma_theta=sigma_theta,
    )

    y = model.residual_2d
    H = model.H
    R = model.R

    S, _ = _innovation_gain(P, H, R)
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
            final_innovation=y,
            iterations=0,
            converged=False,
        )

    x_prior = x.copy()
    x_iter = x.copy()
    converged = False
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        iter_model = bearing_measurement_model(
            x=x_iter,
            u_meas=u_meas,
            r_body=r_body,
            sigma_theta=sigma_theta,
        )
        _, K_iter = _innovation_gain(P, iter_model.H, iter_model.R)

        correction_residual = iter_model.residual_2d + iter_model.H @ (x_iter - x_prior)
        x_next = x_prior + K_iter @ correction_residual

        step_norm = float(np.linalg.norm(x_next - x_iter))
        x_iter = np.asarray(x_next, dtype=float).reshape(6)

        if step_norm <= float(step_tolerance) * (1.0 + float(np.linalg.norm(x_iter))):
            converged = True
            break

    final_model = bearing_measurement_model(
        x=x_iter,
        u_meas=u_meas,
        r_body=r_body,
        sigma_theta=sigma_theta,
    )
    final_y = final_model.residual_2d
    H_final = final_model.H
    R_final = final_model.R
    S_final, K_final = _innovation_gain(P, H_final, R_final)
    P_upd = _joseph_covariance_update(P, H_final, R_final, K_final)

    # Warn if the final innovation is suspiciously large (> 5σ in angle units),
    # which suggests the iteration converged to a local minimum rather than
    # driving the residual to zero.
    _innov_norm = float(np.linalg.norm(final_y))
    _innov_thr = 5.0 * float(sigma_theta) * float(np.sqrt(2.0))
    if _innov_norm > _innov_thr:
        warnings.warn(
            f"IEKF bearing update: final innovation norm ({_innov_norm:.3e} rad) "
            f"exceeds 5σ threshold ({_innov_thr:.3e} rad). "
            "The iteration may have converged to a local minimum — "
            "consider increasing max_iterations or inspecting the measurement.",
            RuntimeWarning,
            stacklevel=2,
        )

    return BearingUpdateResult(
        x_upd=x_iter,
        P_upd=P_upd,
        innovation=y,
        nis=nis,
        H=H_final,
        R=R_final,
        S=S_final,
        K=K_final,
        gate=gate,
        accepted=True,
        final_innovation=final_y,
        iterations=iterations,
        converged=converged,
    )
