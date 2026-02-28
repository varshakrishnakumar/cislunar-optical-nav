from __future__ import annotations
import numpy as np

Array = np.ndarray


def tangent_basis(u: Array) -> tuple[Array, Array]:
    u = np.asarray(u, dtype=float).reshape(3)
    u = u / np.linalg.norm(u)

    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, u)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(u, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def los_unit(r_body: Array, r_sc: Array) -> tuple[Array, float]:
    rho = np.asarray(r_body, float).reshape(3) - np.asarray(r_sc, float).reshape(3)
    rng = float(np.linalg.norm(rho))
    u = rho / rng
    return u, rng


def bearing_update_tangent(
    x: Array,
    P: Array,
    u_meas: Array,
    r_body: Array,
    sigma_theta: float,
) -> tuple[Array, Array, Array, float]:
    """
    EKF update using 2D tangent-plane residual for a LOS unit-vector measurement.
    State is [r(3), v(3)] in the same frame as r_body.
    """
    x = np.asarray(x, float).reshape(6)
    P = np.asarray(P, float).reshape(6, 6)

    r_sc = x[:3]
    u_pred, rng = los_unit(r_body, r_sc)

    e1, e2 = tangent_basis(u_pred)
    E = np.vstack([e1, e2])  # 2x3

    u_meas = np.asarray(u_meas, float).reshape(3)
    u_meas = u_meas / np.linalg.norm(u_meas)

    y3 = u_meas - u_pred
    y = E @ y3  # (2,)

    I3 = np.eye(3)
    du_drsc = -(I3 - np.outer(u_pred, u_pred)) / rng  # 3x3

    H = np.zeros((2, 6))
    H[:, :3] = E @ du_drsc

    R = (sigma_theta**2) * np.eye(2)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    x_upd = x + K @ y

    I6 = np.eye(6)
    P_upd = (I6 - K @ H) @ P @ (I6 - K @ H).T + K @ R @ K.T
    P_upd = 0.5 * (P_upd + P_upd.T)

    nis = float(y.T @ np.linalg.inv(S) @ y)
    return x_upd, P_upd, y, nis