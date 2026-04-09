from __future__ import annotations

import warnings

import numpy as np

from dynamics.integrators import propagate
from dynamics.variational import cr3bp_eom_with_stm

Array = np.ndarray

_PD_HARD_FLOOR = -1e-8


def Qd_white_accel(dt: float, q_acc: float) -> Array:
    if dt <= 0.0 or q_acc == 0.0:
        return np.zeros((6, 6), dtype=float)
    I3 = np.eye(3, dtype=float)
    Q = np.zeros((6, 6), dtype=float)
    Q[0:3, 0:3] = (dt**3 / 3.0) * q_acc * I3
    Q[0:3, 3:6] = (dt**2 / 2.0) * q_acc * I3
    Q[3:6, 0:3] = (dt**2 / 2.0) * q_acc * I3
    Q[3:6, 3:6] =  dt            * q_acc * I3
    return Q


def _enforce_pd(P: Array, *, context: str = "") -> Array:
    eigvals, eigvecs = np.linalg.eigh(P)
    min_eig = float(eigvals.min())

    if min_eig >= 0.0:
        return P

    if min_eig < _PD_HARD_FLOOR:
        prefix = f"[{context}] " if context else ""
        raise RuntimeError(
            f"{prefix}P lost positive definiteness: "
            f"min eigenvalue = {min_eig:.3e} (threshold {_PD_HARD_FLOOR:.0e}). "
            "This indicates genuine filter divergence, not floating-point noise."
        )

    prefix = f"[{context}] " if context else ""
    warnings.warn(
        f"{prefix}P had small negative eigenvalue ({min_eig:.3e}); "
        "floored to zero. This is normal floating-point drift near libration points.",
        RuntimeWarning,
        stacklevel=3,
    )
    eigvals_floored = np.maximum(eigvals, 0.0)
    P_fixed = eigvecs @ np.diag(eigvals_floored) @ eigvecs.T
    return 0.5 * (P_fixed + P_fixed.T)


def ekf_propagate_cr3bp_stm(
    *,
    mu: float,
    x: Array,
    P: Array,
    t0: float,
    t1: float,
    q_acc: float = 0.0,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: float = np.inf,
) -> tuple[Array, Array, Array]:
    x = np.asarray(x, dtype=float).reshape(6)
    P = np.asarray(P, dtype=float).reshape(6, 6)

    if t0 == t1:
        return x.copy(), P.copy(), np.eye(6, dtype=float)

    Phi0 = np.eye(6, dtype=float).reshape(-1, order="F")
    z0 = np.concatenate([x, Phi0])

    res = propagate(
        cr3bp_eom_with_stm,
        (t0, t1),
        z0,
        args=(mu,),
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=False,
    )
    if not res.success:
        raise RuntimeError(f"CR3BP propagation failed: {res.message}")

    zf = res.x[-1]
    x_pred = zf[:6]
    Phi = zf[6:].reshape((6, 6), order="F")

    dt = float(t1 - t0)
    Qd = Qd_white_accel(abs(dt), q_acc)
    P_pred = Phi @ P @ Phi.T + Qd
    P_pred = 0.5 * (P_pred + P_pred.T)

    P_pred = _enforce_pd(P_pred, context=f"t0={t0:.4f}→t1={t1:.4f}")

    return x_pred, P_pred, Phi
