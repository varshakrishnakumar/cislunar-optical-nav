# nav/ekf.py
from __future__ import annotations
import numpy as np

from dynamics.integrators import propagate
from dynamics.variational import cr3bp_eom_with_stm

Array = np.ndarray


def Qd_white_accel(dt: float, q_acc: float) -> Array:
    """6x6 discrete white-acceleration noise for [r,v]."""
    if dt <= 0.0:
        return np.zeros((6, 6), dtype=float)
    I3 = np.eye(3)
    Q = np.zeros((6, 6), dtype=float)
    Qrr = (dt**3 / 3.0) * q_acc * I3
    Qrv = (dt**2 / 2.0) * q_acc * I3
    Qvv = (dt)         * q_acc * I3
    Q[0:3, 0:3] = Qrr
    Q[0:3, 3:6] = Qrv
    Q[3:6, 0:3] = Qrv
    Q[3:6, 3:6] = Qvv
    return Q


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
    """
    Propagate [x,P] from t0->t1 using CR3BP + STM from variational.py.
    Returns (x_pred, P_pred, Phi).
    """
    x = np.asarray(x, dtype=float).reshape(6)
    P = np.asarray(P, dtype=float).reshape(6, 6)

    Phi0 = np.eye(6, dtype=float).reshape(-1, order="F")  # must match variational.py
    z0 = np.concatenate([x, Phi0])  # 42

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

    zf = res.x[-1]  # (42,)
    x_pred = zf[:6]
    Phi = zf[6:].reshape((6, 6), order="F")

    dt = float(t1 - t0)
    Qd = Qd_white_accel(abs(dt), q_acc)
    P_pred = Phi @ P @ Phi.T + Qd
    P_pred = 0.5 * (P_pred + P_pred.T)

    return x_pred, P_pred, Phi