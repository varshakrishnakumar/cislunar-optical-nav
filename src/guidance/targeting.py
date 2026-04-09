from __future__ import annotations

import warnings
from dataclasses import dataclass
import numpy as np
from typing import Callable, Optional

from dynamics.variational import cr3bp_eom_with_stm

@dataclass
class TargetingResult:
    dv: np.ndarray
    converged: bool
    iterations: int
    final_pos_error: np.ndarray
    history: list[dict]

def _pack_state_and_stm(x: np.ndarray, phi: Optional[np.ndarray] = None) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(6,)
    if phi is None:
        phi = np.eye(6, dtype=float)
    phi = np.asarray(phi, dtype=float).reshape(6, 6)
    return np.concatenate([x, phi.reshape(-1, order="F")])

def _unpack(zf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    zf = np.asarray(zf, dtype=float).reshape(-1,)
    x = zf[:6].copy()
    phi = zf[6:].reshape(6, 6, order="F").copy()
    return x, phi

def solve_single_impulse_position_target(
    *,
    propagate: Callable,
    mu: float,
    x0: np.ndarray,
    t0: float,
    tc: float,
    tf: float,
    r_target: np.ndarray,
    dv0: Optional[np.ndarray] = None,
    max_iter: int = 10,
    tol: float = 1e-10,
) -> TargetingResult:
    x0 = np.asarray(x0, dtype=float).reshape(6,)
    r_target = np.asarray(r_target, dtype=float).reshape(3,)

    z0 = _pack_state_and_stm(x0)
    res_tc = propagate(cr3bp_eom_with_stm, (t0, tc), z0, args=(mu,), dense_output=False)
    if not res_tc.success:
        raise RuntimeError(f"Propagation to tc failed: {res_tc.message}")
    x_tc, _ = _unpack(res_tc.x[-1])

    dv = np.zeros(3) if dv0 is None else np.asarray(dv0, dtype=float).reshape(3,)
    history: list[dict] = []

    err = np.full(3, np.inf)

    for k in range(1, max_iter + 1):
        x_burn = x_tc.copy()
        x_burn[3:6] += dv

        z_tc = _pack_state_and_stm(x_burn, np.eye(6))
        res_tf = propagate(cr3bp_eom_with_stm, (tc, tf), z_tc, args=(mu,), dense_output=False)
        if not res_tf.success:
            raise RuntimeError(f"Propagation to tf failed: {res_tf.message}")

        x_tf, phi = _unpack(res_tf.x[-1])
        r_tf = x_tf[:3]
        err = r_tf - r_target

        history.append({"iter": k, "dv": dv.copy(), "err": err.copy(), "err_norm": float(np.linalg.norm(err))})

        if np.linalg.norm(err) < tol:
            return TargetingResult(dv=dv, converged=True, iterations=k, final_pos_error=err, history=history)

        Phi_rv = phi[0:3, 3:6]

        cond = float(np.linalg.cond(Phi_rv))
        if not np.isfinite(cond) or cond > 1e10:
            warnings.warn(
                f"targeting iter {k}: Phi_rv is near-singular (cond = {cond:.3e}). "
                "Newton step may be unreliable; consider re-initializing dv0 or "
                "checking the target geometry.",
                stacklevel=2,
            )

        try:
            delta = np.linalg.solve(Phi_rv, err)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(Phi_rv, err, rcond=None)

        dv = dv - delta

    return TargetingResult(dv=dv, converged=False, iterations=max_iter, final_pos_error=err, history=history)
