from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2


Array = np.ndarray


@dataclass(frozen=True)
class MatrixHealth:
    name: str
    shape: tuple[int, ...]
    is_finite: bool
    symmetry_error_fro: float
    min_eig: float
    max_eig: float
    cond: float
    is_spd: bool
    chol_ok: bool


@dataclass(frozen=True)
class GateDecision:
    enabled: bool
    accepted: bool
    threshold: float
    statistic: float
    probability: float
    dof: int
    reason: str


def symmetrize(P: Array) -> Array:
    P = np.asarray(P, dtype=float)
    return 0.5 * (P + P.T)


def symmetry_error_fro(P: Array) -> float:
    P = np.asarray(P, dtype=float)
    return float(np.linalg.norm(P - P.T, ord="fro"))


def safe_eigvalsh(P: Array) -> Array:
    P = np.asarray(P, dtype=float)
    P = symmetrize(P)
    try:
        return np.linalg.eigvalsh(P)
    except np.linalg.LinAlgError:
        return np.full(P.shape[0], np.nan, dtype=float)


def min_eig(P: Array) -> float:
    eigs = safe_eigvalsh(P)
    return float(np.nanmin(eigs))


def max_eig(P: Array) -> float:
    eigs = safe_eigvalsh(P)
    return float(np.nanmax(eigs))


def matrix_condition_number(P: Array) -> float:
    P = np.asarray(P, dtype=float)
    try:
        return float(np.linalg.cond(P))
    except np.linalg.LinAlgError:
        return float("inf")


def is_spd(P: Array, *, atol: float = 1e-12) -> bool:
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False
    if not np.all(np.isfinite(P)):
        return False
    Ps = symmetrize(P)
    try:
        np.linalg.cholesky(Ps + atol * np.eye(P.shape[0]))
        return True
    except np.linalg.LinAlgError:
        return False


def check_matrix_health(name: str, P: Array, *, atol: float = 1e-12) -> MatrixHealth:
    P = np.asarray(P, dtype=float)
    finite = bool(np.all(np.isfinite(P)))
    sym_err = symmetry_error_fro(P)

    eigs = safe_eigvalsh(P)
    mn = float(np.nanmin(eigs)) if eigs.size else float("nan")
    mx = float(np.nanmax(eigs)) if eigs.size else float("nan")
    cond = matrix_condition_number(P)

    chol_ok = False
    if finite and P.ndim == 2 and P.shape[0] == P.shape[1]:
        try:
            np.linalg.cholesky(symmetrize(P) + atol * np.eye(P.shape[0]))
            chol_ok = True
        except np.linalg.LinAlgError:
            chol_ok = False

    return MatrixHealth(
        name=name,
        shape=P.shape,
        is_finite=finite,
        symmetry_error_fro=sym_err,
        min_eig=mn,
        max_eig=mx,
        cond=cond,
        is_spd=bool(finite and mn > -atol),
        chol_ok=chol_ok,
    )


def regularize_spd(P: Array, *, eps: float = 1e-12, max_tries: int = 6) -> Array:
    P = symmetrize(np.asarray(P, dtype=float))
    n = P.shape[0]
    I = np.eye(n, dtype=float)

    shift = 0.0
    for k in range(max_tries + 1):
        try:
            np.linalg.cholesky(P + shift * I)
            return P + shift * I
        except np.linalg.LinAlgError:
            shift = eps if k == 0 else 10.0 * shift

    return P + shift * I


def chol_solve_spd(S: Array, b: Array, *, regularize: bool = True) -> Array:
    S = np.asarray(S, dtype=float)
    b = np.asarray(b, dtype=float)

    S_use = regularize_spd(S) if regularize else symmetrize(S)
    c, lower = cho_factor(S_use, lower=True, check_finite=False)
    return cho_solve((c, lower), b, check_finite=False)


def mahalanobis2(v: Array, S: Array, *, regularize: bool = True) -> float:
    v = np.asarray(v, dtype=float).reshape(-1)
    z = chol_solve_spd(S, v, regularize=regularize)
    return float(v.T @ z)


def normalized_innovation_squared(y: Array, S: Array, *, regularize: bool = True) -> float:
    return mahalanobis2(y, S, regularize=regularize)


def gate_threshold(probability: float, dof: int) -> float:
    if not (0.0 < probability < 1.0):
        raise ValueError("probability must be in (0, 1)")
    if dof <= 0:
        raise ValueError("dof must be > 0")
    return float(chi2.ppf(probability, dof))


def decide_gate(
    statistic: float,
    *,
    probability: float,
    dof: int,
    enabled: bool = True,
) -> GateDecision:
    thr = gate_threshold(probability, dof)

    if not enabled:
        return GateDecision(
            enabled=False,
            accepted=True,
            threshold=thr,
            statistic=float(statistic),
            probability=float(probability),
            dof=int(dof),
            reason="gating_disabled",
        )

    if not np.isfinite(statistic):
        return GateDecision(
            enabled=True,
            accepted=False,
            threshold=thr,
            statistic=float(statistic),
            probability=float(probability),
            dof=int(dof),
            reason="nonfinite_statistic",
        )

    accepted = bool(statistic <= thr)
    return GateDecision(
        enabled=True,
        accepted=accepted,
        threshold=thr,
        statistic=float(statistic),
        probability=float(probability),
        dof=int(dof),
        reason="accepted" if accepted else "rejected_by_chi_square_gate",
    )


def joseph_update(
    x: Array,
    P: Array,
    H: Array,
    R: Array,
    y: Array,
) -> tuple[Array, Array, Array, Array]:
    x = np.asarray(x, dtype=float).reshape(-1)
    P = symmetrize(np.asarray(P, dtype=float))
    H = np.asarray(H, dtype=float)
    R = symmetrize(np.asarray(R, dtype=float))
    y = np.asarray(y, dtype=float).reshape(-1)

    S = symmetrize(H @ P @ H.T + R)
    PHt = P @ H.T
    K = chol_solve_spd(S, PHt.T).T

    x_upd = x + K @ y
    I = np.eye(P.shape[0], dtype=float)
    P_upd = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
    P_upd = symmetrize(P_upd)

    return x_upd, P_upd, K, S


def health_dict(h: MatrixHealth) -> dict[str, Any]:
    return {
        "name": h.name,
        "shape": h.shape,
        "is_finite": h.is_finite,
        "symmetry_error_fro": h.symmetry_error_fro,
        "min_eig": h.min_eig,
        "max_eig": h.max_eig,
        "cond": h.cond,
        "is_spd": h.is_spd,
        "chol_ok": h.chol_ok,
    }


def gate_dict(g: GateDecision) -> dict[str, Any]:
    return {
        "enabled": g.enabled,
        "accepted": g.accepted,
        "threshold": g.threshold,
        "statistic": g.statistic,
        "probability": g.probability,
        "dof": g.dof,
        "reason": g.reason,
    }
