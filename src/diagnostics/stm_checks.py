from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dynamics.integrators import propagate
from dynamics.variational import cr3bp_eom_with_stm


Array = np.ndarray


@dataclass(frozen=True)
class STMColumnCheck:
    column_index: int
    abs_error_norm: float
    rel_error_norm: float
    analytic_col_norm: float
    numeric_col_norm: float


@dataclass(frozen=True)
class STMComparison:
    name: str
    passed: bool
    shape: tuple[int, int]
    max_abs_error: float
    max_rel_error: float
    fro_abs_error: float
    fro_rel_error: float
    worst_index: tuple[int, int]
    analytic_value: float
    numeric_value: float
    diff_value: float
    dt: float
    fd_eps: float
    atol: float
    rtol: float
    column_checks: list[STMColumnCheck]
    details: dict[str, Any]


def _as_state(x: Array) -> Array:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.shape != (6,):
        raise ValueError(f"x must have shape (6,), got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite.")
    return x


def _pack_state_and_stm(x: Array, phi: Array | None = None) -> Array:
    x = _as_state(x)
    if phi is None:
        phi = np.eye(6, dtype=float)
    phi = np.asarray(phi, dtype=float).reshape(6, 6)
    return np.concatenate([x, phi.reshape(-1, order="F")])


def _unpack_state_and_stm(z: Array) -> tuple[Array, Array]:
    z = np.asarray(z, dtype=float).reshape(-1)
    if z.size != 42:
        raise ValueError(f"Expected vector of length 42, got {z.size}")
    x = z[:6].copy()
    phi = z[6:].reshape(6, 6, order="F").copy()
    return x, phi


def _safe_rel_matrix(
    A: Array,
    B: Array,
    *,
    rel_floor: float = 1e-14,
) -> Array:
    denom = np.maximum(np.abs(B), rel_floor)
    return np.abs(A - B) / denom


def propagate_state_only(
    *,
    mu: float,
    x0: Array,
    t0: float,
    t1: float,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: float = np.inf,
) -> Array:
    z0 = _pack_state_and_stm(x0, np.eye(6, dtype=float))
    res = propagate(
        cr3bp_eom_with_stm,
        (float(t0), float(t1)),
        z0,
        args=(float(mu),),
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=False,
    )
    if not res.success:
        raise RuntimeError(f"Propagation failed: {res.message}")
    xf, _ = _unpack_state_and_stm(res.x[-1])
    return xf


def propagate_state_and_stm(
    *,
    mu: float,
    x0: Array,
    t0: float,
    t1: float,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: float = np.inf,
) -> tuple[Array, Array]:
    z0 = _pack_state_and_stm(x0, np.eye(6, dtype=float))
    res = propagate(
        cr3bp_eom_with_stm,
        (float(t0), float(t1)),
        z0,
        args=(float(mu),),
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=False,
    )
    if not res.success:
        raise RuntimeError(f"Propagation failed: {res.message}")
    return _unpack_state_and_stm(res.x[-1])


def finite_difference_stm(
    *,
    mu: float,
    x0: Array,
    t0: float,
    t1: float,
    fd_eps: float = 1e-7,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: float = np.inf,
    method: str = "central",
) -> tuple[Array, Array]:
    x0 = _as_state(x0)
    if fd_eps <= 0.0:
        raise ValueError("fd_eps must be > 0")
    if method not in {"forward", "central"}:
        raise ValueError(f"Unknown method: {method}")

    x_nom_t1 = propagate_state_only(
        mu=mu,
        x0=x0,
        t0=t0,
        t1=t1,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )

    Phi_fd = np.zeros((6, 6), dtype=float)

    for j in range(6):
        dx = np.zeros(6, dtype=float)
        dx[j] = fd_eps

        if method == "forward":
            xp_t1 = propagate_state_only(
                mu=mu,
                x0=x0 + dx,
                t0=t0,
                t1=t1,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
            )
            Phi_fd[:, j] = (xp_t1 - x_nom_t1) / fd_eps
        else:
            xp_t1 = propagate_state_only(
                mu=mu,
                x0=x0 + dx,
                t0=t0,
                t1=t1,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
            )
            xm_t1 = propagate_state_only(
                mu=mu,
                x0=x0 - dx,
                t0=t0,
                t1=t1,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
            )
            Phi_fd[:, j] = (xp_t1 - xm_t1) / (2.0 * fd_eps)

    return x_nom_t1, Phi_fd


def compare_stm_to_finite_difference(
    *,
    mu: float,
    x0: Array,
    t0: float,
    t1: float,
    fd_eps: float = 1e-7,
    method: str = "central",
    rtol_prop: float = 1e-10,
    atol_prop: float = 1e-12,
    max_step: float = np.inf,
    atol: float = 1e-6,
    rtol: float = 1e-4,
    name: str = "cr3bp_stm",
) -> STMComparison:
    x0 = _as_state(x0)
    xf_int, Phi_int = propagate_state_and_stm(
        mu=mu,
        x0=x0,
        t0=t0,
        t1=t1,
        rtol=rtol_prop,
        atol=atol_prop,
        max_step=max_step,
    )
    xf_fd, Phi_fd = finite_difference_stm(
        mu=mu,
        x0=x0,
        t0=t0,
        t1=t1,
        fd_eps=fd_eps,
        method=method,
        rtol=rtol_prop,
        atol=atol_prop,
        max_step=max_step,
    )

    state_mismatch = xf_int - xf_fd

    diff = Phi_int - Phi_fd
    abs_err = np.abs(diff)
    rel_err = _safe_rel_matrix(Phi_int, Phi_fd)

    max_abs_error = float(np.max(abs_err))
    max_rel_error = float(np.max(rel_err))
    fro_abs_error = float(np.linalg.norm(diff, ord="fro"))
    fro_rel_error = float(
        fro_abs_error / max(np.linalg.norm(Phi_fd, ord="fro"), 1e-14)
    )

    worst_index = tuple(int(i) for i in np.unravel_index(np.argmax(abs_err), abs_err.shape))
    wi, wj = worst_index

    column_checks: list[STMColumnCheck] = []
    for j in range(6):
        a_col = Phi_int[:, j]
        n_col = Phi_fd[:, j]
        col_diff = a_col - n_col
        abs_norm = float(np.linalg.norm(col_diff))
        denom = max(float(np.linalg.norm(n_col)), 1e-14)
        rel_norm = abs_norm / denom
        column_checks.append(
            STMColumnCheck(
                column_index=j,
                abs_error_norm=abs_norm,
                rel_error_norm=rel_norm,
                analytic_col_norm=float(np.linalg.norm(a_col)),
                numeric_col_norm=float(np.linalg.norm(n_col)),
            )
        )

    passed = bool(np.allclose(Phi_int, Phi_fd, atol=atol, rtol=rtol))

    return STMComparison(
        name=name,
        passed=passed,
        shape=Phi_int.shape,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        fro_abs_error=fro_abs_error,
        fro_rel_error=fro_rel_error,
        worst_index=worst_index,
        analytic_value=float(Phi_int[wi, wj]),
        numeric_value=float(Phi_fd[wi, wj]),
        diff_value=float(diff[wi, wj]),
        dt=float(t1 - t0),
        fd_eps=float(fd_eps),
        atol=float(atol),
        rtol=float(rtol),
        column_checks=column_checks,
        details={
            "x0": x0,
            "xf_integrated": xf_int,
            "xf_fd_nominal": xf_fd,
            "state_mismatch": state_mismatch,
            "Phi_integrated": Phi_int,
            "Phi_fd": Phi_fd,
            "abs_error_matrix": abs_err,
            "rel_error_matrix": rel_err,
            "diff_matrix": diff,
            "method": method,
            "rtol_prop": rtol_prop,
            "atol_prop": atol_prop,
            "max_step": max_step,
        },
    )

