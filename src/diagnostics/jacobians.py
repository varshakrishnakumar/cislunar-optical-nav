from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from nav.measurements.bearing import bearing_measurement_model


Array = np.ndarray
VectorFn = Callable[[Array], Array]
JacobianFn = Callable[[Array], Array]


@dataclass(frozen=True)
class JacobianComparison:
    name: str
    passed: bool
    shape: tuple[int, int]
    max_abs_error: float
    max_rel_error: float
    rms_abs_error: float
    worst_index: tuple[int, int]
    analytic_value: float
    numeric_value: float
    diff_value: float
    atol: float
    rtol: float
    details: dict[str, Any]


def _as_vector(x: Array, *, name: str = "x") -> Array:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    return arr


def _as_matrix(J: Array, *, name: str = "J") -> Array:
    arr = np.asarray(J, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    return arr


def _safe_rel_error(
    Ja: Array,
    Jn: Array,
    *,
    rel_floor: float = 1e-14,
) -> Array:
    denom = np.maximum(np.abs(Jn), rel_floor)
    return np.abs(Ja - Jn) / denom


def numeric_jacobian(
    fun: VectorFn,
    x: Array,
    *,
    eps: float = 1e-6,
    method: str = "central",
) -> Array:
    x = _as_vector(x, name="x")
    f0 = _as_vector(fun(x), name="fun(x)")
    m = f0.size
    n = x.size

    if eps <= 0.0:
        raise ValueError("eps must be > 0")
    if method not in {"forward", "central"}:
        raise ValueError(f"Unknown method: {method}")

    J = np.zeros((m, n), dtype=float)

    for j in range(n):
        dx = np.zeros_like(x)
        dx[j] = eps

        if method == "forward":
            fp = _as_vector(fun(x + dx), name="fun(x+dx)")
            J[:, j] = (fp - f0) / eps
        else:
            fp = _as_vector(fun(x + dx), name="fun(x+dx)")
            fm = _as_vector(fun(x - dx), name="fun(x-dx)")
            J[:, j] = (fp - fm) / (2.0 * eps)

    return J


def compare_jacobians(
    analytic: Array,
    numeric: Array,
    *,
    name: str = "jacobian",
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> JacobianComparison:
    Ja = _as_matrix(analytic, name="analytic")
    Jn = _as_matrix(numeric, name="numeric")

    if Ja.shape != Jn.shape:
        raise ValueError(f"Shape mismatch: analytic {Ja.shape} vs numeric {Jn.shape}")

    diff = Ja - Jn
    abs_err = np.abs(diff)
    rel_err = _safe_rel_error(Ja, Jn)

    max_abs_error = float(np.max(abs_err))
    max_rel_error = float(np.max(rel_err))
    rms_abs_error = float(np.sqrt(np.mean(abs_err**2)))

    worst_index = tuple(int(i) for i in np.unravel_index(np.argmax(abs_err), abs_err.shape))
    wi, wj = worst_index

    passed = bool(np.allclose(Ja, Jn, atol=atol, rtol=rtol))

    return JacobianComparison(
        name=name,
        passed=passed,
        shape=Ja.shape,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        rms_abs_error=rms_abs_error,
        worst_index=worst_index,
        analytic_value=float(Ja[wi, wj]),
        numeric_value=float(Jn[wi, wj]),
        diff_value=float(diff[wi, wj]),
        atol=float(atol),
        rtol=float(rtol),
        details={
            "abs_error_matrix": abs_err,
            "rel_error_matrix": rel_err,
            "diff_matrix": diff,
        },
    )


def run_jacobian_check(
    *,
    fun: VectorFn,
    analytic_jacobian: JacobianFn,
    x: Array,
    name: str,
    eps: float = 1e-6,
    method: str = "central",
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> JacobianComparison:
    x = _as_vector(x, name="x")
    Ja = _as_matrix(analytic_jacobian(x), name=f"{name}_analytic")
    Jn = numeric_jacobian(fun, x, eps=eps, method=method)
    return compare_jacobians(Ja, Jn, name=name, atol=atol, rtol=rtol)


def check_bearing_measurement_jacobian(
    *,
    x: Array,
    r_body: Array,
    sigma_theta: float,
    u_meas: Optional[Array] = None,
    eps: float = 1e-6,
    method: str = "central",
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> JacobianComparison:
    x = _as_vector(x, name="x")
    r_body = _as_vector(r_body, name="r_body")
    if r_body.size != 3:
        raise ValueError(f"r_body must have shape (3,), got {r_body.shape}")

    if u_meas is None:
        model0 = bearing_measurement_model(
            x=x,
            u_meas=(r_body - x[:3]) / np.linalg.norm(r_body - x[:3]),
            r_body=r_body,
            sigma_theta=sigma_theta,
        )
        u_meas_use = model0.u_pred
    else:
        u_meas_use = _as_vector(u_meas, name="u_meas")
        if u_meas_use.size != 3:
            raise ValueError(f"u_meas must have shape (3,), got {u_meas_use.shape}")

    def residual_fun(x_state: Array) -> Array:
        model = bearing_measurement_model(
            x=x_state,
            u_meas=u_meas_use,
            r_body=r_body,
            sigma_theta=sigma_theta,
        )
        return -model.residual_2d

    def analytic_jac(x_state: Array) -> Array:
        model = bearing_measurement_model(
            x=x_state,
            u_meas=u_meas_use,
            r_body=r_body,
            sigma_theta=sigma_theta,
        )
        return model.H

    return run_jacobian_check(
        fun=residual_fun,
        analytic_jacobian=analytic_jac,
        x=x,
        name="bearing_measurement_H",
        eps=eps,
        method=method,
        atol=atol,
        rtol=rtol,
    )
