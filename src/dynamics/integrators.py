from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union, Any, Dict, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp


Array = np.ndarray
DynamicsFn = Callable[[float, Array], Array]
EventFn = Callable[[float, Array], float]


@dataclass
class PropagationResult:
    """
    Standard result object we control.

    Attributes
    ----------
    t : (N,) ndarray
        Times returned by the integrator (always increasing or decreasing with integration direction).
    x : (N, n) ndarray
        States at those times.
    success : bool
        Whether the integrator reported success.
    message : str
        SciPy's message or our own.
    nfev, njev, nlu : int | None
        Diagnostics from SciPy (if available).
    t_events : list[ndarray] | None
        Event trigger times per event function, if events were provided.
    x_events : list[ndarray] | None
        States at event times per event function, if events were provided.
    sol : Callable[[float | ndarray], ndarray] | None
        Dense output callable if dense_output=True, else None.
        SciPy returns shape (n, m) for m evaluation points; we standardize in helpers.
    raw : Any
        The raw SciPy result object for debugging / advanced use.
    """
    t: Array
    x: Array
    success: bool
    message: str
    nfev: Optional[int] = None
    njev: Optional[int] = None
    nlu: Optional[int] = None
    t_events: Optional[List[Array]] = None
    x_events: Optional[List[Array]] = None
    sol: Optional[Callable[[Union[float, Array]], Array]] = None
    raw: Any = None


def propagate(
    f: DynamicsFn,
    t_span: Tuple[float, float],
    x0: Union[Sequence[float], Array],
    *,
    t_eval: Optional[Union[Sequence[float], Array]] = None,
    events: Optional[Union[EventFn, Sequence[EventFn]]] = None,
    dense_output: bool = False,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: Union[float, Array] = 1e-12,
    max_step: float = np.inf,
    vectorized: bool = False,
    args: Optional[tuple] = None,
    **solve_ivp_kwargs: Any,
) -> PropagationResult:
    """
    Integrate x' = f(t, x) using SciPy solve_ivp, returning a standardized result.

    Parameters
    ----------
    f : callable
        Dynamics function f(t, x) -> dx/dt, with x a 1D ndarray shape (n,).
    t_span : (t0, tf)
        Integration interval.
    x0 : array-like
        Initial state (n,).
    t_eval : array-like, optional
        Times at which to store the computed solution. If None, uses solver internal steps.
    events : callable or list of callables, optional
        Event functions g(t, x) -> float. Passed through to solve_ivp.
        You can set attributes on event functions:
            - event.terminal = True/False
            - event.direction = -1/0/+1
    dense_output : bool
        If True, request dense output (continuous solution interpolation).
    method : str
        solve_ivp method, default "DOP853" (good for smooth problems).
    rtol, atol : float or ndarray
        Integration tolerances.
    max_step : float
        Maximum step size.
    vectorized : bool
        If True, indicates f can accept x as shape (n, m) and return same.
    args : tuple, optional
        Extra arguments passed to f and events (SciPy convention).
    solve_ivp_kwargs :
        Any other solve_ivp kwargs.

    Returns
    -------
    PropagationResult
    """
    x0_arr = np.asarray(x0, dtype=float).reshape(-1)

    t_eval_arr: Optional[Array]
    if t_eval is None:
        t_eval_arr = None
    else:
        t_eval_arr = np.asarray(t_eval, dtype=float)


    if args is None:
        def rhs(t: float, x: Array) -> Array:
            return np.asarray(f(t, x), dtype=float).reshape(-1)
    else:
        def rhs(t: float, x: Array) -> Array:
            return np.asarray(f(t, x, *args), dtype=float).reshape(-1)

    # Events: pass through, but if args is provided, SciPy expects event(t, y, *args).
    events_pass = events

    sol = solve_ivp(
        rhs,
        (float(t_span[0]), float(t_span[1])),
        x0_arr,
        method=method,
        t_eval=t_eval_arr,
        events=events_pass,
        dense_output=dense_output,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        vectorized=vectorized,
        args=args,
        **solve_ivp_kwargs,
    )

    # SciPy returns y as (n, N). We standardize to (N, n).
    t_out = np.asarray(sol.t, dtype=float)
    if sol.y is None or sol.y.size == 0:
        x_out = np.empty((0, x0_arr.size), dtype=float)
    else:
        x_out = np.asarray(sol.y, dtype=float).T

    # Events (if any)
    t_events = None
    x_events = None
    if sol.t_events is not None:
        t_events = [np.asarray(te, dtype=float) for te in sol.t_events]
    if sol.y_events is not None:

        x_events = []
        for ye in sol.y_events:
            ye_arr = np.asarray(ye, dtype=float)
            if ye_arr.ndim == 2 and ye_arr.shape[0] == x0_arr.size:
                # likely (n, k) -> transpose
                ye_arr = ye_arr.T
            x_events.append(ye_arr)

    return PropagationResult(
        t=t_out,
        x=x_out,
        success=bool(sol.success),
        message=str(sol.message),
        nfev=getattr(sol, "nfev", None),
        njev=getattr(sol, "njev", None),
        nlu=getattr(sol, "nlu", None),
        t_events=t_events,
        x_events=x_events,
        sol=sol.sol if dense_output else None,
        raw=sol,
    )


def sample_at_times(result: PropagationResult, times: Union[Sequence[float], Array]) -> Array:
    """
    Sample states at requested times using dense output if available, otherwise interpolation.

    Parameters
    ----------
    result : PropagationResult
        Output from propagate().
    times : array-like
        Times to sample.

    Returns
    -------
    X : (M, n) ndarray
        States evaluated at each requested time.
    """
    t_req = np.asarray(times, dtype=float)
    if t_req.ndim == 0:
        t_req = t_req.reshape(1)

    if result.sol is not None:
        y = result.sol(t_req)  # SciPy returns shape (n, M)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            return y.reshape(1, -1)
        return y.T  # (M, n)

    if result.t.size == 0 or result.x.size == 0:
        raise ValueError("Cannot sample: result has no stored trajectory. Use dense_output=True or provide t_eval.")

    t = result.t
    X = result.x
    n = X.shape[1]

    t_min, t_max = (t[0], t[-1]) if t[0] <= t[-1] else (t[-1], t[0])
    if np.any(t_req < t_min) or np.any(t_req > t_max):
        raise ValueError("Requested sample times fall outside the integration interval. Use dense_output=True or re-propagate.")


    if t[0] > t[-1]:
        t_mono = t[::-1]
        X_mono = X[::-1, :]
    else:
        t_mono = t
        X_mono = X

    out = np.empty((t_req.size, n), dtype=float)
    for i in range(n):
        out[:, i] = np.interp(t_req, t_mono, X_mono[:, i])

    return out


def sample_at_time(result: PropagationResult, time: float) -> Array:
    """
    Sample state at a single time. Returns shape (n,).
    """
    return sample_at_times(result, np.array([time], dtype=float))[0]
