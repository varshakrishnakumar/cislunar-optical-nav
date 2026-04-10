from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union, Any, Dict, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp


Array = np.ndarray
DynamicsFn = Callable[[float, Array], Array]
EventFn = Callable[[float, Array], float]


@dataclass
class PropagationResult:
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
        def rhs(t: float, x: Array, *a: Any) -> Array:
            return np.asarray(f(t, x, *a), dtype=float).reshape(-1)

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

    t_out = np.asarray(sol.t, dtype=float)
    if sol.y is None or sol.y.size == 0:
        x_out = np.empty((0, x0_arr.size), dtype=float)
    else:
        x_out = np.asarray(sol.y, dtype=float).T

    t_events = None
    x_events = None
    if sol.t_events is not None:
        t_events = [np.asarray(te, dtype=float) for te in sol.t_events]
    if sol.y_events is not None:
        x_events = []
        for ye in sol.y_events:
            ye_arr = np.atleast_2d(np.asarray(ye, dtype=float))
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
    t_req = np.asarray(times, dtype=float)
    if t_req.ndim == 0:
        t_req = t_req.reshape(1)

    if result.sol is not None:
        y = result.sol(t_req)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            return y.reshape(1, -1)
        return y.T

    if result.t.size == 0 or result.x.size == 0:
        raise ValueError("Cannot sample: result has no stored trajectory. Use dense_output=True or provide t_eval.")

    warnings.warn(
        "sample_at_times: falling back to linear (np.interp) interpolation because "
        "dense_output=False. This discards the 8th-order accuracy of DOP853 at "
        "intermediate times. Pass dense_output=True to propagate() for full accuracy.",
        stacklevel=2,
    )

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

