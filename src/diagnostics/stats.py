from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


Array = np.ndarray


def as_float_array(x: Sequence[float] | Array) -> Array:
    return np.asarray(x, dtype=float).reshape(-1)


def finite_values(x: Sequence[float] | Array) -> Array:
    arr = as_float_array(x)
    return arr[np.isfinite(arr)]


def safe_mean(x: Sequence[float] | Array) -> float:
    vals = finite_values(x)
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def safe_std(x: Sequence[float] | Array, *, ddof: int = 0) -> float:
    vals = finite_values(x)
    if vals.size == 0:
        return float("nan")
    return float(np.std(vals, ddof=ddof))


def safe_median(x: Sequence[float] | Array) -> float:
    vals = finite_values(x)
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def safe_percentile(x: Sequence[float] | Array, q: float) -> float:
    vals = finite_values(x)
    if vals.size == 0:
        return float("nan")
    return float(np.percentile(vals, q))


def safe_min(x: Sequence[float] | Array) -> float:
    vals = finite_values(x)
    if vals.size == 0:
        return float("nan")
    return float(np.min(vals))


def safe_max(x: Sequence[float] | Array) -> float:
    vals = finite_values(x)
    if vals.size == 0:
        return float("nan")
    return float(np.max(vals))


def safe_frac_true(mask: Sequence[bool] | Array) -> float:
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def safe_frac_finite(x: Sequence[float] | Array) -> float:
    arr = as_float_array(x)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(np.isfinite(arr)))


def first_finite(x: Sequence[float] | Array) -> float:
    vals = finite_values(x)
    if vals.size == 0:
        return float("nan")
    return float(vals[0])


def last_finite(x: Sequence[float] | Array) -> float:
    vals = finite_values(x)
    if vals.size == 0:
        return float("nan")
    return float(vals[-1])


def summarize_scalar_series(x: Sequence[float] | Array) -> dict[str, float]:
    arr = as_float_array(x)
    vals = finite_values(arr)

    return {
        "n_total": float(arr.size),
        "n_finite": float(vals.size),
        "finite_rate": safe_frac_finite(arr),
        "mean": safe_mean(vals),
        "std": safe_std(vals),
        "median": safe_median(vals),
        "min": safe_min(vals),
        "max": safe_max(vals),
        "p05": safe_percentile(vals, 5.0),
        "p95": safe_percentile(vals, 95.0),
        "first": first_finite(vals),
        "last": last_finite(vals),
    }


def rms(x: Sequence[float] | Array) -> float:
    vals = finite_values(x)
    if vals.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(vals**2)))


def vector_norm_rows(X: Array) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")
    return np.linalg.norm(X, axis=1)


def summarize_vector_series(X: Array) -> dict[str, float]:
    norms = vector_norm_rows(X)
    return summarize_scalar_series(norms)
