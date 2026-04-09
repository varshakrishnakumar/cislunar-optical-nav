from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .types import TrialResult





def _field(results: List[TrialResult], field: str) -> np.ndarray:
    return np.array([getattr(r, field) for r in results], dtype=float)


def _nanstat(x: np.ndarray, fn: str, **kwargs: Any) -> float:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("nan")
    return float(getattr(np, fn)(finite, **kwargs))


def _summary_block(x: np.ndarray) -> Dict[str, float]:
    finite = x[np.isfinite(x)]
    n_nan = int(np.sum(~np.isfinite(x)))
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "std":  float("nan"),
            "median": float("nan"),
            "p05": float("nan"),
            "p95": float("nan"),
            "n_finite": 0,
            "n_nan": n_nan,
        }
    return {
        "mean":     float(np.mean(finite)),
        "std":      float(np.std(finite, ddof=1)),
        "median":   float(np.median(finite)),
        "p05":      float(np.percentile(finite, 5)),
        "p95":      float(np.percentile(finite, 95)),
        "n_finite": int(finite.size),
        "n_nan":    n_nan,
    }





def success_rate(results: List[TrialResult], tol: float) -> float:
    if not results:
        return float("nan")
    miss = _field(results, "miss_ekf")
    return float(np.mean(miss < float(tol)))


def summarize_results(
    results: List[TrialResult],
    tol: Optional[float] = None,
) -> Dict[str, Any]:
    if not results:
        return {"n": 0}

    summary: Dict[str, Any] = {"n": len(results)}

    r0 = results[0]
    summary["tc"]           = r0.tc
    summary["sigma_px"]     = r0.sigma_px
    summary["dropout_prob"] = r0.dropout_prob
    summary["camera_mode"]  = r0.camera_mode

    _metrics = [
        "dv_inflation",
        "dv_inflation_pct",
        "miss_ekf",
        "miss_uncorrected",
        "miss_perfect",
        "pos_err_tc",
        "tracePpos_tc",
        "nis_mean",
        "valid_rate",
        "dx0_norm_r",
        "dx0_norm_v",
    ]
    for field in _metrics:
        summary[field] = _summary_block(_field(results, field))

    if tol is not None:
        summary["success_rate"] = success_rate(results, tol)

    return summary
