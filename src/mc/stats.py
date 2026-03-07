from __future__ import annotations

from dataclasses import asdict
from typing import List, Dict, Any, Optional
import numpy as np

from .types import TrialResult


def _arr(results: List[TrialResult], field: str) -> np.ndarray:
    return np.array([getattr(r, field) for r in results], dtype=float)


def success_rate(results: List[TrialResult], tol: float) -> float:
    """Fraction of trials with miss_ekf < tol."""
    if not results:
        return float("nan")
    return float(np.mean(_arr(results, "miss_ekf") < float(tol)))


def summarize_results(results: List[TrialResult], tol: Optional[float] = None) -> Dict[str, Any]:
    """Compute summary statistics for a list of TrialResult."""
    if not results:
        return {"n": 0}

    summary: Dict[str, Any] = {"n": len(results)}

    # Common metrics
    for field in [
        "dv_inflation",
        "dv_inflation_pct",
        "miss_ekf",
        "miss_uncorrected",
        "pos_err_tc",
        "tracePpos_tc",
        "nis_mean",
        "valid_rate",
    ]:
        x = _arr(results, field)
        summary[f"{field}_mean"] = float(np.mean(x))
        summary[f"{field}_std"] = float(np.std(x))
        summary[f"{field}_median"] = float(np.median(x))
        summary[f"{field}_p95"] = float(np.percentile(x, 95))

    if tol is not None:
        summary["success_rate"] = success_rate(results, tol)

    # Include config-like identifiers (from the first trial)
    r0 = results[0]
    summary["tc"] = r0.tc
    summary["sigma_px"] = r0.sigma_px
    summary["dropout_prob"] = r0.dropout_prob
    summary["tracking_attitude"] = r0.tracking_attitude

    return summary
