from __future__ import annotations

import numpy as np
from typing import Dict, Any

from .types import TrialResult


def trial_result_from_run_case(
    *,
    trial_id: int,
    seed: int,
    tc: float,
    sigma_px: float,
    dropout_prob: float,
    tracking_attitude: bool,
    dx0: np.ndarray,
    out: Dict[str, Any],
) -> TrialResult:
    """Convert the output dict of 06A `run_case` into a typed TrialResult."""

    # 06A outputs we expect (scalar metrics)
    dv_perfect_mag = float(out["dv_perfect_mag"])
    dv_ekf_mag = float(out["dv_ekf_mag"])
    dv_delta_mag = float(out["dv_delta_mag"])
    dv_inflation = float(out.get("dv_inflation", dv_ekf_mag - dv_perfect_mag))
    dv_inflation_pct = float(out.get("dv_inflation_pct", dv_ekf_mag / dv_perfect_mag - 1.0))

    miss_unc = float(out["miss_uncorrected"])
    miss_perf = float(out["miss_perfect"])
    miss_ekf = float(out["miss_ekf"])

    pos_err_tc = float(out["pos_err_tc"])
    tracePpos_tc = float(out["tracePpos_tc"])
    nis_mean = float(out["nis_mean"])
    valid_rate = float(out["valid_rate"])

    dx0 = np.asarray(dx0, dtype=float).reshape(6,)
    dx0_norm_r = float(np.linalg.norm(dx0[:3]))
    dx0_norm_v = float(np.linalg.norm(dx0[3:]))

    return TrialResult(
        trial_id=trial_id,
        seed=seed,
        tc=float(tc),
        sigma_px=float(sigma_px),
        dropout_prob=float(dropout_prob),
        tracking_attitude=bool(tracking_attitude),
        dv_perfect_mag=dv_perfect_mag,
        dv_ekf_mag=dv_ekf_mag,
        dv_delta_mag=dv_delta_mag,
        dv_inflation=dv_inflation,
        dv_inflation_pct=dv_inflation_pct,
        miss_uncorrected=miss_unc,
        miss_perfect=miss_perf,
        miss_ekf=miss_ekf,
        pos_err_tc=pos_err_tc,
        tracePpos_tc=tracePpos_tc,
        nis_mean=nis_mean,
        valid_rate=valid_rate,
        dx0_norm_r=dx0_norm_r,
        dx0_norm_v=dx0_norm_v,
        notes=str(out.get("notes")) if out.get("notes") is not None else None,
    )
