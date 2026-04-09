from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .types import CameraMode, TrialResult


def trial_result_from_run_case(
    *,
    trial_id: int,
    seed: int,
    tc: float,
    sigma_px: float,
    dropout_prob: float,
    camera_mode: str,
    dx0: np.ndarray,
    out: Dict[str, Any],
) -> TrialResult:
    dv_perfect_mag = float(out["dv_perfect_mag"])
    dv_ekf_mag     = float(out["dv_ekf_mag"])
    dv_delta_mag   = float(out["dv_delta_mag"])

    dv_inflation = float(
        out["dv_inflation"]
        if "dv_inflation" in out
        else dv_ekf_mag - dv_perfect_mag
    )

    if "dv_inflation_pct" in out:
        dv_inflation_pct = float(out["dv_inflation_pct"])
    elif dv_perfect_mag == 0.0:
        dv_inflation_pct = float("nan")
    else:
        dv_inflation_pct = dv_ekf_mag / dv_perfect_mag - 1.0

    miss_uncorrected = float(out["miss_uncorrected"])
    miss_perfect     = float(out["miss_perfect"])
    miss_ekf         = float(out["miss_ekf"])

    pos_err_tc   = float(out["pos_err_tc"])
    tracePpos_tc = float(out["tracePpos_tc"])
    nis_mean     = float(out["nis_mean"])
    valid_rate   = float(out["valid_rate"])

    dx0 = np.asarray(dx0, dtype=float).reshape(6)
    dx0_norm_r = float(np.linalg.norm(dx0[:3]))
    dx0_norm_v = float(np.linalg.norm(dx0[3:]))

    _notes = out.get("notes")
    notes: Optional[str] = str(_notes) if _notes is not None else None

    return TrialResult(
        trial_id=trial_id,
        seed=seed,
        tc=float(tc),
        sigma_px=float(sigma_px),
        dropout_prob=float(dropout_prob),
        camera_mode=str(camera_mode),
        dv_perfect_mag=dv_perfect_mag,
        dv_ekf_mag=dv_ekf_mag,
        dv_delta_mag=dv_delta_mag,
        dv_inflation=dv_inflation,
        dv_inflation_pct=dv_inflation_pct,
        miss_uncorrected=miss_uncorrected,
        miss_perfect=miss_perfect,
        miss_ekf=miss_ekf,
        pos_err_tc=pos_err_tc,
        tracePpos_tc=tracePpos_tc,
        nis_mean=nis_mean,
        valid_rate=valid_rate,
        dx0_norm_r=dx0_norm_r,
        dx0_norm_v=dx0_norm_v,
        notes=notes,
    )
