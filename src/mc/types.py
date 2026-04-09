from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence


CameraMode = Literal["fixed", "truth_tracking", "estimate_tracking"]

_VALID_CAMERA_MODES = ("fixed", "truth_tracking", "estimate_tracking")


@dataclass(frozen=True)
class MonteCarloConfig:

    mu: float
    t0: float
    tf: float
    tc: float
    dt_meas: float

    sigma_px: float
    dropout_prob: float = 0.0
    camera_mode: CameraMode = "estimate_tracking"

    n_trials: int = 100
    base_seed: int = 7

    sigma_r_inj: float = 1e-4
    sigma_v_inj: float = 1e-4

    sigma_r_est: float = 1e-4
    sigma_v_est: float = 1e-4

    planar_only: bool = False
    study_name: str = "mc_study"

    def __post_init__(self) -> None:
        if not (0.0 < self.mu < 0.5):
            raise ValueError(f"mu must be in (0, 0.5), got {self.mu}")
        if self.tf <= self.t0:
            raise ValueError(f"tf ({self.tf}) must be > t0 ({self.t0})")
        if not (self.t0 <= self.tc <= self.tf):
            raise ValueError(
                f"tc ({self.tc}) must satisfy t0 ({self.t0}) <= tc <= tf ({self.tf})"
            )
        if self.dt_meas <= 0.0:
            raise ValueError(f"dt_meas must be > 0, got {self.dt_meas}")
        if self.sigma_px < 0.0:
            raise ValueError(f"sigma_px must be >= 0, got {self.sigma_px}")
        if not (0.0 <= self.dropout_prob <= 1.0):
            raise ValueError(f"dropout_prob must be in [0, 1], got {self.dropout_prob}")
        if self.camera_mode not in _VALID_CAMERA_MODES:
            raise ValueError(
                f"camera_mode must be one of {_VALID_CAMERA_MODES}, got {self.camera_mode!r}"
            )
        if self.n_trials <= 0:
            raise ValueError(f"n_trials must be > 0, got {self.n_trials}")
        for name in ("sigma_r_inj", "sigma_v_inj", "sigma_r_est", "sigma_v_est"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")


@dataclass(frozen=True)
class TrialResult:

    trial_id: int
    seed: int
    tc: float
    sigma_px: float
    dropout_prob: float
    camera_mode: CameraMode

    dv_perfect_mag: float
    dv_ekf_mag: float
    dv_delta_mag: float
    dv_inflation: float
    dv_inflation_pct: float

    miss_uncorrected: float
    miss_perfect: float
    miss_ekf: float

    pos_err_tc: float
    tracePpos_tc: float
    nis_mean: float
    valid_rate: float

    dx0_norm_r: float
    dx0_norm_v: float

    notes: Optional[str] = None
