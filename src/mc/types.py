from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class MonteCarloConfig:
    """Configuration for a Monte Carlo study around the 06A single-run pipeline.

    Notes:
      - sigma_px and tc can be single values (for MC) or grids (for sweeps) handled by scripts.
      - tracking_attitude=True means you use time-varying camera attitude (R_cam_from_frame varies).
        tracking_attitude=False means fixed pointing; the 06A run_case should support this via
        fixed_camera_pointing=True.
    """

    # dynamics / timeline
    mu: float
    t0: float
    tf: float
    tc: float
    dt_meas: float

    # measurement model
    sigma_px: float
    dropout_prob: float = 0.0
    tracking_attitude: bool = True  # False => fixed camera pointing (bonus case)

    # trial control
    n_trials: int = 100
    base_seed: int = 7

    # injection (truth) error distribution (1-sigma, in nondimensional units)
    sigma_r_inj: float = 1e-4
    sigma_v_inj: float = 1e-4

    # filter initial estimation error distribution (1-sigma, in nondimensional units)
    sigma_r_est: float = 1e-4
    sigma_v_est: float = 1e-4

    planar_only: bool = False

    study_name: str = "mc06c"


@dataclass(frozen=True)
class TrialInput:
    trial_id: int
    seed: int
    dx0: Sequence[float]          # 6-vector
    est_err: Sequence[float]      # 6-vector


@dataclass(frozen=True)
class TrialResult:
    # identifiers
    trial_id: int
    seed: int
    tc: float
    sigma_px: float
    dropout_prob: float
    tracking_attitude: bool

    # burn comparison
    dv_perfect_mag: float
    dv_ekf_mag: float
    dv_delta_mag: float
    dv_inflation: float
    dv_inflation_pct: float

    # terminal miss
    miss_uncorrected: float
    miss_perfect: float
    miss_ekf: float

    # EKF stats at correction time
    pos_err_tc: float
    tracePpos_tc: float
    nis_mean: float
    valid_rate: float

    # injection size
    dx0_norm_r: float
    dx0_norm_v: float

    notes: Optional[str] = None
