from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np


Array = np.ndarray
Severity = Literal["info", "warning", "failure"]


@dataclass(frozen=True)
class HealthRecord:
    name: str
    shape: tuple[int, ...]
    is_finite: bool
    symmetry_error_fro: float
    min_eig: float
    max_eig: float
    cond: float
    is_spd: bool
    chol_ok: bool


@dataclass(frozen=True)
class GateRecord:
    enabled: bool
    accepted: bool
    threshold: float
    statistic: float
    probability: float
    dof: int
    reason: str


@dataclass
class UpdateRecord:
    t: float
    valid_measurement: bool
    update_used: bool

    gate: Optional[GateRecord] = None

    innovation: Optional[Array] = None
    nis: float = float("nan")
    sigma_theta: float = float("nan")

    u_meas: Optional[Array] = None
    u_pred: Optional[Array] = None
    pixel_uv: Optional[Array] = None

    H: Optional[Array] = None
    R: Optional[Array] = None
    S: Optional[Array] = None
    K: Optional[Array] = None

    measurement_meta: Optional[dict[str, Any]] = None
    S_health: Optional[HealthRecord] = None
    pixel_uv_pred: Optional[Array] = None
    u_meas_cam: Optional[Array] = None
    u_pred_cam: Optional[Array] = None


@dataclass
class RunSummary:
    camera_mode: str
    num_steps: int
    valid_rate: float
    update_rate: float
    gate_accept_rate: float
    nis_mean: float
    nees_minus_mean: float
    nees_plus_mean: float
    final_pos_err: float
    final_vel_err: float
    final_los_angle: float


@dataclass
class RunTrace:
    t_meas: Array
    x_true_hist: Array
    xhat_minus_hist: Array
    xhat_plus_hist: Array

    P_minus_hist: Array
    P_plus_hist: Array
    Phi_hist: Array

    err_minus_hist: Array
    err_plus_hist: Array

    nees_minus_hist: Array
    nees_plus_hist: Array

    los_true_hist: Array
    los_est_hist: Array
    los_angle_hist: Array

    camera_R_hist: Array

    updates: list[UpdateRecord] = field(default_factory=list)
    P_minus_health: list[Optional[HealthRecord]] = field(default_factory=list)
    P_plus_health: list[Optional[HealthRecord]] = field(default_factory=list)


@dataclass
class RunResult:
    config: dict[str, Any]
    summary: RunSummary
    trace: RunTrace


@dataclass(frozen=True)
class HypothesisResult:
    name: str
    passed: bool
    severity: Severity
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SweepRow:
    case_id: str
    seed: int
    camera_mode: str
    sigma_px: float
    q_acc: float
    dropout_prob: float
    outlier_prob: float
    measurement_delay_steps: int
    gate_probability: float
    valid_rate: float
    update_rate: float
    gate_accept_rate: float
    nis_mean: float
    nees_minus_mean: float
    nees_plus_mean: float
    final_pos_err: float
    final_vel_err: float
    final_los_angle: float
    diverged: bool = False
