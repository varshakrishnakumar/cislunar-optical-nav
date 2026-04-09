from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


Array = np.ndarray
CameraMode = Literal["fixed", "truth_tracking", "estimate_tracking"]

_DEFAULT_X0_NOM = np.array([1.02, 0.0, 0.0, 0.0, -0.18, 0.0], dtype=float)


@dataclass(frozen=True)
class GatingConfig:
    enabled: bool = True
    probability: float = 0.9973
    measurement_dim: int = 2
    reject_on_nan: bool = True
    preset: str = "baseline"


@dataclass(frozen=True)
class FaultInjectionConfig:
    dropout_prob: float = 0.0
    outlier_prob: float = 0.0
    outlier_sigma_scale: float = 10.0
    measurement_delay_steps: int = 0


@dataclass(frozen=True)
class NoiseConfig:
    sigma_px: float = 1.5
    q_acc: float = 0.0
    p0_diag: tuple[float, float, float, float, float, float] = (
        1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6,
    )


@dataclass(frozen=True)
class CaseConfig:
    mu: float
    t0: float
    tf: float
    dt_meas: float
    seed: int
    dx0: Array
    est_err: Array
    camera_mode: CameraMode = "estimate_tracking"
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    faults: FaultInjectionConfig = field(default_factory=FaultInjectionConfig)
    x0_nom: Array = field(
        default_factory=lambda: _DEFAULT_X0_NOM.copy()
    )

    def __post_init__(self) -> None:
        if not (0.0 < self.mu < 0.5):
            raise ValueError("mu must be in (0, 0.5)")
        if self.tf <= self.t0:
            raise ValueError("tf must be > t0")
        if self.dt_meas <= 0.0:
            raise ValueError("dt_meas must be > 0")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")

        x0_nom = np.asarray(self.x0_nom, dtype=float).reshape(-1)
        if x0_nom.shape != (6,):
            raise ValueError(f"x0_nom must have shape (6,), got {x0_nom.shape}")

        dx0 = np.asarray(self.dx0, dtype=float).reshape(-1)
        est_err = np.asarray(self.est_err, dtype=float).reshape(-1)
        if dx0.shape != (6,):
            raise ValueError(f"dx0 must have shape (6,), got {dx0.shape}")
        if est_err.shape != (6,):
            raise ValueError(f"est_err must have shape (6,), got {est_err.shape}")

        if self.camera_mode not in ("fixed", "truth_tracking", "estimate_tracking"):
            raise ValueError(f"invalid camera_mode: {self.camera_mode!r}")

        if not (0.0 < self.gating.probability < 1.0):
            raise ValueError("gating.probability must be in (0, 1)")
        if self.faults.dropout_prob < 0.0 or self.faults.dropout_prob > 1.0:
            raise ValueError("dropout_prob must be in [0, 1]")
        if self.faults.outlier_prob < 0.0 or self.faults.outlier_prob > 1.0:
            raise ValueError("outlier_prob must be in [0, 1]")
        if self.faults.measurement_delay_steps < 0:
            raise ValueError("measurement_delay_steps must be >= 0")
        if any(v <= 0.0 for v in self.noise.p0_diag):
            raise ValueError("all p0_diag entries must be > 0")


@dataclass(frozen=True)
class OutputConfig:
    root_dir: Path = Path("results/diagnostics/06_ekf")
    save_plots: bool = True
    save_npz: bool = True
    save_json_summary: bool = True

    def case_dir(self, camera_mode: CameraMode) -> Path:
        return self.root_dir / camera_mode
