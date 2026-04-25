"""CR3BP halo-L1 baseline scenario from scripts/06_midcourse_ekf_correction.py.

Pulled out so it can be invoked from the new config-driven runner. Wraps the
existing CR3BP machinery; no algorithm changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dynamics.cr3bp import CR3BP
from dynamics.integrators import propagate
from dynamics.models import CR3BPDynamics

from ..protocols import StateEstimate


Array = np.ndarray


@dataclass
class HaloL1CR3BPScenario:
    name: str = "halo_l1_cr3bp"
    mu: float = 0.0121505856
    t0_s: float = 0.0
    tf_s: float = 6.0
    tc_s: float = 2.0
    dt_meas_s: float = 0.02
    P0_diag: tuple[float, ...] = (1e-6, 1e-6, 1e-6, 1e-7, 1e-7, 1e-7)
    target_body: str = "Moon"
    nominal_offset: tuple[float, ...] = (-1e-3, 0.0, 0.0, 0.0, 0.05, 0.0)

    dynamics: CR3BPDynamics = field(init=False)
    _cr3bp: CR3BP = field(init=False)
    _r_target: Array | None = field(default=None, init=False, repr=False)
    _x0_nom: Array | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.dynamics = CR3BPDynamics(mu=float(self.mu))
        self._cr3bp = CR3BP(mu=float(self.mu))

    def _x0_nominal(self) -> Array:
        if self._x0_nom is None:
            L1x = self._cr3bp.lagrange_points()["L1"][0]
            offset = np.asarray(self.nominal_offset, dtype=float).reshape(6)
            x0 = np.array([L1x, 0.0, 0.0, 0.0, 0.0, 0.0]) + offset
            self._x0_nom = x0
        return self._x0_nom.copy()

    def initial_truth(self, *, dx0: Array | None = None) -> Array:
        x0 = self._x0_nominal()
        if dx0 is None:
            return x0
        return x0 + np.asarray(dx0, dtype=float).reshape(6)

    def initial_estimate(self, *, est_err: Array | None = None) -> StateEstimate:
        x0 = self._x0_nominal()
        if est_err is not None:
            x0 = x0 + np.asarray(est_err, dtype=float).reshape(6)
        P = np.diag(np.asarray(self.P0_diag, dtype=float))
        return StateEstimate(t_s=float(self.t0_s), x=x0, P=P)

    def target_position(self) -> Array:
        if self._r_target is None:
            res = propagate(
                self.dynamics.eom,
                (float(self.t0_s), float(self.tf_s)),
                self._x0_nominal(),
                t_eval=np.linspace(self.t0_s, self.tf_s, 2001),
                rtol=1e-11,
                atol=1e-13,
            )
            if not res.success:
                raise RuntimeError(f"Nominal CR3BP propagation failed: {res.message}")
            self._r_target = res.x[-1, :3].copy()
        return self._r_target.copy()

    def body_position(self, body: str, t_s: float) -> Array:
        b = body.strip().lower()
        if b in ("moon", "m2", "secondary"):
            return np.array([1.0 - float(self.mu), 0.0, 0.0], dtype=float)
        if b in ("earth", "m1", "primary"):
            return np.array([-float(self.mu), 0.0, 0.0], dtype=float)
        raise KeyError(f"Unknown body {body!r} for CR3BP scenario.")

    def units(self) -> dict[str, str]:
        return {"length": "DU (CR3BP)", "time": "TU (CR3BP)", "velocity": "DU/TU"}


def build_halo_l1_cr3bp(params: dict[str, Any]) -> HaloL1CR3BPScenario:
    if "P0_diag" in params:
        params = dict(params)
        params["P0_diag"] = tuple(params["P0_diag"])
    if "nominal_offset" in params:
        params = dict(params)
        params["nominal_offset"] = tuple(params["nominal_offset"])
    return HaloL1CR3BPScenario(**params)
