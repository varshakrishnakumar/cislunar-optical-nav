"""NRHO (Near-Rectilinear Halo) CR3BP scenario for the scenario library.

Item 11 of the refactor brief asks for reusable scenarios beyond the L1 halo.
NRHOs are operationally relevant (Gateway is in one) and have very different
optical-nav geometry: long quiet arcs followed by fast perilune passes, with
the spacecraft far above the Earth-Moon line for most of the orbit.

The default seed is a cached JPL L2-S halo (period ≈ 1.80 TU ≈ 7.8 days,
Jacobi ≈ 3.029). Users can pass any ND state via ``state_nd``.
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

DEFAULT_NRHO_L2S_STATE_ND = (
    1.0436659621147042,
    0.0,
    -0.1938548765603215,
    0.0,
    -0.1458145571897725,
    0.0,
)
DEFAULT_NRHO_L2S_PERIOD_TU = 1.80288


@dataclass
class NRHOCR3BPScenario:
    name: str = "nrho_cr3bp"
    mu: float = 0.0121505856

    state_nd: tuple[float, ...] = DEFAULT_NRHO_L2S_STATE_ND
    period_tu: float = DEFAULT_NRHO_L2S_PERIOD_TU

    span_periods: float = 0.50
    correction_at_period_frac: float = 0.40
    dt_meas_periods: float = 0.005

    P0_diag: tuple[float, ...] = (1e-6, 1e-6, 1e-6, 1e-7, 1e-7, 1e-7)
    target_body: str = "Moon"

    t0_s: float = field(init=False)
    tf_s: float = field(init=False)
    tc_s: float = field(init=False)
    dt_meas_s: float = field(init=False)

    dynamics: CR3BPDynamics = field(init=False)
    _r_target: Array | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.dynamics = CR3BPDynamics(mu=float(self.mu))
        T = float(self.period_tu)
        # Stored as "_s" by Scenario protocol convention; for CR3BP they are
        # dimensionless TU.
        self.t0_s = 0.0
        self.tf_s = float(self.span_periods) * T
        self.tc_s = float(self.correction_at_period_frac) * float(self.span_periods) * T
        self.dt_meas_s = float(self.dt_meas_periods) * T

        if not (0.0 < self.tc_s < self.tf_s):
            raise ValueError(
                f"NRHOCR3BPScenario: tc={self.tc_s} must lie strictly in (0, tf={self.tf_s})"
            )

    def _x0_nominal(self) -> Array:
        return np.asarray(self.state_nd, dtype=float).reshape(6).copy()

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
                t_eval=np.linspace(self.t0_s, self.tf_s, 4001),
                rtol=1e-11,
                atol=1e-13,
            )
            if not res.success:
                raise RuntimeError(f"NRHO nominal propagation failed: {res.message}")
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


def build_nrho_cr3bp(params: dict[str, Any]) -> NRHOCR3BPScenario:
    p = dict(params)
    if "state_nd" in p:
        p["state_nd"] = tuple(p["state_nd"])
    if "P0_diag" in p:
        p["P0_diag"] = tuple(p["P0_diag"])
    return NRHOCR3BPScenario(**p)
