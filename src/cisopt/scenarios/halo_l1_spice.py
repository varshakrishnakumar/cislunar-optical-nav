"""SPICE/JPL-ephemeris halo-L1 scenario (item 12: multi-fidelity toggle).

Mirrors the CR3BP halo-L1 scenario but with point-mass dynamics driven by
DE442 ephemerides for Sun/Earth/Moon. Times are real seconds; lengths are km
in J2000. Conversion from the dimensionless CR3BP halo seed to inertial km
uses the synodic-frame bridge already in ``orbits.spice_bridge``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dynamics.cr3bp import CR3BP
from dynamics.integrators import propagate
from dynamics.spice_ephemeris import (
    GM_EARTH_KM3_S2,
    GM_MOON_KM3_S2,
    make_spice_point_mass_dynamics,
)
from orbits.conversion import normalized_to_dimensional_state
from orbits.spice_bridge import (
    dimensional_synodic_to_spice_inertial_state,
    earth_moon_synodic_frame_from_spice,
)
from orbits.types import CR3BPSystemUnits

from ..protocols import StateEstimate


Array = np.ndarray
_DEFAULT_KERNELS = (
    "data/kernels/naif0012.tls",
    "data/kernels/de442s.bsp",
)
_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class HaloL1SpiceScenario:
    name: str = "halo_l1_spice"
    mu: float = 0.0121505856
    epoch: str = "2026 APR 10 00:00:00 TDB"

    duration_days: float = 6.5
    correction_at_days: float = 2.5
    dt_meas_min: float = 30.0

    kernels: tuple[str, ...] = _DEFAULT_KERNELS
    targets: tuple[str, ...] = ("SUN", "EARTH", "MOON")

    nominal_offset_nd: tuple[float, ...] = (-1e-3, 0.0, 0.0, 0.0, 0.05, 0.0)
    target_body: str = "Moon"

    P0_pos_var_km2: float | None = None
    P0_vel_var_km2_s2: float | None = None

    t0_s: float = field(init=False)
    tf_s: float = field(init=False)
    tc_s: float = field(init=False)
    dt_meas_s: float = field(init=False)

    dynamics: Any = field(init=False)
    _ephemeris: Any = field(init=False, repr=False)
    _x0_nom_km: Array | None = field(default=None, init=False, repr=False)
    _r_target_km: Array | None = field(default=None, init=False, repr=False)
    _lunit_km: float = field(init=False, default=0.0)
    _tunit_s: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        kernel_paths = [
            (Path(k) if Path(k).is_absolute() else _REPO_ROOT / k)
            for k in self.kernels
        ]
        ephemeris, dynamics = make_spice_point_mass_dynamics(
            kernels=kernel_paths,
            epoch=self.epoch,
            targets=list(self.targets),
        )
        object.__setattr__(self, "_ephemeris", ephemeris)
        object.__setattr__(self, "dynamics", dynamics)

        # CR3BP scale factors derived from SPICE at epoch (matches run_case_spice).
        r_earth_0 = ephemeris.position_km("EARTH", 0.0)
        r_moon_0 = ephemeris.position_km("MOON", 0.0)
        lunit = float(np.linalg.norm(r_moon_0 - r_earth_0))
        tunit = float(np.sqrt(lunit ** 3 / (GM_EARTH_KM3_S2 + GM_MOON_KM3_S2)))
        object.__setattr__(self, "_lunit_km", lunit)
        object.__setattr__(self, "_tunit_s", tunit)

        object.__setattr__(self, "t0_s", 0.0)
        object.__setattr__(self, "tf_s", float(self.duration_days) * 86400.0)
        object.__setattr__(self, "tc_s", float(self.correction_at_days) * 86400.0)
        object.__setattr__(self, "dt_meas_s", float(self.dt_meas_min) * 60.0)

        if self.P0_pos_var_km2 is None:
            object.__setattr__(self, "P0_pos_var_km2", 1e-6 * lunit ** 2)
        if self.P0_vel_var_km2_s2 is None:
            vunit = lunit / tunit
            object.__setattr__(self, "P0_vel_var_km2_s2", 1e-7 * vunit ** 2)

    def _x0_nominal_km(self) -> Array:
        if self._x0_nom_km is None:
            cr3bp_model = CR3BP(mu=float(self.mu))
            L1x = cr3bp_model.lagrange_points()["L1"][0]
            offset = np.asarray(self.nominal_offset_nd, dtype=float).reshape(6)
            x0_nd = np.array([L1x, 0.0, 0.0, 0.0, 0.0, 0.0]) + offset

            system = CR3BPSystemUnits(
                name="earth-moon-spice",
                mass_ratio=float(self.mu),
                radius_secondary_km=1737.4,
                lunit_km=float(self._lunit_km),
                tunit_s=float(self._tunit_s),
                libration_points={},
            )
            synodic = earth_moon_synodic_frame_from_spice(
                self._ephemeris, t_s=self.t0_s, mass_ratio=float(self.mu),
            )
            x0_dim = normalized_to_dimensional_state(tuple(x0_nd.tolist()), system)
            x0_km = np.asarray(
                dimensional_synodic_to_spice_inertial_state(x0_dim, synodic),
                dtype=float,
            )
            object.__setattr__(self, "_x0_nom_km", x0_km)
        return self._x0_nom_km.copy()

    def initial_truth(self, *, dx0: Array | None = None) -> Array:
        x0 = self._x0_nominal_km()
        if dx0 is None:
            return x0
        # Treat dx0 as ND CR3BP error so the same dx0 sample seeds CR3BP and
        # SPICE runs identically (matches run_case_spice convention).
        err_nd = np.asarray(dx0, dtype=float).reshape(6)
        err_km = err_nd.copy()
        err_km[:3] *= self._lunit_km
        err_km[3:] *= self._lunit_km / self._tunit_s
        return x0 + err_km

    def initial_estimate(self, *, est_err: Array | None = None) -> StateEstimate:
        x0 = self._x0_nominal_km()
        if est_err is not None:
            err_nd = np.asarray(est_err, dtype=float).reshape(6)
            err_km = err_nd.copy()
            err_km[:3] *= self._lunit_km
            err_km[3:] *= self._lunit_km / self._tunit_s
            x0 = x0 + err_km
        P = np.diag(np.array(
            [self.P0_pos_var_km2] * 3 + [self.P0_vel_var_km2_s2] * 3,
            dtype=float,
        ))
        return StateEstimate(t_s=float(self.t0_s), x=x0, P=P)

    def target_position(self) -> Array:
        if self._r_target_km is None:
            res = propagate(
                self.dynamics.eom,
                (self.t0_s, self.tf_s),
                self._x0_nominal_km(),
                t_eval=np.linspace(self.t0_s, self.tf_s, 2001),
                rtol=1e-10,
                atol=1e-12,
            )
            if not res.success:
                raise RuntimeError(f"SPICE nominal propagation failed: {res.message}")
            object.__setattr__(self, "_r_target_km", res.x[-1, :3].copy())
        return self._r_target_km.copy()

    def body_position(self, body: str, t_s: float) -> Array:
        return self._ephemeris.position_km(body.upper(), float(t_s))

    def units(self) -> dict[str, str]:
        return {"length": "km (J2000)", "time": "s", "velocity": "km/s"}


def build_halo_l1_spice(params: dict[str, Any]) -> HaloL1SpiceScenario:
    p = dict(params)
    if "kernels" in p:
        p["kernels"] = tuple(p["kernels"])
    if "targets" in p:
        p["targets"] = tuple(p["targets"])
    if "nominal_offset_nd" in p:
        p["nominal_offset_nd"] = tuple(p["nominal_offset_nd"])
    return HaloL1SpiceScenario(**p)
