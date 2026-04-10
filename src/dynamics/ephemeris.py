from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Callable

import numpy as np


Array = np.ndarray
PositionFn = Callable[[float], Array]

GM_EARTH_KM3_S2 = 398600.435436
GM_MOON_KM3_S2 = 4902.800066


@dataclass(frozen=True)
class PointMassBody:
    name: str
    gm_km3_s2: float
    position_km: PositionFn


@dataclass(frozen=True)
class CircularEarthMoonEphemeris:
    """Circular barycentric Earth-Moon ephemeris for CR3BP-to-inertial bridging."""

    distance_km: float
    time_unit_s: float
    mass_ratio: float
    theta0_rad: float = 0.0
    gm_earth_km3_s2: float | None = None
    gm_moon_km3_s2: float | None = None

    @property
    def angular_rate_rad_s(self) -> float:
        return 1.0 / self.time_unit_s

    @property
    def gm_total_km3_s2(self) -> float:
        return self.distance_km**3 / self.time_unit_s**2

    @property
    def earth_gm_km3_s2(self) -> float:
        if self.gm_earth_km3_s2 is not None:
            return float(self.gm_earth_km3_s2)
        return (1.0 - self.mass_ratio) * self.gm_total_km3_s2

    @property
    def moon_gm_km3_s2(self) -> float:
        if self.gm_moon_km3_s2 is not None:
            return float(self.gm_moon_km3_s2)
        return self.mass_ratio * self.gm_total_km3_s2

    def _unit_vector(self, t_s: float) -> Array:
        theta = self.theta0_rad + self.angular_rate_rad_s * float(t_s)
        return np.array([cos(theta), sin(theta), 0.0], dtype=float)

    def earth_position_km(self, t_s: float) -> Array:
        return -self.mass_ratio * self.distance_km * self._unit_vector(t_s)

    def moon_position_km(self, t_s: float) -> Array:
        return (1.0 - self.mass_ratio) * self.distance_km * self._unit_vector(t_s)

    def bodies(self) -> tuple[PointMassBody, PointMassBody]:
        return (
            PointMassBody("Earth", self.earth_gm_km3_s2, self.earth_position_km),
            PointMassBody("Moon", self.moon_gm_km3_s2, self.moon_position_km),
        )
