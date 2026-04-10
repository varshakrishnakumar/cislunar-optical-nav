from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


Vector3 = tuple[float, float, float]
State6 = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class CR3BPSystemUnits:
    name: str
    mass_ratio: float
    radius_secondary_km: float
    lunit_km: float
    tunit_s: float
    libration_points: dict[str, Vector3]


@dataclass(frozen=True)
class PeriodicOrbitRecord:
    system: CR3BPSystemUnits
    family: str
    libration_point: Optional[int]
    branch: Optional[str]
    state_norm: State6
    jacobi: float
    period: float
    stability: float

    @property
    def period_seconds(self) -> float:
        return self.period * self.system.tunit_s

    @property
    def period_days(self) -> float:
        return self.period_seconds / 86400.0


@dataclass(frozen=True)
class PeriodicOrbitCatalog:
    signature_version: str
    system: CR3BPSystemUnits
    family: str
    libration_point: Optional[int]
    branch: Optional[str]
    records: list[PeriodicOrbitRecord]
    filters: dict[str, object]
    limits: dict[str, object]
