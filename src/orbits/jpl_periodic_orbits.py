from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlencode
from urllib.request import urlopen

from .types import CR3BPSystemUnits, PeriodicOrbitCatalog, PeriodicOrbitRecord


BASE_URL = "https://ssd-api.jpl.nasa.gov/periodic_orbits.api"
SUPPORTED_SIGNATURE_VERSION = "1.0"


@dataclass(frozen=True)
class PeriodicOrbitQuery:
    system: str = "earth-moon"
    family: str = "halo"
    libration_point: Optional[int] = 2
    branch: Optional[str] = "S"
    period_min: Optional[float] = None
    period_max: Optional[float] = None
    period_units: str = "d"
    jacobi_min: Optional[float] = None
    jacobi_max: Optional[float] = None
    stability_min: Optional[float] = None
    stability_max: Optional[float] = None

    def params(self) -> dict[str, object]:
        params: dict[str, object] = {
            "sys": self.system,
            "family": self.family,
            "periodunits": self.period_units,
        }
        if self.libration_point is not None:
            params["libr"] = int(self.libration_point)
        if self.branch:
            params["branch"] = self.branch
        if self.period_min is not None:
            params["periodmin"] = float(self.period_min)
        if self.period_max is not None:
            params["periodmax"] = float(self.period_max)
        if self.jacobi_min is not None:
            params["jacobimin"] = float(self.jacobi_min)
        if self.jacobi_max is not None:
            params["jacobimax"] = float(self.jacobi_max)
        if self.stability_min is not None:
            params["stabmin"] = float(self.stability_min)
        if self.stability_max is not None:
            params["stabmax"] = float(self.stability_max)
        return params


def fetch_periodic_orbits(
    query: PeriodicOrbitQuery,
    *,
    timeout_s: float = 30.0,
) -> PeriodicOrbitCatalog:
    url = f"{BASE_URL}?{urlencode(query.params())}"
    with urlopen(url, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return parse_periodic_orbits_payload(payload)


def parse_periodic_orbits_payload(payload: dict) -> PeriodicOrbitCatalog:
    if "warning" in payload:
        raise ValueError(str(payload["warning"]))

    signature = payload.get("signature", {})
    signature_version = str(signature.get("version", ""))
    if signature_version != SUPPORTED_SIGNATURE_VERSION:
        raise ValueError(
            "Unexpected JPL Periodic Orbits API version "
            f"{signature_version!r}; expected {SUPPORTED_SIGNATURE_VERSION!r}."
        )

    system = _parse_system(payload["system"])
    family = str(payload["family"])
    libration_point = _optional_int(payload.get("libration_point"))
    branch = _optional_str(payload.get("branch"))
    fields = [str(f) for f in payload["fields"]]
    records = [
        _parse_record(
            row=row,
            fields=fields,
            system=system,
            family=family,
            libration_point=libration_point,
            branch=branch,
        )
        for row in payload.get("data", [])
    ]

    return PeriodicOrbitCatalog(
        signature_version=signature_version,
        system=system,
        family=family,
        libration_point=libration_point,
        branch=branch,
        records=records,
        filters=dict(payload.get("filter", {})),
        limits=dict(payload.get("limits", {})),
    )


def _parse_system(raw: dict) -> CR3BPSystemUnits:
    libration_points = {
        key: _parse_vector3(raw[key])
        for key in ("L1", "L2", "L3", "L4", "L5")
        if key in raw
    }
    return CR3BPSystemUnits(
        name=str(raw["name"]),
        mass_ratio=float(raw["mass_ratio"]),
        radius_secondary_km=float(raw["radius_secondary"]),
        lunit_km=float(raw["lunit"]),
        tunit_s=float(raw["tunit"]),
        libration_points=libration_points,
    )


def _parse_record(
    *,
    row: list[object],
    fields: list[str],
    system: CR3BPSystemUnits,
    family: str,
    libration_point: Optional[int],
    branch: Optional[str],
) -> PeriodicOrbitRecord:
    values = dict(zip(fields, row))
    state = (
        float(values["x"]),
        float(values["y"]),
        float(values["z"]),
        float(values["vx"]),
        float(values["vy"]),
        float(values["vz"]),
    )
    return PeriodicOrbitRecord(
        system=system,
        family=family,
        libration_point=libration_point,
        branch=branch,
        state_norm=state,
        jacobi=float(values["jacobi"]),
        period=float(values["period"]),
        stability=float(values["stability"]),
    )


def _optional_int(value: object) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _parse_vector3(value: object) -> tuple[float, float, float]:
    seq = list(value)  # type: ignore[arg-type]
    if len(seq) != 3:
        raise ValueError(f"Expected length-3 vector, got {seq!r}")
    return (float(seq[0]), float(seq[1]), float(seq[2]))
