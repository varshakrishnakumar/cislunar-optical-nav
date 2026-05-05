"""Lunar surface landmark catalog and lat/lon → Moon-fixed conversion.

This module provides Level-2 landmark cases (Section 'Realism and
Sensitivity Extensions' / Synthetic-Landmark Extension in the report):
real crater coordinates from public selenographic catalogs, with the
identity of each feature assumed known.  No image recognition,
catalog-matching, or tracking is included; the bearings are generated
from the catalog positions directly so the experiment isolates
navigation geometry from computer-vision performance.

Coordinate convention
---------------------
Selenographic latitude φ ∈ [−90°, 90°] is positive northward.
Selenographic longitude λ ∈ [−180°, 180°] is positive eastward, with
the prime meridian (λ = 0) being the sub-Earth point of the
tidally-locked Moon. Under that convention the unit Moon-fixed
position of a surface feature is

    r̂_mf = (cos φ cos λ, cos φ sin λ, sin φ)

In the CR3BP rotating Earth–Moon frame the x-axis points from Earth
to Moon, which approximately coincides with the +x of the Moon-fixed
frame (sub-Earth direction). The +y of the rotating frame is the
in-plane prograde direction (≈ +y of Moon-fixed); +z is normal to the
orbit plane (≈ +z Moon-fixed under the tidal-locked, zero-obliquity
assumption used here).

The catalog positions ride with the Moon center; for the SPICE arm a
single body-attitude assumption (J2000 axes ≈ Moon-fixed at epoch) is
applied. Libration is not modeled. Both idealizations are flagged in
the report.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def latlon_to_unit_offset(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Selenographic (lat, lon) → unit vector in Moon-fixed Cartesian.

    Multiply by the lunar radius (km or ND) to get the actual
    surface-position offset relative to the Moon center.
    """
    phi = float(np.deg2rad(lat_deg))
    lam = float(np.deg2rad(lon_deg))
    return np.array(
        [np.cos(phi) * np.cos(lam),
         np.cos(phi) * np.sin(lam),
         np.sin(phi)],
        dtype=np.float64,
    )


# Public catalog of well-known nearside craters. Coordinates are
# rounded selenographic centers from the IAU planetary nomenclature
# database / common references; precision well below the bearing
# uncertainty driven by pixel noise.
LunarCrater = Tuple[str, float, float]   # (name, lat_deg, lon_deg)


CATALOG_CRATERS_6: List[LunarCrater] = [
    ("Tycho",       -43.3,  -11.2),
    ("Copernicus",    9.7,  -20.1),
    ("Aristarchus",  23.7,  -47.4),
    ("Plato",        51.6,   -9.3),
    ("Kepler",        8.1,  -38.0),
    ("Grimaldi",     -5.2,  -68.4),
]


CATALOG_CRATERS_12: List[LunarCrater] = CATALOG_CRATERS_6 + [
    ("Aristoteles",  50.2,   17.4),
    ("Theophilus",  -11.4,   26.4),
    ("Langrenus",    -8.9,   60.9),
    ("Petavius",    -25.3,   60.4),
    ("Endymion",     53.6,   57.0),
    ("Hipparchus",   -5.5,    4.8),
]


_CATALOGS: Dict[str, List[LunarCrater]] = {
    "catalog_craters_6":  CATALOG_CRATERS_6,
    "catalog_craters_12": CATALOG_CRATERS_12,
}


def catalog_unit_offsets(case: str) -> np.ndarray:
    """Return an (N, 3) array of unit Moon-fixed offsets for the named
    catalog case.  Caller multiplies by the lunar radius in km
    (SPICE) or ND (CR3BP) to get the actual surface positions.
    """
    if case not in _CATALOGS:
        raise ValueError(
            f"Unknown landmark catalog case: {case!r} "
            f"(known: {sorted(_CATALOGS)})"
        )
    crater_list = _CATALOGS[case]
    return np.array(
        [latlon_to_unit_offset(lat, lon) for _, lat, lon in crater_list],
        dtype=np.float64,
    )


def catalog_names(case: str) -> List[str]:
    if case not in _CATALOGS:
        raise ValueError(f"Unknown landmark catalog case: {case!r}")
    return [name for name, _, _ in _CATALOGS[case]]


__all__ = [
    "LunarCrater",
    "CATALOG_CRATERS_6",
    "CATALOG_CRATERS_12",
    "latlon_to_unit_offset",
    "catalog_unit_offsets",
    "catalog_names",
]
