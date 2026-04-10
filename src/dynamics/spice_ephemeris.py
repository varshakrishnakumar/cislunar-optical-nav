from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .ephemeris import (
    GM_EARTH_KM3_S2,
    GM_MOON_KM3_S2,
    PointMassBody,
)
from .point_mass import PointMassDynamics


Array = np.ndarray
GM_SUN_KM3_S2 = 132712440041.93938
DEFAULT_GM_KM3_S2 = {
    "SUN": GM_SUN_KM3_S2,
    "EARTH": GM_EARTH_KM3_S2,
    "MOON": GM_MOON_KM3_S2,
}


@dataclass
class SpiceEphemeris:
    """SPICE-backed barycentric body positions for high-fidelity point-mass dynamics."""

    kernels: Sequence[str | Path]
    epoch: str | float
    frame: str = "J2000"
    observer: str = "SOLAR SYSTEM BARYCENTER"
    aberration_correction: str = "NONE"
    load_on_init: bool = True

    def __post_init__(self) -> None:
        self._spice = _import_spiceypy()
        self._loaded = False
        self._epoch_et: float | None = None
        if self.load_on_init:
            self.load()

    @property
    def epoch_et(self) -> float:
        self.load()
        assert self._epoch_et is not None
        return self._epoch_et

    def load(self) -> None:
        if not self._loaded:
            for kernel in self.kernels:
                kernel_path = Path(kernel).expanduser()
                if not kernel_path.exists():
                    msg = f"SPICE kernel does not exist: {kernel_path}"
                    txt_path = Path(str(kernel_path) + ".txt")
                    if txt_path.exists():
                        msg += f" (found {txt_path}; pass that path or rename it to {kernel_path.name})"
                    raise FileNotFoundError(msg)
                self._spice.furnsh(str(kernel_path))
            self._loaded = True

        if self._epoch_et is None:
            if isinstance(self.epoch, (int, float)):
                self._epoch_et = float(self.epoch)
            else:
                self._epoch_et = float(self._spice.str2et(str(self.epoch)))

    def unload(self) -> None:
        if not self._loaded:
            return
        for kernel in reversed(self.kernels):
            self._spice.unload(str(Path(kernel).expanduser()))
        self._loaded = False

    def close(self) -> None:
        self.unload()

    def __enter__(self) -> SpiceEphemeris:
        self.load()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.unload()

    def position_km(self, target: str, t_s: float) -> Array:
        self.load()
        et = self.epoch_et + float(t_s)
        position, _ = self._spice.spkpos(
            target,
            et,
            self.frame,
            self.aberration_correction,
            self.observer,
        )
        return np.asarray(position, dtype=float)

    def state_km_s(self, target: str, t_s: float) -> Array:
        self.load()
        et = self.epoch_et + float(t_s)
        state, _ = self._spice.spkezr(
            target,
            et,
            self.frame,
            self.aberration_correction,
            self.observer,
        )
        return np.asarray(state, dtype=float)

    def point_mass_body(self, target: str, gm_km3_s2: float | None = None) -> PointMassBody:
        target_key = target.upper()
        gm = DEFAULT_GM_KM3_S2[target_key] if gm_km3_s2 is None else float(gm_km3_s2)
        return PointMassBody(
            name=target_key.title(),
            gm_km3_s2=gm,
            position_km=lambda t_s, target_name=target_key: self.position_km(target_name, t_s),
        )

    def point_mass_bodies(
        self,
        targets: Sequence[str] = ("SUN", "EARTH", "MOON"),
        gm_overrides_km3_s2: dict[str, float] | None = None,
    ) -> tuple[PointMassBody, ...]:
        overrides = {k.upper(): v for k, v in (gm_overrides_km3_s2 or {}).items()}
        return tuple(
            self.point_mass_body(target, gm_km3_s2=overrides.get(target.upper()))
            for target in targets
        )


def make_spice_point_mass_dynamics(
    *,
    kernels: Sequence[str | Path],
    epoch: str | float,
    targets: Sequence[str] = ("SUN", "EARTH", "MOON"),
    frame: str = "J2000",
    observer: str = "SOLAR SYSTEM BARYCENTER",
    aberration_correction: str = "NONE",
    gm_overrides_km3_s2: dict[str, float] | None = None,
) -> tuple[SpiceEphemeris, PointMassDynamics]:
    ephemeris = SpiceEphemeris(
        kernels=kernels,
        epoch=epoch,
        frame=frame,
        observer=observer,
        aberration_correction=aberration_correction,
    )
    dynamics = PointMassDynamics(
        ephemeris.point_mass_bodies(
            targets=targets,
            gm_overrides_km3_s2=gm_overrides_km3_s2,
        )
    )
    return ephemeris, dynamics


def _import_spiceypy():
    try:
        import spiceypy
    except ImportError as exc:
        raise ImportError(
            "SPICE ephemeris support requires the optional 'spiceypy' package. "
            "Install it with `python3 -m pip install spiceypy` in your environment."
        ) from exc
    return spiceypy
