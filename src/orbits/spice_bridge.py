from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics.spice_ephemeris import SpiceEphemeris

from .conversion import normalized_to_dimensional_state
from .types import CR3BPSystemUnits, PeriodicOrbitRecord, State6


Array = np.ndarray


@dataclass(frozen=True)
class SynodicFrameState:
    origin_position_km: Array
    origin_velocity_km_s: Array
    rotation_synodic_to_inertial: Array
    angular_velocity_rad_s: Array


def earth_moon_synodic_frame_from_spice(
    ephemeris: SpiceEphemeris,
    *,
    t_s: float = 0.0,
    mass_ratio: float,
) -> SynodicFrameState:
    earth_state = ephemeris.state_km_s("EARTH", t_s)
    moon_state = ephemeris.state_km_s("MOON", t_s)

    r_earth = earth_state[:3]
    v_earth = earth_state[3:]
    r_moon = moon_state[:3]
    v_moon = moon_state[3:]

    r_rel = r_moon - r_earth
    v_rel = v_moon - v_earth
    r_rel_norm = float(np.linalg.norm(r_rel))
    if r_rel_norm <= 0.0:
        raise RuntimeError("Earth-Moon relative position is zero; cannot build synodic frame.")

    h = np.cross(r_rel, v_rel)
    h_norm = float(np.linalg.norm(h))
    if h_norm <= 0.0:
        raise RuntimeError("Earth-Moon angular momentum is zero; cannot build synodic frame.")

    x_hat = r_rel / r_rel_norm
    z_hat = h / h_norm
    y_hat = np.cross(z_hat, x_hat)
    y_hat = y_hat / np.linalg.norm(y_hat)

    rotation = np.column_stack([x_hat, y_hat, z_hat])
    angular_velocity = h / r_rel_norm**2

    origin_position = (1.0 - mass_ratio) * r_earth + mass_ratio * r_moon
    origin_velocity = (1.0 - mass_ratio) * v_earth + mass_ratio * v_moon

    return SynodicFrameState(
        origin_position_km=origin_position,
        origin_velocity_km_s=origin_velocity,
        rotation_synodic_to_inertial=rotation,
        angular_velocity_rad_s=angular_velocity,
    )


def dimensional_synodic_to_spice_inertial_state(
    state_synodic: State6,
    frame: SynodicFrameState,
) -> Array:
    state = np.asarray(state_synodic, dtype=float).reshape(6)
    r_syn = state[:3]
    v_syn = state[3:]

    r_rel_inertial = frame.rotation_synodic_to_inertial @ r_syn
    v_rel_rotating = frame.rotation_synodic_to_inertial @ v_syn
    v_rel_inertial = v_rel_rotating + np.cross(frame.angular_velocity_rad_s, r_rel_inertial)

    return np.concatenate(
        [
            frame.origin_position_km + r_rel_inertial,
            frame.origin_velocity_km_s + v_rel_inertial,
        ]
    )


def normalized_synodic_to_spice_inertial_state(
    state_norm: State6,
    system: CR3BPSystemUnits,
    ephemeris: SpiceEphemeris,
    *,
    t_s: float = 0.0,
) -> Array:
    frame = earth_moon_synodic_frame_from_spice(
        ephemeris,
        t_s=t_s,
        mass_ratio=system.mass_ratio,
    )
    state_synodic = normalized_to_dimensional_state(state_norm, system)
    return dimensional_synodic_to_spice_inertial_state(state_synodic, frame)


def periodic_orbit_record_to_spice_inertial_state(
    record: PeriodicOrbitRecord,
    ephemeris: SpiceEphemeris,
    *,
    t_s: float = 0.0,
) -> Array:
    return normalized_synodic_to_spice_inertial_state(
        record.state_norm,
        record.system,
        ephemeris,
        t_s=t_s,
    )
