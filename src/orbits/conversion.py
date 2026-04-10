from __future__ import annotations

from math import cos, sin

from .types import CR3BPSystemUnits, State6


def normalized_to_dimensional_state(
    state_norm: State6,
    system: CR3BPSystemUnits,
) -> State6:
    """Convert normalized CR3BP synodic state to km and km/s."""
    if len(state_norm) != 6:
        raise ValueError(f"state_norm must have length 6, got {len(state_norm)}")

    velocity_unit_km_s = system.lunit_km / system.tunit_s
    return (
        float(state_norm[0]) * system.lunit_km,
        float(state_norm[1]) * system.lunit_km,
        float(state_norm[2]) * system.lunit_km,
        float(state_norm[3]) * velocity_unit_km_s,
        float(state_norm[4]) * velocity_unit_km_s,
        float(state_norm[5]) * velocity_unit_km_s,
    )


def synodic_to_inertial_state(
    state_synodic: State6,
    system: CR3BPSystemUnits,
    *,
    theta_rad: float = 0.0,
) -> State6:
    """Rotate a dimensional synodic CR3BP state into barycentric inertial axes."""
    if len(state_synodic) != 6:
        raise ValueError(f"state_synodic must have length 6, got {len(state_synodic)}")

    x, y, z, vx, vy, vz = (float(v) for v in state_synodic)
    c = cos(theta_rad)
    s = sin(theta_rad)
    omega = 1.0 / system.tunit_s

    # First convert rotating-frame velocity to inertial velocity in synodic axes.
    vx_i_syn = vx - omega * y
    vy_i_syn = vy + omega * x

    return (
        c * x - s * y,
        s * x + c * y,
        z,
        c * vx_i_syn - s * vy_i_syn,
        s * vx_i_syn + c * vy_i_syn,
        vz,
    )


def normalized_synodic_to_inertial_state(
    state_norm: State6,
    system: CR3BPSystemUnits,
    *,
    theta_rad: float = 0.0,
) -> State6:
    state_dim = normalized_to_dimensional_state(state_norm, system)
    return synodic_to_inertial_state(state_dim, system, theta_rad=theta_rad)
