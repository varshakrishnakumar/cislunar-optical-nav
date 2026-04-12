from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dynamics.spice_ephemeris import SpiceEphemeris
from orbits.spice_bridge import earth_moon_synodic_frame_from_spice
from .style import (
    AMBER,
    BG,
    BORDER,
    CYAN,
    EARTH,
    GREEN,
    MOON,
    RED,
    TEXT,
    VIOLET,
    apply_dark_theme,
    plt,
    style_axis,
)


Array = np.ndarray


@dataclass(frozen=True)
class SpiceRelativeTrajectory:
    times_s: Array
    states_j2000_km_km_s: Array
    earth_rel_km: Array
    moon_rel_km: Array
    synodic_rel_km: Array
    earth_moon_distance_km: Array


def spice_relative_trajectory(
    times_s: Array,
    states_j2000_km_km_s: Array,
    ephemeris: SpiceEphemeris,
    *,
    mass_ratio: float,
) -> SpiceRelativeTrajectory:
    times = np.asarray(times_s, dtype=float).reshape(-1)
    states = np.asarray(states_j2000_km_km_s, dtype=float)
    if states.shape != (times.size, 6):
        raise ValueError(f"states must have shape ({times.size}, 6), got {states.shape}")

    earth_rel = np.empty((times.size, 3), dtype=float)
    moon_rel = np.empty((times.size, 3), dtype=float)
    synodic_rel = np.empty((times.size, 3), dtype=float)
    earth_moon_distance = np.empty(times.size, dtype=float)

    for idx, t_s in enumerate(times):
        earth_state = ephemeris.state_km_s("EARTH", float(t_s))
        moon_state = ephemeris.state_km_s("MOON", float(t_s))
        frame = earth_moon_synodic_frame_from_spice(
            ephemeris,
            t_s=float(t_s),
            mass_ratio=mass_ratio,
        )

        r_sc = states[idx, :3]
        earth_rel[idx] = r_sc - earth_state[:3]
        moon_rel[idx] = r_sc - moon_state[:3]
        synodic_rel[idx] = frame.rotation_synodic_to_inertial.T @ (r_sc - frame.origin_position_km)
        earth_moon_distance[idx] = float(np.linalg.norm(moon_state[:3] - earth_state[:3]))

    return SpiceRelativeTrajectory(
        times_s=times,
        states_j2000_km_km_s=states,
        earth_rel_km=earth_rel,
        moon_rel_km=moon_rel,
        synodic_rel_km=synodic_rel,
        earth_moon_distance_km=earth_moon_distance,
    )


def save_spice_trajectory_report(
    times_s: Array,
    states_j2000_km_km_s: Array,
    ephemeris: SpiceEphemeris,
    outpath: str | Path,
    *,
    mass_ratio: float,
    title: str = "SPICE-Backed Cislunar Trajectory",
) -> Path:
    apply_dark_theme()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    traj = spice_relative_trajectory(
        times_s,
        states_j2000_km_km_s,
        ephemeris,
        mass_ratio=mass_ratio,
    )
    days = traj.times_s / 86400.0
    moon_range = np.linalg.norm(traj.moon_rel_km, axis=1)
    earth_range = np.linalg.norm(traj.earth_rel_km, axis=1)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 11),
        gridspec_kw={"height_ratios": [1.15, 1.0], "wspace": 0.28, "hspace": 0.36},
    )
    fig.patch.set_facecolor(BG)

    ax = axes[0, 0]
    style_axis(
        ax,
        title="Earth-Moon Synodic XY",
        xlabel="x from barycenter [km]",
        ylabel="y [km]",
    )
    ax.plot(traj.synodic_rel_km[:, 0], traj.synodic_rel_km[:, 1], color=CYAN, lw=2.0, label="spacecraft")
    ax.scatter(
        [traj.synodic_rel_km[0, 0]],
        [traj.synodic_rel_km[0, 1]],
        s=55,
        color=GREEN,
        edgecolors=BG,
        label="start",
        zorder=5,
    )
    ax.scatter(
        [traj.synodic_rel_km[-1, 0]],
        [traj.synodic_rel_km[-1, 1]],
        s=55,
        color=RED,
        edgecolors=BG,
        label="end",
        zorder=5,
    )
    mean_em = float(np.mean(traj.earth_moon_distance_km))
    ax.scatter([-mass_ratio * mean_em], [0.0], s=75, color=EARTH, edgecolors=BG, label="Earth", zorder=6)
    ax.scatter([(1.0 - mass_ratio) * mean_em], [0.0], s=55, color=MOON, edgecolors=BG, label="Moon", zorder=6)
    ax.set_title("Earth-Moon Synodic XY Overview", color=TEXT, pad=14)
    ax.set_aspect("auto")
    ax.margins(x=0.05, y=0.22)
    ax.legend(
        fontsize=8,
        loc="lower center",
        ncol=5,
        framealpha=0.92,
    )

    ax = axes[0, 1]
    style_axis(ax, title="Moon-Relative XY", xlabel="x from Moon [km]", ylabel="y from Moon [km]")
    ax.plot(traj.moon_rel_km[:, 0], traj.moon_rel_km[:, 1], color=VIOLET, lw=2.0, label="spacecraft")
    ax.scatter([0.0], [0.0], s=70, color=MOON, edgecolors=BG, label="Moon", zorder=6)
    ax.scatter(
        [traj.moon_rel_km[0, 0]],
        [traj.moon_rel_km[0, 1]],
        s=50,
        color=GREEN,
        edgecolors=BG,
        label="start",
        zorder=5,
    )
    ax.scatter(
        [traj.moon_rel_km[-1, 0]],
        [traj.moon_rel_km[-1, 1]],
        s=50,
        color=RED,
        edgecolors=BG,
        label="end",
        zorder=5,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.92)

    ax = axes[1, 0]
    style_axis(ax, title="Body-Relative Range", xlabel="time [days]", ylabel="range [km]")
    ax.plot(days, moon_range, color=VIOLET, lw=2.0, label="Moon-relative")
    ax.plot(days, earth_range, color=EARTH, lw=2.0, label="Earth-relative")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    style_axis(ax, title="Synodic Z Excursion", xlabel="time [days]", ylabel="z [km]")
    ax.plot(days, traj.synodic_rel_km[:, 2], color=AMBER, lw=2.0, label="z from Earth-Moon plane")
    ax.axhline(0.0, color=BORDER, lw=1.0)
    ax.legend(fontsize=9)

    fig.suptitle(title, color=TEXT, fontsize=15, y=0.985)
    fig.subplots_adjust(top=0.90, bottom=0.09)
    fig.savefig(outpath, dpi=220, facecolor=BG)
    plt.close(fig)
    return outpath


def save_spice_relative_trajectory_csv(
    times_s: Array,
    states_j2000_km_km_s: Array,
    ephemeris: SpiceEphemeris,
    outpath: str | Path,
    *,
    mass_ratio: float,
) -> Path:
    traj = spice_relative_trajectory(
        times_s,
        states_j2000_km_km_s,
        ephemeris,
        mass_ratio=mass_ratio,
    )
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "t_s",
        "t_days",
        "earth_rel_x_km",
        "earth_rel_y_km",
        "earth_rel_z_km",
        "earth_range_km",
        "moon_rel_x_km",
        "moon_rel_y_km",
        "moon_rel_z_km",
        "moon_range_km",
        "synodic_x_km",
        "synodic_y_km",
        "synodic_z_km",
        "earth_moon_distance_km",
    ]
    earth_range = np.linalg.norm(traj.earth_rel_km, axis=1)
    moon_range = np.linalg.norm(traj.moon_rel_km, axis=1)

    with outpath.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, t_s in enumerate(traj.times_s):
            writer.writerow(
                {
                    "t_s": float(t_s),
                    "t_days": float(t_s / 86400.0),
                    "earth_rel_x_km": float(traj.earth_rel_km[idx, 0]),
                    "earth_rel_y_km": float(traj.earth_rel_km[idx, 1]),
                    "earth_rel_z_km": float(traj.earth_rel_km[idx, 2]),
                    "earth_range_km": float(earth_range[idx]),
                    "moon_rel_x_km": float(traj.moon_rel_km[idx, 0]),
                    "moon_rel_y_km": float(traj.moon_rel_km[idx, 1]),
                    "moon_rel_z_km": float(traj.moon_rel_km[idx, 2]),
                    "moon_range_km": float(moon_range[idx]),
                    "synodic_x_km": float(traj.synodic_rel_km[idx, 0]),
                    "synodic_y_km": float(traj.synodic_rel_km[idx, 1]),
                    "synodic_z_km": float(traj.synodic_rel_km[idx, 2]),
                    "earth_moon_distance_km": float(traj.earth_moon_distance_km[idx]),
                }
            )

    return outpath
