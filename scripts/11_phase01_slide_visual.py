"""Slide visual for phases 00 + 01: Moon-relative cislunar arc with Earth-Moon inset.

Main panel is a 2D Moon-relative XY view of the SPICE-propagated truth trajectory,
with the Moon rendered at true radius and an Earth-direction arrow. A small inset
shows the same arc in an Earth-centered XY frame so viewers see where in the
Earth-Moon system the arc lives.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from _common import ensure_src_on_path, kernel_paths, repo_path

ensure_src_on_path()


MOON_RADIUS_KM = 1737.4
EARTH_RADIUS_KM = 6378.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", action="append", required=True, help="SPICE kernel; repeat for LSK/SPK.")
    parser.add_argument("--epoch", default="2026 APR 10 00:00:00 TDB")
    parser.add_argument(
        "--trajectory-csv",
        type=Path,
        default=Path("results/seeds/spice_nrho_seed.csv"),
        help="Inertial J2000 barycentric truth trajectory from script 01.",
    )
    parser.add_argument("--label", default="L2 S halo truth arc")
    parser.add_argument("--out-plot", type=Path, default=Path("reports/phase01_slide_visual.png"))
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import numpy as np

    from dynamics.spice_ephemeris import SpiceEphemeris
    from visualization.style import (
        AMBER,
        BG,
        BORDER,
        CYAN,
        DIM,
        EARTH,
        GREEN,
        MOON,
        PANEL,
        RED,
        TEXT,
        apply_dark_theme,
        plt,
        style_axis,
    )

    apply_dark_theme()

    traj_path = repo_path(args.trajectory_csv)
    if not traj_path.exists():
        print(f"Trajectory CSV not found: {traj_path}", file=sys.stderr)
        return 1

    t_list: list[float] = []
    x_list: list[list[float]] = []
    with traj_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_list.append(float(row["t_s"]))
            x_list.append([
                float(row["x_km"]), float(row["y_km"]), float(row["z_km"]),
                float(row["vx_km_s"]), float(row["vy_km_s"]), float(row["vz_km_s"]),
            ])
    times = np.asarray(t_list, dtype=float)
    sc_inertial = np.asarray(x_list, dtype=float)

    ephemeris = SpiceEphemeris(kernels=kernel_paths(args.kernel), epoch=args.epoch)
    try:
        moon_rel = np.empty((times.size, 3), dtype=float)
        earth_rel = np.empty((times.size, 3), dtype=float)
        moon_wrt_earth = np.empty((times.size, 3), dtype=float)
        for i, t_s in enumerate(times):
            moon_state = ephemeris.state_km_s("MOON", float(t_s))
            earth_state = ephemeris.state_km_s("EARTH", float(t_s))
            moon_rel[i] = sc_inertial[i, :3] - moon_state[:3]
            earth_rel[i] = sc_inertial[i, :3] - earth_state[:3]
            moon_wrt_earth[i] = moon_state[:3] - earth_state[:3]

        earth_from_moon_t0 = -moon_wrt_earth[0]
    finally:
        ephemeris.close()

    duration_days = float(times[-1] - times[0]) / 86400.0

    # --------------------------------------------------------------
    # Figure: single main panel (Moon-relative XY) + Earth-Moon inset
    # --------------------------------------------------------------
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor(BG)

    ax = fig.add_subplot(111)
    style_axis(
        ax,
        title=None,
        xlabel="x from Moon [km]",
        ylabel="y from Moon [km]",
    )
    ax.set_aspect("equal", adjustable="box")

    # Trajectory curve with a subtle glow.
    ax.plot(moon_rel[:, 0], moon_rel[:, 1], color=CYAN, lw=6.5, alpha=0.18, zorder=3)
    ax.plot(
        moon_rel[:, 0], moon_rel[:, 1],
        color=CYAN, lw=2.4, zorder=4,
        label=f"{args.label} ({duration_days:.2f} d)",
    )

    # Start / end markers.
    ax.scatter(
        moon_rel[0, 0], moon_rel[0, 1],
        s=90, color=GREEN, edgecolors=BG, linewidths=1.0, zorder=6, label="start",
    )
    ax.scatter(
        moon_rel[-1, 0], moon_rel[-1, 1],
        s=90, color=RED, edgecolors=BG, linewidths=1.0, zorder=6, label="end",
    )

    # True-scale Moon circle at origin.
    theta = np.linspace(0.0, 2.0 * np.pi, 256)
    ax.fill(
        MOON_RADIUS_KM * np.cos(theta),
        MOON_RADIUS_KM * np.sin(theta),
        color=MOON, alpha=0.95, zorder=5, linewidth=0,
    )
    ax.plot(
        MOON_RADIUS_KM * np.cos(theta),
        MOON_RADIUS_KM * np.sin(theta),
        color=DIM, lw=0.8, alpha=0.6, zorder=5,
    )
    ax.text(
        0.0, -MOON_RADIUS_KM * 1.6, "Moon (true scale)",
        color=MOON, fontsize=9, ha="center", va="top", zorder=7,
    )

    # Earth-direction arrow anchored near the Moon, pointing toward Earth.
    data_span = max(
        float(np.ptp(moon_rel[:, 0])),
        float(np.ptp(moon_rel[:, 1])),
        1.0,
    )
    earth_hat_xy = earth_from_moon_t0[:2]
    earth_hat_xy = earth_hat_xy / max(float(np.linalg.norm(earth_hat_xy)), 1e-9)
    arrow_len = 0.32 * data_span
    arrow_start = earth_hat_xy * (MOON_RADIUS_KM * 2.2)
    arrow_tip = arrow_start + earth_hat_xy * arrow_len
    ax.annotate(
        "",
        xy=arrow_tip,
        xytext=arrow_start,
        arrowprops=dict(arrowstyle="->", color=EARTH, lw=2.4),
        zorder=6,
    )
    # Offset the label perpendicular to the arrow so it never sits on top of the
    # y-axis label or the Moon marker, regardless of which way Earth happens to point.
    perp = np.array([-earth_hat_xy[1], earth_hat_xy[0]])
    if perp[1] < 0.0:  # always push the label upward
        perp = -perp
    arrow_mid = 0.5 * (arrow_start + arrow_tip)
    label_pos = arrow_mid + perp * (arrow_len * 0.18)
    ax.text(
        float(label_pos[0]), float(label_pos[1]),
        "toward Earth (t=0)",
        color=EARTH, fontsize=10, fontweight="bold",
        ha="center", va="bottom", zorder=7,
    )

    ax.margins(x=0.10, y=0.18)
    leg = ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
    for txt in leg.get_texts():
        txt.set_color(TEXT)

    # --------------------------------------------------------------
    # Inset: Earth-Moon system overview (Earth-centered XY).
    # --------------------------------------------------------------
    inset = fig.add_axes([0.685, 0.545, 0.27, 0.27])
    inset.set_facecolor(PANEL)
    for spine in inset.spines.values():
        spine.set_edgecolor(BORDER)
    inset.tick_params(colors=DIM, labelsize=7)
    inset.grid(True, color=BORDER, linestyle="--", alpha=0.6)
    inset.set_aspect("equal", adjustable="box")

    # Moon path (Earth-relative) over the window.
    inset.plot(
        moon_wrt_earth[:, 0], moon_wrt_earth[:, 1],
        color=MOON, lw=1.0, alpha=0.55, linestyle=":", zorder=2,
    )
    # Spacecraft path in Earth-centered frame.
    inset.plot(
        earth_rel[:, 0], earth_rel[:, 1],
        color=CYAN, lw=1.6, zorder=4,
    )
    # Earth at origin.
    inset.scatter([0.0], [0.0], s=110, color=EARTH, edgecolors=BG, linewidths=0.8, zorder=5)
    inset.annotate(
        "Earth",
        xy=(0.0, 0.0),
        xytext=(8, -4),
        textcoords="offset points",
        color=EARTH, fontsize=9, fontweight="bold",
        ha="left", va="top", zorder=6,
    )
    # Moon at t=0.
    inset.scatter(
        [moon_wrt_earth[0, 0]], [moon_wrt_earth[0, 1]],
        s=65, color=MOON, edgecolors=BG, linewidths=0.8, zorder=5,
    )
    inset.annotate(
        "Moon",
        xy=(float(moon_wrt_earth[0, 0]), float(moon_wrt_earth[0, 1])),
        xytext=(10, -4),
        textcoords="offset points",
        color=MOON, fontsize=9, fontweight="bold",
        ha="left", va="center", zorder=6,
    )
    inset.set_title("Earth-centered overview", color=TEXT, fontsize=10, pad=8)
    inset.set_xlabel("x [km, Earth-rel]", color=DIM, fontsize=8, labelpad=2)
    inset.set_ylabel("y [km, Earth-rel]", color=DIM, fontsize=8, labelpad=2)
    inset.margins(x=0.22, y=0.30)

    # --------------------------------------------------------------
    # Title, subtitle, footer.
    # --------------------------------------------------------------
    fig.suptitle(
        "Cislunar scenario — Moon-relative truth arc",
        color=TEXT, fontsize=15, y=0.965,
    )
    fig.text(
        0.5, 0.925,
        f"SPICE-backed propagation   •   epoch {args.epoch}   •   {duration_days:.2f}-day window",
        color=DIM, fontsize=10, ha="center",
    )
    fig.text(
        0.5, 0.045,
        "axes in km, Moon-relative   •   inset: Earth-centered XY   •   markers: start (green), end (red)",
        color=DIM, fontsize=10, ha="center",
    )

    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.13)

    out_path = repo_path(args.out_plot)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, facecolor=BG)
    plt.close(fig)
    print(f"wrote_plot {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
