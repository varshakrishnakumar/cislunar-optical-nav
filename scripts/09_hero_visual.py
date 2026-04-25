"""Hero visual: SPICE NRHO + onboard optical navigation beacons.

Renders a single dark 3D scene in the Moon-centered frame showing the
propagated NRHO trajectory, a textured Moon, two labeled maneuvers, and a
camera waypoint with line-of-sight rays to Moon / Earth / Sun. Intended as a
one-shot "what is this repo?" figure.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from _common import (
    ensure_src_on_path,
    kernel_paths,
    repo_path,
)

ensure_src_on_path()


MOON_RADIUS_KM = 1737.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", action="append", required=True, help="SPICE kernel path; repeat for LSK/SPK.")
    parser.add_argument("--epoch", default="2026 APR 10 00:00:00 TDB")
    parser.add_argument(
        "--trajectory-csv",
        type=Path,
        default=Path("results/seeds/spice_nrho_seed.csv"),
        help="Inertial J2000 barycentric truth trajectory (columns: t_s,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s).",
    )
    parser.add_argument(
        "--estimate-csv",
        type=Path,
        default=None,
        help="Optional IEKF estimate trajectory in the same CSV format. If omitted, a synthetic "
             "converging estimate is generated for illustration.",
    )
    parser.add_argument(
        "--estimate-initial-err-km",
        type=float,
        default=2600.0,
        help="Initial error amplitude for the synthetic IEKF estimate.",
    )
    parser.add_argument("--label", default="L2 S halo NRHO (truth)", help="Truth trajectory label for the legend.")
    parser.add_argument("--burn1-frac", type=float, default=0.02, help="Fractional time along trajectory for burn 1.")
    parser.add_argument("--burn2-frac", type=float, default=0.62, help="Fractional time along trajectory for burn 2.")
    parser.add_argument(
        "--camera-frac",
        type=float,
        default=None,
        help="Fractional time for camera waypoint. Default picks a point near --camera-range-km.",
    )
    parser.add_argument(
        "--camera-range-km",
        type=float,
        default=15000.0,
        help="Target Moon-relative range for the default camera waypoint selection.",
    )
    parser.add_argument("--moon-display-scale", type=float, default=4.0, help="Artistic Moon display radius multiplier.")
    parser.add_argument("--los-len-km", type=float, default=7000.0, help="Display length for the Moon LOS arrow.")
    parser.add_argument("--moon-texture", type=Path, default=Path("results/seeds/moon_texture.jpg"))
    parser.add_argument("--out-plot", type=Path, default=Path("reports/hero_visual.png"))
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=-62.0)
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import numpy as np
    from PIL import Image

    from dynamics.spice_ephemeris import SpiceEphemeris
    from visualization.style import BG, PANEL, apply_dark_theme, plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401  (import side effects)

    apply_dark_theme()

    # Local semantic palette (overrides the base style colors for this figure).
    CYAN = "#33D1FF"    # truth / geometry
    VIOLET = "#A78BFA"  # estimate / IEKF
    AMBER = "#F6A91A"   # measurement / highlight (also insertion burn)
    ORANGE = "#F6A91A"  # reuse amber for second burn tint
    GREEN = "#22C55E"   # accepted / success
    RED = "#FF4D6D"     # warning / correction marker
    MOON = "#D5D9E3"
    EARTH = "#4B82F8"
    BORDER = "#1A2744"
    TEXT = "#E8EEF9"
    DIM = "#A9B4C8"

    # ------------------------------------------------------------------
    # 1. Load pre-propagated NRHO trajectory (J2000 barycentric).
    # ------------------------------------------------------------------
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

        # Moon-relative trajectory + velocities (for ΔV arrow orientations).
        moon_rel = np.empty((times.size, 3), dtype=float)
        sc_vel_moon_rel = np.empty((times.size, 3), dtype=float)
        for i, t_s in enumerate(times):
            moon_state = ephemeris.state_km_s("MOON", float(t_s))
            moon_rel[i] = sc_inertial[i, :3] - moon_state[:3]
            sc_vel_moon_rel[i] = sc_inertial[i, 3:6] - moon_state[3:6]

        # IEKF estimate (Moon-relative). Either loaded from CSV or synthesized as
        # a smooth decaying-error perturbation of truth to illustrate filter convergence.
        if args.estimate_csv is not None:
            est_path = repo_path(args.estimate_csv)
            if not est_path.exists():
                print(f"Estimate CSV not found: {est_path}", file=sys.stderr)
                return 1
            est_inertial_list: list[list[float]] = []
            with est_path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    est_inertial_list.append([
                        float(row["x_km"]), float(row["y_km"]), float(row["z_km"]),
                    ])
            est_inertial = np.asarray(est_inertial_list, dtype=float)
            if est_inertial.shape[0] != times.size:
                print(
                    f"Estimate CSV row count ({est_inertial.shape[0]}) does not match "
                    f"truth ({times.size}); skipping estimate.",
                    file=sys.stderr,
                )
                estimate_moon_rel = None
            else:
                estimate_moon_rel = np.empty((times.size, 3), dtype=float)
                for i, t_s in enumerate(times):
                    moon_pos = ephemeris.state_km_s("MOON", float(t_s))[:3]
                    estimate_moon_rel[i] = est_inertial[i] - moon_pos
        else:
            t_rel = times - times[0]
            span = float(t_rel[-1]) if t_rel[-1] > 0 else 1.0
            # Exponentially decaying amplitude: starts at initial_err, converges toward ~3% of it.
            amp = float(args.estimate_initial_err_km) * (0.03 + 0.97 * np.exp(-3.5 * t_rel / span))
            phase_x = 2.0 * np.pi * t_rel / (span * 0.33)
            phase_y = 2.0 * np.pi * t_rel / (span * 0.47) + 1.2
            phase_z = 2.0 * np.pi * t_rel / (span * 0.61) + 2.4
            estimate_moon_rel = moon_rel.copy()
            estimate_moon_rel[:, 0] += amp * np.sin(phase_x)
            estimate_moon_rel[:, 1] += amp * np.cos(phase_y)
            estimate_moon_rel[:, 2] += 0.55 * amp * np.sin(phase_z)

        # ------------------------------------------------------------------
        # 2. Pick waypoints: burns and camera.
        # ------------------------------------------------------------------
        def frac_index(frac: float) -> int:
            return int(np.clip(round(frac * (times.size - 1)), 0, times.size - 1))

        i_burn1 = frac_index(args.burn1_frac)
        i_burn2 = frac_index(args.burn2_frac)
        if args.camera_frac is None:
            # Pick the point along the trajectory with Moon-relative range
            # closest to args.camera_range_km — gives a visible camera + LOS geometry.
            ranges = np.linalg.norm(moon_rel, axis=1)
            i_cam = int(np.argmin(np.abs(ranges - float(args.camera_range_km))))
        else:
            i_cam = frac_index(args.camera_frac)

        # Line-of-sight vectors at the camera waypoint (from spacecraft inertial pos).
        t_cam = float(times[i_cam])
        r_sc_cam = sc_inertial[i_cam, :3]
        moon_pos_cam = ephemeris.state_km_s("MOON", t_cam)[:3]
        earth_pos_cam = ephemeris.state_km_s("EARTH", t_cam)[:3]
        sun_pos_cam = ephemeris.state_km_s("SUN", t_cam)[:3]

        def unit(v: np.ndarray) -> np.ndarray:
            n = float(np.linalg.norm(v))
            return v / n if n > 0.0 else v

        los_moon = unit(moon_pos_cam - r_sc_cam)
        los_earth = unit(earth_pos_cam - r_sc_cam)
        los_sun = unit(sun_pos_cam - r_sc_cam)

        earth_range_km = float(np.linalg.norm(earth_pos_cam - r_sc_cam))
    finally:
        ephemeris.close()

    # Camera position in Moon-relative frame (matches the 3D axis coordinates).
    cam_rel = moon_rel[i_cam]

    # ------------------------------------------------------------------
    # 3. Build figure.
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass

    # Strip default matplotlib 3D panels / grid lines — dimmed so the scene pops.
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(BG)
        pane.set_edgecolor(BORDER)
        pane.set_alpha(0.12)
    ax.grid(True, color=BORDER, linestyle=":", alpha=0.18)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(TEXT)
        axis.set_tick_params(colors=DIM, labelsize=7)
        for line in axis.get_gridlines():
            line.set_alpha(0.18)
    # Fade the axis spine lines themselves.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color(BORDER)
        axis.line.set_alpha(0.5)

    # ------------------------------------------------------------------
    # 4. Textured Moon sphere at origin.
    # ------------------------------------------------------------------
    tex_path = repo_path(args.moon_texture)
    if tex_path.exists():
        img = np.asarray(Image.open(tex_path).convert("L"), dtype=float) / 255.0
    else:
        img = None

    moon_display_radius = MOON_RADIUS_KM * float(args.moon_display_scale)
    n_lon, n_lat = 200, 100
    lon = np.linspace(0.0, 2.0 * np.pi, n_lon)
    lat = np.linspace(0.0, np.pi, n_lat)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    mx = moon_display_radius * np.cos(lon_grid) * np.sin(lat_grid)
    my = moon_display_radius * np.sin(lon_grid) * np.sin(lat_grid)
    mz = moon_display_radius * np.cos(lat_grid)

    if img is not None:
        ix = (lon_grid / (2.0 * np.pi) * (img.shape[1] - 1)).astype(int)
        iy = (lat_grid / np.pi * (img.shape[0] - 1)).astype(int)
        shade = img[iy, ix]
        # Stretch contrast and bias toward cool tone for dark background.
        lo, hi = float(np.percentile(shade, 5)), float(np.percentile(shade, 95))
        shade = np.clip((shade - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        rgba = np.zeros(shade.shape + (4,), dtype=float)
        base = 0.12 + 0.88 * shade
        rgba[..., 0] = base * 0.86
        rgba[..., 1] = base * 0.90
        rgba[..., 2] = base * 1.00
        rgba[..., 3] = 1.0
        ax.plot_surface(
            mx, my, mz,
            facecolors=rgba[:-1, :-1],
            rstride=1, cstride=1,
            linewidth=0, antialiased=False, shade=False,
            zorder=1,
        )
    else:
        ax.plot_surface(
            mx, my, mz,
            color=MOON, alpha=0.95,
            rstride=2, cstride=2,
            linewidth=0, antialiased=True, shade=True,
            zorder=1,
        )

    # Soft glow ring around the Moon terminator.
    ring_theta = np.linspace(0.0, 2.0 * np.pi, 200)
    ring_r = moon_display_radius * 1.03
    ax.plot(
        ring_r * np.cos(ring_theta),
        ring_r * np.sin(ring_theta),
        np.zeros_like(ring_theta),
        color=DIM, lw=0.8, alpha=0.5, zorder=2,
    )

    # ------------------------------------------------------------------
    # 5. NRHO trajectory.
    # ------------------------------------------------------------------
    duration_days = float(times[-1] - times[0]) / 86400.0
    ax.plot(
        moon_rel[:, 0], moon_rel[:, 1], moon_rel[:, 2],
        color=CYAN, lw=3.2, alpha=1.0, zorder=5,
        solid_capstyle="round",
        label=f"{args.label} ({duration_days:.2f} d)",
    )
    if estimate_moon_rel is not None:
        ax.plot(
            estimate_moon_rel[:, 0], estimate_moon_rel[:, 1], estimate_moon_rel[:, 2],
            color=VIOLET, lw=2.7, alpha=1.0, zorder=4,
            linestyle=(0, (7, 3)),
            dash_capstyle="round",
            label="IEKF estimate",
        )
    # Start / end markers.
    ax.scatter(*moon_rel[0], color=GREEN, s=55, edgecolors=BG, linewidths=0.6, zorder=6, label="start")
    ax.scatter(*moon_rel[-1], color=RED, s=55, edgecolors=BG, linewidths=0.6, zorder=6, label="end")

    # ------------------------------------------------------------------
    # 6. Burn waypoints with ΔV arrows.
    # ------------------------------------------------------------------
    def draw_burn(
        idx: int,
        label: str,
        color: str,
        label_offset: np.ndarray,
        dv_scale_km: float = 6500.0,
        bold: bool = False,
    ) -> None:
        r_b = moon_rel[idx]
        v_b = sc_vel_moon_rel[idx]
        v_hat = unit(v_b)
        # Synthetic illustrative ΔV direction: orthogonal to v within orbit plane.
        up = np.array([0.0, 0.0, 1.0])
        side = unit(np.cross(v_hat, up))
        dv_dir = unit(0.6 * v_hat + 0.8 * side)
        ax.scatter(*r_b, color=color, s=90, marker="o", edgecolors=BG, linewidths=0.8, zorder=7)
        ax.quiver(
            r_b[0], r_b[1], r_b[2],
            dv_dir[0] * dv_scale_km, dv_dir[1] * dv_scale_km, dv_dir[2] * dv_scale_km,
            color=color, lw=2.4, arrow_length_ratio=0.28, zorder=7,
        )
        # Short connector line from marker to the label anchor so it stays legible.
        anchor = r_b + label_offset
        ax.plot(
            [r_b[0], anchor[0]], [r_b[1], anchor[1]], [r_b[2], anchor[2]],
            color=color, lw=0.8, alpha=0.55, zorder=7,
        )
        ax.text(
            anchor[0], anchor[1], anchor[2], label,
            color=color, fontsize=10 if bold else 9,
            fontweight="bold" if bold else "normal",
            zorder=8, ha="left", va="center",
        )

    # Push burn labels out away from the orbit curve in explicit directions.
    draw_burn(i_burn1, "(1) insertion burn", AMBER, label_offset=np.array([7500.0, -2500.0, -4500.0]))
    midcourse_offset = np.array([8500.0, 8000.0, 8000.0])
    draw_burn(
        i_burn2, "(3) midcourse correction", RED,
        label_offset=midcourse_offset,
        bold=False,
    )
    midcourse_label_pos = moon_rel[i_burn2] + midcourse_offset

    # ------------------------------------------------------------------
    # 7. Camera waypoint + frustum + LOS rays.
    # ------------------------------------------------------------------
    # Soft stacked halo at the "current" spacecraft position.
    for halo_s, halo_a in ((520.0, 0.08), (320.0, 0.14), (180.0, 0.25)):
        ax.scatter(*cam_rel, color=VIOLET, s=halo_s, alpha=halo_a, linewidths=0, zorder=7)
    ax.scatter(*cam_rel, color=VIOLET, s=95, marker="o", edgecolors=BG, linewidths=0.8, zorder=8)

    # Camera basis: boresight along LOS-to-Moon from the waypoint, up = inertial Z.
    boresight = unit(-cam_rel)  # from spacecraft toward Moon center
    world_up = np.array([0.0, 0.0, 1.0])
    right = unit(np.cross(boresight, world_up))
    if not np.isfinite(right).all() or np.linalg.norm(right) < 1e-9:
        right = unit(np.cross(boresight, np.array([0.0, 1.0, 0.0])))
    up_cam = unit(np.cross(right, boresight))

    frustum_depth = 3200.0
    frustum_half_w = 1300.0
    frustum_half_h = 950.0
    apex = cam_rel
    center = apex + boresight * frustum_depth
    corners = [
        center + right * frustum_half_w + up_cam * frustum_half_h,
        center + right * frustum_half_w - up_cam * frustum_half_h,
        center - right * frustum_half_w - up_cam * frustum_half_h,
        center - right * frustum_half_w + up_cam * frustum_half_h,
    ]
    # Edges apex -> corners.
    for c in corners:
        ax.plot(
            [apex[0], c[0]], [apex[1], c[1]], [apex[2], c[2]],
            color=VIOLET, lw=1.2, alpha=0.85, zorder=8,
        )
    # Image-plane rectangle.
    rect = corners + [corners[0]]
    ax.plot(
        [p[0] for p in rect], [p[1] for p in rect], [p[2] for p in rect],
        color=VIOLET, lw=1.2, alpha=0.9, zorder=8,
    )

    # Camera label: offset well below the apex so it never sits on top of an arrow.
    cam_label_anchor = apex + np.array([-2500.0, 0.0, -5500.0])
    ax.plot(
        [apex[0], cam_label_anchor[0]], [apex[1], cam_label_anchor[1]], [apex[2], cam_label_anchor[2]],
        color=VIOLET, lw=0.8, alpha=0.55, zorder=8,
    )
    ax.text(
        cam_label_anchor[0], cam_label_anchor[1], cam_label_anchor[2],
        "(2) tracking phase",
        color=VIOLET, fontsize=9, ha="right", va="top", zorder=9,
    )

    # Moon LOS ray (the only LOS drawn as a line) — clipped at the Moon's display surface.
    sc_to_moon_km = float(np.linalg.norm(cam_rel))
    moon_los_len = min(
        max(sc_to_moon_km - moon_display_radius * 1.05, 1000.0),
        float(args.los_len_km),
    )
    ax.plot(
        [apex[0], apex[0] + los_moon[0] * moon_los_len],
        [apex[1], apex[1] + los_moon[1] * moon_los_len],
        [apex[2], apex[2] + los_moon[2] * moon_los_len],
        color=CYAN, lw=2.0, alpha=0.95, zorder=9,
    )

    # Earth / Sun: no arrow lines — just point-mass markers along each LOS direction.
    beacon_len = float(args.los_len_km) * 1.05

    def draw_beacon_offset(
        direction: np.ndarray,
        color: str,
        label: str,
        length_km: float,
        text_offset: np.ndarray,
        ha: str,
    ) -> None:
        pos = apex + direction * length_km
        ax.scatter(*pos, color=color, s=120, edgecolors=BG, linewidths=0.8, zorder=10)
        tp = pos + text_offset
        ax.text(tp[0], tp[1], tp[2], label, color=color, fontsize=9, ha=ha, va="center", zorder=11)

    draw_beacon_offset(
        los_earth, EARTH, f"Earth ({earth_range_km/1000:.0f}e3 km)",
        length_km=beacon_len,
        text_offset=np.array([-1800.0, 0.0, 800.0]),
        ha="right",
    )
    draw_beacon_offset(
        los_sun, AMBER, "Sun",
        length_km=beacon_len * 1.15,
        text_offset=np.array([1800.0, 0.0, -1200.0]),
        ha="left",
    )

    # Label the Moon itself, offset above the sphere.
    ax.text(
        0.0, 0.0, moon_display_radius * 1.35,
        "Moon",
        color=MOON, fontsize=10, ha="center", zorder=9,
    )

    # ------------------------------------------------------------------
    # 8. Axis bounds, labels, title, legend.
    # ------------------------------------------------------------------
    los_len_km = float(args.los_len_km)
    all_pts = np.vstack([
        moon_rel,
        np.array([cam_rel + los_moon * moon_los_len,
                  cam_rel + los_earth * los_len_km * 1.2,
                  cam_rel + los_sun * los_len_km * 1.2]),
        # Include Moon sphere extent so it never gets clipped.
        np.array([[moon_display_radius, 0, 0],
                  [-moon_display_radius, 0, 0],
                  [0, moon_display_radius, 0],
                  [0, -moon_display_radius, 0],
                  [0, 0, moon_display_radius],
                  [0, 0, -moon_display_radius]]),
        # Include midcourse label anchor so the "(3)" text never gets clipped.
        midcourse_label_pos.reshape(1, 3) + np.array([[8000.0, 0.0, 0.0]]),
    ])
    pad = 1200.0
    xmin, xmax = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
    ymin, ymax = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad
    zmin, zmax = all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad
    span_x = xmax - xmin
    span_y = ymax - ymin
    span_z = zmax - zmin
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    # Proportional box so matplotlib doesn't pad the figure with empty cube space.
    try:
        ax.set_box_aspect((span_x, span_y, span_z))
    except Exception:
        pass

    ax.set_xlabel("x [km, Moon-rel]", color=TEXT, labelpad=6)
    ax.set_ylabel("y [km, Moon-rel]", color=TEXT, labelpad=6)
    ax.set_zlabel("z [km, Moon-rel]", color=TEXT, labelpad=22)
    ax.view_init(elev=args.elev, azim=args.azim)

    fig.suptitle(
        "Cislunar Optical Navigation — SPICE NRHO with onboard beacon LOS",
        color=TEXT, fontsize=15, y=0.965,
    )
    fig.text(
        0.5, 0.925,
        f"epoch {args.epoch}   •   {duration_days:.2f}-day arc   •   Moon shown at {args.moon_display_scale:g}× radius for clarity",
        color=DIM, fontsize=10, ha="center",
    )

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        fontsize=9,
        framealpha=0.92,
        facecolor=PANEL,
        edgecolor=BORDER,
    )
    for txt in legend.get_texts():
        txt.set_color(TEXT)

    out_path = repo_path(args.out_plot)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote_plot {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
