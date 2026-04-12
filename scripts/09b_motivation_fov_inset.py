"""Motivation-slide composite: hero crop + simulated onboard-camera view inset.

Loads reports/motivation_visual.png and renders, on a wider canvas, a second
panel that illustrates what the onboard camera sees at the tracking-phase
waypoint: Moon disk at its real angular size for a ~20 deg FOV, a few stars,
Earth as a small off-axis beacon, and a center reticle. Thin connector lines
tie the panel to the frustum apex in the base image.

Intended strictly as an illustrative / motivation figure, not a real render
from the camera pose.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import ensure_src_on_path, repo_path

ensure_src_on_path()


BG = "#080B14"
PANEL_BG = "#02030A"
VIOLET = "#A78BFA"
CYAN = "#33D1FF"
EARTH = "#4B82F8"
TEXT = "#E8EEF9"
DIM = "#A9B4C8"

MOON_RADIUS_KM = 1737.4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hero", type=Path, default=Path("reports/motivation_visual.png"))
    p.add_argument("--moon-texture", type=Path, default=Path("results/moon_texture.jpg"))
    p.add_argument("--out", type=Path, default=Path("reports/motivation_visual_fov.png"))
    p.add_argument("--range-km", type=float, default=15000.0,
                   help="Assumed Moon-relative range at tracking-phase waypoint.")
    p.add_argument("--fov-deg", type=float, default=20.0,
                   help="Assumed camera field of view (full angle) in degrees.")
    p.add_argument("--apex-x", type=int, default=410,
                   help="Pixel x of the frustum apex in the hero crop.")
    p.add_argument("--apex-y", type=int, default=510,
                   help="Pixel y of the frustum apex in the hero crop.")
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def main() -> int:
    import numpy as np
    from PIL import Image
    from matplotlib.patches import Rectangle

    from visualization.style import apply_dark_theme, plt

    apply_dark_theme()

    args = parse_args()

    hero_path = repo_path(args.hero)
    if not hero_path.exists():
        raise SystemExit(f"hero image not found: {hero_path}")

    hero = np.asarray(Image.open(hero_path).convert("RGB"))
    hh, hw = hero.shape[:2]

    # Canvas: hero on left, camera panel on right with a gutter.
    gutter = int(hw * 0.05)
    panel_w = int(hw * 0.58)
    panel_h = int(hh * 0.58)
    panel_x0 = hw + gutter
    panel_y0 = int(hh * 0.18)
    panel_x1 = panel_x0 + panel_w
    panel_y1 = panel_y0 + panel_h
    total_w = panel_x1 + gutter
    total_h = hh

    fig = plt.figure(figsize=(total_w / args.dpi, total_h / args.dpi), dpi=args.dpi)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor(BG)
    ax.set_xlim(0, total_w)
    ax.set_ylim(total_h, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Hero image on the left.
    ax.imshow(hero, extent=(0, hw, hh, 0), zorder=1, interpolation="bilinear")

    # Connector lines from frustum apex out to the camera panel edge.
    apex_x = int(args.apex_x)
    apex_y = int(args.apex_y)
    for corner_y in (panel_y0, panel_y1):
        ax.plot(
            [apex_x, panel_x0], [apex_y, corner_y],
            color=VIOLET, lw=1.1, alpha=0.55, zorder=2,
            solid_capstyle="round",
        )
    # Tiny dot on the apex to anchor the lines visually.
    ax.scatter([apex_x], [apex_y], s=18, color=VIOLET, linewidths=0, zorder=3)

    # Camera panel background.
    ax.add_patch(Rectangle(
        (panel_x0, panel_y0), panel_w, panel_h,
        facecolor=PANEL_BG, edgecolor=VIOLET, linewidth=1.6, zorder=3,
    ))

    # --- Camera-view content, expressed in panel-normalized [-1, 1] x [-1, 1] ---
    def to_panel(xn: float, yn: float) -> tuple[float, float]:
        px = panel_x0 + (xn + 1.0) * 0.5 * panel_w
        # yn=+1 at top of panel, yn=-1 at bottom; image y grows downward.
        py = panel_y0 + (1.0 - (yn + 1.0) * 0.5) * panel_h
        return px, py

    # Stars.
    rng = np.random.default_rng(11)
    n_stars = 110
    sx = rng.uniform(-1.0, 1.0, n_stars)
    sy = rng.uniform(-1.0, 1.0, n_stars)
    star_sizes = rng.uniform(1.5, 9.0, n_stars) * (panel_w / 520.0)
    star_alpha = rng.uniform(0.35, 0.95, n_stars)
    spx, spy = to_panel(sx, sy)
    ax.scatter(spx, spy, s=star_sizes, c="white", alpha=star_alpha, linewidths=0, zorder=4)

    # Moon: angular size from range.
    moon_half_angle_deg = float(np.degrees(np.arctan(MOON_RADIUS_KM / float(args.range_km))))
    moon_radius_norm = (moon_half_angle_deg * 2.0) / float(args.fov_deg)
    moon_radius_norm = float(np.clip(moon_radius_norm, 0.05, 0.95))

    # Slight off-center bias so the disk isn't perfectly centered — feels less canned.
    moon_center_norm = (0.08, -0.04)
    moon_cx, moon_cy = to_panel(*moon_center_norm)
    moon_radius_px = moon_radius_norm * 0.5 * panel_w

    tex_path = repo_path(args.moon_texture)
    if tex_path.exists():
        tex = np.asarray(Image.open(tex_path).convert("L"), dtype=float) / 255.0
        size = 400
        yy, xx = np.mgrid[:size, :size]
        dx = (xx - size / 2.0) / (size / 2.0)
        dy = (yy - size / 2.0) / (size / 2.0)
        r = np.sqrt(dx * dx + dy * dy)
        mask_disk = r <= 1.0
        # Map the texture onto the disk using a simple planar sampling (good enough
        # for an illustration) and add limb darkening so it reads as a sphere.
        u = np.clip(0.5 + 0.5 * dx, 0.0, 1.0)
        v = np.clip(0.5 + 0.5 * dy, 0.0, 1.0)
        ti = (v * (tex.shape[0] - 1)).astype(int)
        tj = (u * (tex.shape[1] - 1)).astype(int)
        shade = tex[ti, tj]
        # Stretch contrast.
        lo, hi = float(np.percentile(shade[mask_disk], 5)), float(np.percentile(shade[mask_disk], 95))
        shade = np.clip((shade - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        limb = np.clip(1.0 - r, 0.0, 1.0) ** 0.35
        # Simple side illumination so it looks 3D (light from upper left).
        light_dir = np.array([-0.55, -0.55])
        light = np.clip(dx * light_dir[0] + dy * light_dir[1] + 0.85, 0.15, 1.2)
        base = shade * limb * light
        base = np.clip(base, 0.0, 1.0)
        rgba = np.zeros(base.shape + (4,), dtype=float)
        rgba[..., 0] = base * 0.88
        rgba[..., 1] = base * 0.92
        rgba[..., 2] = base * 1.00
        rgba[..., 3] = mask_disk.astype(float)
        extent = (
            moon_cx - moon_radius_px, moon_cx + moon_radius_px,
            moon_cy + moon_radius_px, moon_cy - moon_radius_px,
        )
        ax.imshow(rgba, extent=extent, zorder=5, interpolation="bilinear")
    else:
        from matplotlib.patches import Circle
        ax.add_patch(Circle((moon_cx, moon_cy), moon_radius_px,
                            facecolor="#B4B8C4", edgecolor="none", zorder=5))

    # Soft Moon glow.
    from matplotlib.patches import Circle
    for ring_r, ring_a in ((1.06, 0.18), (1.12, 0.10), (1.22, 0.05)):
        ax.add_patch(Circle(
            (moon_cx, moon_cy), moon_radius_px * ring_r,
            facecolor="none", edgecolor="#9FB3D1", lw=1.2, alpha=ring_a, zorder=4,
        ))

    # Reticle at center of frame (boresight).
    cx_r, cy_r = to_panel(0.0, 0.0)
    reticle_outer = 0.07 * panel_w
    reticle_gap = 0.018 * panel_w
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ax.plot(
            [cx_r + dx * reticle_gap, cx_r + dx * reticle_outer],
            [cy_r + dy * reticle_gap, cy_r + dy * reticle_outer],
            color=VIOLET, lw=1.2, alpha=0.85, zorder=7,
        )
    ax.add_patch(Circle((cx_r, cy_r), 0.005 * panel_w, facecolor=VIOLET, edgecolor="none", zorder=7))

    # Earth as a small off-axis beacon.
    ex, ey = to_panel(-0.74, 0.68)
    for halo_s, halo_a in ((220, 0.10), (120, 0.18)):
        ax.scatter([ex], [ey], s=halo_s, color=EARTH, alpha=halo_a, linewidths=0, zorder=6)
    ax.scatter([ex], [ey], s=55, color=EARTH, edgecolors="white", linewidths=0.6, zorder=7)
    ax.text(ex + 0.025 * panel_w, ey - 0.005 * panel_h, "Earth",
            color=EARTH, fontsize=10, ha="left", va="center", zorder=8)

    # Panel header label, above the frame.
    ax.text(
        (panel_x0 + panel_x1) / 2.0, panel_y0 - 0.035 * panel_h,
        f"onboard camera view  ·  ~{int(round(args.fov_deg))}\u00b0 FOV",
        color=VIOLET, fontsize=13, ha="center", va="bottom", zorder=8,
    )
    # Subtext below, inside the panel.
    ax.text(
        panel_x0 + 0.04 * panel_w, panel_y0 + 0.055 * panel_h,
        f"range to Moon \u2248 {int(round(args.range_km / 1000.0))} 000 km",
        color=DIM, fontsize=9, ha="left", va="top", zorder=8,
    )
    ax.text(
        panel_x1 - 0.04 * panel_w, panel_y1 - 0.05 * panel_h,
        "bearings \u2192 EKF update",
        color=CYAN, fontsize=9, ha="right", va="bottom", zorder=8,
        alpha=0.9,
    )

    out_path = repo_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, facecolor=BG, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote_plot {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
