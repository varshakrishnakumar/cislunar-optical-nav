"""Static concept diagram explaining a bearing-only measurement.

Shows the spacecraft camera, the target body, the line-of-sight arrow, a small
image-plane inset with a pixel centroid, the pixel→bearing conversion, and the
unobservable range annotation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import ensure_src_on_path

ensure_src_on_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-plot", type=Path,
                        default=Path("reports/bearing_concept_visual.png"))
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                    Rectangle)

    from visualization.style import apply_dark_theme, plt

    apply_dark_theme()

    BG       = "#050B16"
    PANEL    = "#0B1220"
    BORDER   = "#1A2744"
    TEXT     = "#EAF1FB"
    DIM      = "#A9B6CA"
    SC_COL   = "#C8D3E6"
    MOON_COL = "#D5D9E3"
    CYAN     = "#33D1FF"
    AMBER    = "#F6A91A"
    RED      = "#FF4D6D"

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── starfield ─────────────────────────────────────────────────────────
    import numpy as np
    rng = np.random.default_rng(3)
    n_stars = 220
    sx = rng.uniform(0, 100, n_stars)
    sy = rng.uniform(0, 60, n_stars)
    ax.scatter(sx, sy, s=rng.uniform(0.2, 2.4, n_stars),
               color="white", alpha=0.18, zorder=0)

    # ── title ─────────────────────────────────────────────────────────────
    ax.text(50, 56.5, "Bearing-Only Measurement Geometry",
            color=TEXT, fontsize=15, ha="center", va="center",
            fontweight="bold", zorder=10)

    # ======================================================================
    # Element 1 — spacecraft camera
    # ======================================================================
    sc_x, sc_y = 16.0, 36.0

    # camera body (rounded box)
    body = FancyBboxPatch((sc_x - 3.2, sc_y - 2.2), 6.4, 4.4,
                          boxstyle="round,pad=0.15,rounding_size=0.6",
                          facecolor=PANEL, edgecolor=SC_COL, linewidth=1.6,
                          zorder=5)
    ax.add_patch(body)
    # lens barrel pointing right (toward target)
    lens = Rectangle((sc_x + 3.2, sc_y - 1.1), 2.0, 2.2,
                     facecolor=PANEL, edgecolor=SC_COL, linewidth=1.4,
                     zorder=5)
    ax.add_patch(lens)
    # lens aperture
    ax.add_patch(Circle((sc_x + 5.1, sc_y), 0.75,
                        facecolor=CYAN, edgecolor=SC_COL, linewidth=0.8,
                        alpha=0.75, zorder=6))
    # solar panels
    for side in (+1, -1):
        ax.add_patch(Rectangle((sc_x - 3.2 - 4.0, sc_y + side * 0.8 - 0.3),
                               3.8, 0.6, facecolor=PANEL, edgecolor=SC_COL,
                               linewidth=0.8, zorder=4))
    ax.plot([sc_x - 3.2, sc_x - 3.2 - 4.0], [sc_y, sc_y],
            color=SC_COL, linewidth=0.6, zorder=4)

    ax.text(sc_x, sc_y + 4.6, "Spacecraft camera",
            color=SC_COL, fontsize=10.5, ha="center", va="bottom",
            fontweight="bold", zorder=6)

    # ======================================================================
    # Element 2 — target body (Moon)
    # ======================================================================
    moon_x, moon_y, moon_r = 84.0, 36.0, 6.8
    ax.add_patch(Circle((moon_x, moon_y), moon_r,
                        facecolor=MOON_COL, edgecolor="#8891A3",
                        linewidth=1.0, zorder=5, alpha=0.95))
    # subtle craters
    for (cx, cy, cr) in [(-2.0, 1.3, 1.1), (1.6, -1.4, 0.9),
                         (0.4, 2.4, 0.6), (-1.2, -2.6, 0.7)]:
        ax.add_patch(Circle((moon_x + cx, moon_y + cy), cr,
                            facecolor="#B7BCC7", edgecolor="none",
                            alpha=0.55, zorder=6))
    ax.text(moon_x, moon_y - moon_r - 1.6, "Target body",
            color=MOON_COL, fontsize=10.5, ha="center", va="top",
            fontweight="bold", zorder=6)

    # ======================================================================
    # Element 3 — line-of-sight arrow
    # ======================================================================
    los_start = (sc_x + 5.9, sc_y)
    los_end   = (moon_x - moon_r - 0.6, moon_y)

    # soft glow
    ax.add_patch(FancyArrowPatch(los_start, los_end,
                                  arrowstyle="-|>,head_length=14,head_width=8",
                                  color=CYAN, linewidth=8, alpha=0.10,
                                  zorder=3, mutation_scale=1))
    ax.add_patch(FancyArrowPatch(los_start, los_end,
                                  arrowstyle="-|>,head_length=14,head_width=8",
                                  color=CYAN, linewidth=2.4, alpha=0.95,
                                  zorder=4, mutation_scale=1))

    los_mid_x = (los_start[0] + los_end[0]) / 2
    ax.text(los_mid_x, los_start[1] + 2.4, "Line of sight",
            color=CYAN, fontsize=11, ha="center", va="bottom",
            fontweight="bold", zorder=6)
    # unit vector label
    ax.text(los_mid_x, los_start[1] + 0.8,
            r"$\hat{u}$  (unit vector)",
            color=CYAN, fontsize=9, ha="center", va="bottom",
            alpha=0.85, zorder=6)

    # ======================================================================
    # Element 4 — image-plane inset
    # ======================================================================
    ip_cx, ip_cy = 15.0, 16.0
    ip_w, ip_h = 13.0, 9.0
    ip_x0 = ip_cx - ip_w / 2
    ip_y0 = ip_cy - ip_h / 2

    ax.add_patch(FancyBboxPatch((ip_x0, ip_y0), ip_w, ip_h,
                                boxstyle="round,pad=0.1,rounding_size=0.4",
                                facecolor="#0B1220", edgecolor="#1A2744",
                                linewidth=1.4, zorder=5))
    # faint grid inside the frame
    for gx in np.linspace(ip_x0, ip_x0 + ip_w, 5)[1:-1]:
        ax.plot([gx, gx], [ip_y0 + 0.4, ip_y0 + ip_h - 0.4],
                color=BORDER, linewidth=0.5, alpha=0.9, zorder=5)
    for gy in np.linspace(ip_y0, ip_y0 + ip_h, 4)[1:-1]:
        ax.plot([ip_x0 + 0.4, ip_x0 + ip_w - 0.4], [gy, gy],
                color=BORDER, linewidth=0.5, alpha=0.9, zorder=5)

    # orange pixel dot
    dot_x, dot_y = ip_cx - 1.2, ip_cy + 0.6
    ax.add_patch(Circle((dot_x, dot_y), 0.75,
                        facecolor=AMBER, edgecolor="white",
                        linewidth=0.7, zorder=7))
    # glow
    ax.add_patch(Circle((dot_x, dot_y), 1.6,
                        facecolor=AMBER, edgecolor="none",
                        alpha=0.18, zorder=6))
    ax.text(dot_x, dot_y - 1.9, r"$(u, v)$",
            color=AMBER, fontsize=10, ha="center", va="top",
            fontweight="bold", zorder=8)

    ax.text(ip_cx, ip_y0 - 1.2, "Image-plane measurement",
            color=DIM, fontsize=9.5, ha="center", va="top", zorder=6)

    # subtle tether from spacecraft to image-plane inset
    ax.plot([sc_x - 1.0, ip_cx + 2.0],
            [sc_y - 2.4, ip_y0 + ip_h],
            color=SC_COL, linewidth=0.6, linestyle=":",
            alpha=0.55, zorder=3)

    # ======================================================================
    # Element 5 — conversion arrow (pixel centroid → bearing)
    # ======================================================================
    conv_start = (ip_x0 + ip_w + 2.5, ip_cy + 0.2)
    conv_end   = (conv_start[0] + 10.5, ip_cy + 0.2)
    ax.add_patch(FancyArrowPatch(conv_start, conv_end,
                                  arrowstyle="-|>,head_length=10,head_width=6",
                                  color=AMBER, linewidth=1.8, alpha=0.95,
                                  zorder=6, mutation_scale=1))
    ax.text((conv_start[0] + conv_end[0]) / 2, conv_start[1] + 1.6,
            "pixel centroid → bearing",
            color=AMBER, fontsize=9.5, ha="center", va="bottom",
            fontweight="bold", zorder=7)
    ax.text((conv_start[0] + conv_end[0]) / 2, conv_start[1] - 1.4,
            r"$(u,v)\;\to\;\hat{u}$",
            color=AMBER, fontsize=9, ha="center", va="top",
            alpha=0.85, zorder=7)

    # ======================================================================
    # Element 6 — missing-range annotation
    # ======================================================================
    br_x0, br_x1 = los_start[0] + 1.0, los_end[0] - 1.0
    br_y = los_start[1] - 4.2
    # dashed bracket
    ax.plot([br_x0, br_x1], [br_y, br_y],
            color=RED, linewidth=1.6, linestyle=(0, (5, 3)), zorder=5)
    # bracket caps
    for bx in (br_x0, br_x1):
        ax.plot([bx, bx], [br_y, br_y + 1.2],
                color=RED, linewidth=1.6, linestyle="-", zorder=5)

    ax.text((br_x0 + br_x1) / 2, br_y - 1.6,
            r"Range  $\rho$  is unknown",
            color=RED, fontsize=11, ha="center", va="top",
            fontweight="bold", zorder=6)

    # ── footer note ───────────────────────────────────────────────────────
    ax.text(50, 3.0,
            "A single bearing pins the direction to the target but leaves "
            "depth along the line of sight unobserved.",
            color=DIM, fontsize=9, ha="center", va="center",
            style="italic", zorder=6)

    # ── save ──────────────────────────────────────────────────────────────
    args.out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_plot, dpi=args.dpi,
                facecolor=fig.get_facecolor(), bbox_inches="tight",
                pad_inches=0.25)
    print(f"  saved → {args.out_plot}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
