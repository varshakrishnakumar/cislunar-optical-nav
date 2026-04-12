"""Static software pipeline / state diagram for the refactored cislunar opt-nav stack.

Presentation-oriented: zoned layout, phase pills, hero contributions highlighted,
closed-loop feedback story, takeaway band at the bottom.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import ensure_src_on_path, repo_path

ensure_src_on_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-plot", type=Path, default=Path("reports/pipeline_diagram.png"))
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    from visualization.style import apply_dark_theme, plt

    apply_dark_theme()

    # ------------------------------------------------------------------
    # Standardized palette (per latest spec).
    # ------------------------------------------------------------------
    BG     = "#050B16"
    PANEL  = "#0B1220"
    BORDER = "#1A2744"
    TEXT   = "#EAF1FB"
    DIM    = "#A9B6CA"
    MUTED  = "#70809A"  # file paths, very quiet

    CYAN   = "#33D1FF"  # truth / geometry
    AMBER  = "#F6A91A"  # measurement path
    VIOLET = "#A78BFA"  # filter / estimate
    VIOLET_HERO = "#C4B5FD"  # Phase 7 glow accent
    RED    = "#FF4D6D"  # guidance solve
    GREEN  = "#22C55E"  # correction applied

    TOP_DIM_ALPHA = 0.82  # fade the top pipeline row so the loop story dominates

    fig, ax = plt.subplots(figsize=(19, 12))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.set_aspect("equal")
    ax.axis("off")

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------
    def draw_zone(
        x: float, y: float, w: float, h: float, color: str,
        label: str | None = None, *, fill_alpha: float = 0.05,
    ) -> None:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            linewidth=0.9, edgecolor=color, facecolor=color, alpha=fill_alpha,
            zorder=1,
        ))
        if label:
            # Header sits ABOVE the panel so it never collides with phase pills
            # or block content inside the zone. A solid BG bbox keeps arrows
            # passing through the header area from clipping the text.
            ax.text(
                x + 1.4, y + h + 0.35, label,
                color=color, fontsize=7.6, fontweight="bold",
                ha="left", va="bottom", alpha=0.85, zorder=8,
                bbox=dict(facecolor=BG, edgecolor="none", pad=1.6),
            )

    def draw_phase_pill(cx: float, cy: float, phase_text: str, accent: str) -> None:
        pill_w, pill_h = 4.6, 1.7
        ax.add_patch(FancyBboxPatch(
            (cx - pill_w / 2, cy - pill_h / 2),
            pill_w, pill_h,
            boxstyle="round,pad=0.18,rounding_size=0.8",
            linewidth=0.9, edgecolor=accent, facecolor=PANEL,
            zorder=6,
        ))
        ax.text(
            cx, cy, phase_text,
            color=TEXT, fontsize=7.6, fontweight="bold",
            ha="center", va="center", zorder=7,
        )

    def draw_block(
        x: float, y: float, w: float, h: float,
        title: str, subtitle: str,
        color: str,
        *,
        dim: bool = False,
        hero: bool = False,
        emphasize: bool = False,
    ) -> tuple[float, float, float, float]:
        alpha = TOP_DIM_ALPHA if dim else 1.0

        if hero:
            for pad, ga in ((1.8, 0.06), (1.1, 0.10), (0.5, 0.15)):
                ax.add_patch(FancyBboxPatch(
                    (x - pad, y - pad),
                    w + 2 * pad, h + 2 * pad,
                    boxstyle="round,pad=0.35,rounding_size=1.4",
                    linewidth=0, edgecolor="none",
                    facecolor=VIOLET_HERO, alpha=ga, zorder=2,
                ))
        elif emphasize:
            ax.add_patch(FancyBboxPatch(
                (x - 0.35, y - 0.35),
                w + 0.7, h + 0.7,
                boxstyle="round,pad=0.35,rounding_size=1.2",
                linewidth=0, edgecolor="none",
                facecolor=color, alpha=0.10, zorder=2,
            ))

        lw = 3.0 if hero else (2.4 if emphasize else 1.8)
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.35,rounding_size=1.1",
            linewidth=lw, edgecolor=color, facecolor=PANEL,
            alpha=alpha, zorder=3,
        )
        ax.add_patch(box)

        cx = x + w / 2.0
        title_size = 12.5 if hero else (9.6 if emphasize else 9.2)
        sub_size = 9.2 if hero else 7.6
        multi_line_title = "\n" in title
        title_y = y + h * (0.72 if multi_line_title else 0.66)
        sub_y = y + h * (0.22 if multi_line_title else 0.32)

        ax.text(
            cx, title_y, title,
            color=color, fontsize=title_size, fontweight="bold",
            ha="center", va="center", alpha=alpha, zorder=4,
        )
        ax.text(
            cx, sub_y, subtitle,
            color=TEXT, fontsize=sub_size,
            ha="center", va="center", alpha=alpha, zorder=4,
        )
        return x, y, w, h

    def anchor(b, side: str) -> tuple[float, float]:
        x, y, w, h = b
        if side == "left":   return x, y + h / 2
        if side == "right":  return x + w, y + h / 2
        if side == "top":    return x + w / 2, y + h
        if side == "bottom": return x + w / 2, y
        raise ValueError(side)

    def draw_arrow(
        p0, p1, color=TEXT, dashed=False, curvature=0.0, lw=2.0, alpha=0.95,
    ) -> None:
        ls = (0, (6, 3)) if dashed else "-"
        arrow = FancyArrowPatch(
            p0, p1,
            arrowstyle="-|>,head_length=8,head_width=5",
            linewidth=lw, color=color, alpha=alpha,
            linestyle=ls,
            connectionstyle=f"arc3,rad={curvature}",
            mutation_scale=1.0, zorder=5,
        )
        ax.add_patch(arrow)

    def draw_direction_hint(p0, p1, color, curvature=0.0, alpha=0.95) -> None:
        """Tiny mid-arc arrowhead so curved loops read directionally at a glance.

        Matches matplotlib's arc3 convention: control point is offset from the
        chord midpoint by curvature * (dy, -dx), i.e. the clockwise perpendicular.
        """
        import numpy as np
        a = np.array(p0, dtype=float)
        b = np.array(p1, dtype=float)
        chord = b - a
        dist = float(np.linalg.norm(chord))
        if dist < 1e-6:
            return
        direction = chord / dist
        # Clockwise perpendicular — matches arc3 control-point convention.
        perp = np.array([direction[1], -direction[0]])
        mid = 0.5 * (a + b) + 0.5 * curvature * dist * perp
        size = 1.1
        tail = tuple(mid - direction * size * 0.45)
        head = tuple(mid + direction * size * 0.55)
        hint = FancyArrowPatch(
            tail, head,
            arrowstyle="-|>,head_length=6,head_width=4",
            linewidth=0.6, color=color, alpha=alpha,
            mutation_scale=1.0, zorder=6,
        )
        ax.add_patch(hint)

    # ------------------------------------------------------------------
    # Layout constants.
    # ------------------------------------------------------------------
    block_w, block_h = 14.0, 7.6
    top_y, bot_y = 46.0, 7.0

    cols_top = [3.0, 18.0, 33.0, 48.0, 63.0, 78.0]
    cols_bot = [18.0, 33.0, 48.0, 63.0]

    # ------------------------------------------------------------------
    # 1. Zone backplates (drawn first so everything sits on top).
    # ------------------------------------------------------------------
    # Zone 1 — Dynamics / Truth (top row, SPICE + Dynamics Propagation).
    draw_zone(1.6, 43.6, 30.4, 13.4, CYAN,
              "ZONE 1  ·  Dynamics / Truth", fill_alpha=0.06)
    # Zone 2 — Vision / Measurement generation (top row, cols 2–5).
    # Slightly desaturated/dimmer amber so it groups without dominating.
    draw_zone(31.6, 43.6, 61.8, 13.4, AMBER,
              "ZONE 2  ·  Vision / Measurement generation", fill_alpha=0.035)
    # Zone 3 — Estimation (Phase 7 bridge + IEKF + State Estimate).
    # Two panels so Zone 3 doesn't have to engulf Zone 4 horizontally.
    draw_zone(30.6, 22.5, 38.8, 15.5, VIOLET,
              "ZONE 3  ·  Estimation", fill_alpha=0.055)
    draw_zone(46.4, 4.8, 32.2, 13.4, VIOLET, None, fill_alpha=0.055)
    # Zone 4 — Guidance / Correction (bottom row, Midcourse + Correction).
    draw_zone(16.4, 4.8, 32.2, 13.4, RED,
              "ZONE 4  ·  Guidance / Correction", fill_alpha=0.06)

    # ------------------------------------------------------------------
    # 2. TOP ROW — truth & measurement chain (dimmed; the supporting cast).
    # ------------------------------------------------------------------
    b_spice = draw_block(
        cols_top[0], top_y, block_w, block_h,
        "Ephemeris / Scenario",
        "JPL DE442 kernels",
        CYAN, dim=True,
    )
    b_truth = draw_block(
        cols_top[1], top_y, block_w, block_h,
        "Dynamics Propagation",
        "Integrated state + STM",
        CYAN, dim=True,
    )
    b_cam = draw_block(
        cols_top[2], top_y, block_w, block_h,
        "Camera Model",
        "Pinhole + FOV projection",
        AMBER, dim=True,
    )
    b_blob = draw_block(
        cols_top[3], top_y, block_w, block_h,
        "Target Detection",
        "Centroid in image plane",
        AMBER, dim=True,
    )
    b_p2b = draw_block(
        cols_top[4], top_y, block_w, block_h,
        "Pixel → Bearing",
        "(u, v) → unit LOS",
        AMBER, dim=True,
    )
    b_bear = draw_block(
        cols_top[5], top_y, block_w, block_h,
        "Bearing Measurement",
        "Residual + covariance",
        AMBER, dim=True,
    )

    # ------------------------------------------------------------------
    # 3. BOTTOM ROW — filter + guidance return (the main story, full brightness).
    # ------------------------------------------------------------------
    b_iekf = draw_block(
        cols_bot[3], bot_y, block_w, block_h,
        "IEKF Measurement\nUpdate",
        "Predict · Update (IEKF)",
        VIOLET, emphasize=True,
    )
    b_est = draw_block(
        cols_bot[2], bot_y, block_w, block_h,
        "State Estimate",
        "x̂ , P",
        VIOLET,
    )
    b_guid = draw_block(
        cols_bot[1], bot_y, block_w, block_h,
        "Midcourse Targeting\nSolve",
        "Δv to reduce terminal miss",
        RED, emphasize=True,
    )
    b_corr = draw_block(
        cols_bot[0], bot_y, block_w, block_h,
        "Apply Correction\n+ Repropagate",
        "Δv injected · truth updated",
        GREEN,
    )

    # ------------------------------------------------------------------
    # 4. MIDDLE BRIDGE — Phase 7 hero block.
    # ------------------------------------------------------------------
    # Centered vertically between top-row bottom (top_y) and bottom-row top
    # (bot_y + block_h). Centered horizontally on the pipeline midline.
    bridge_w, bridge_h = 34.0, 9.6
    bridge_x = 50.0 - bridge_w / 2.0
    bridge_y = ((top_y) + (bot_y + block_h)) / 2.0 - bridge_h / 2.0
    b_point = draw_block(
        bridge_x, bridge_y, bridge_w, bridge_h,
        "Phase 7 · Camera Pointing Logic",
        "Repoints camera from filter estimate\nso the target stays inside the FOV",
        VIOLET_HERO, hero=True,
    )

    # ------------------------------------------------------------------
    # 5. Phase pills — one per group, centered above the group.
    # ------------------------------------------------------------------
    def group_center_x(blocks) -> float:
        xs = [b[0] + b[2] / 2.0 for b in blocks]
        return sum(xs) / len(xs)

    pill_y_top = top_y + block_h + 2.7
    pill_y_bot = bot_y + block_h + 2.7
    # 00–01 : Ephemeris + Dynamics Propagation
    draw_phase_pill(group_center_x([b_spice, b_truth]), pill_y_top, "00–01", CYAN)
    # 04–05 : Camera Model + Target Detection + Pixel→Bearing
    draw_phase_pill(group_center_x([b_cam, b_blob, b_p2b]), pill_y_top, "04–05", AMBER)
    # 03 : Bearing Measurement + IEKF Update + State Estimate
    # One pill above the top-row member, one above the bottom-row pair.
    draw_phase_pill(b_bear[0] + b_bear[2] / 2.0, pill_y_top, "03", VIOLET)
    draw_phase_pill(group_center_x([b_est, b_iekf]), pill_y_bot, "03", VIOLET)
    # 02 : Midcourse Targeting + Apply Correction
    draw_phase_pill(group_center_x([b_corr, b_guid]), pill_y_bot, "02", RED)
    # 07 : Phase 7 hero (pill sits directly above the hero block).
    draw_phase_pill(bridge_x + bridge_w / 2.0, bridge_y + bridge_h + 1.7, "07", VIOLET_HERO)

    # ------------------------------------------------------------------
    # 6. Arrows — solid = main flow, dashed = feedback; main loop thicker.
    # ------------------------------------------------------------------
    TOP_LW = 1.6      # dimmed supporting chain
    LOOP_LW = 3.2     # main closed-loop flow
    FEED_LW = 1.8     # dashed feedback / Phase 7 pointing

    # Top chain (dimmed).
    top_chain = [b_spice, b_truth, b_cam, b_blob, b_p2b, b_bear]
    for a, b in zip(top_chain[:-1], top_chain[1:]):
        col = CYAN if a in (b_spice,) else (CYAN if a is b_truth else AMBER)
        draw_arrow(anchor(a, "right"), anchor(b, "left"),
                   color=col, lw=TOP_LW, alpha=0.70)

    # Bearing → IEKF (top-right down to bottom-right).
    bear_to_iekf = (anchor(b_bear, "bottom"), anchor(b_iekf, "top"))
    draw_arrow(*bear_to_iekf, color=VIOLET, curvature=-0.25, lw=LOOP_LW)
    draw_direction_hint(*bear_to_iekf, color=VIOLET, curvature=-0.25)

    # Bottom chain right → left: IEKF → State → Guidance → Correction.
    draw_arrow(anchor(b_iekf, "left"), anchor(b_est, "right"),
               color=VIOLET, lw=LOOP_LW)
    draw_arrow(anchor(b_est, "left"), anchor(b_guid, "right"),
               color=RED, lw=LOOP_LW)
    draw_arrow(anchor(b_guid, "left"), anchor(b_corr, "right"),
               color=GREEN, lw=LOOP_LW)

    # Trajectory correction loops back up to truth propagator (big green arc).
    corr_to_truth = (anchor(b_corr, "top"), anchor(b_truth, "bottom"))
    draw_arrow(*corr_to_truth, color=GREEN, curvature=-0.25, lw=LOOP_LW)
    draw_direction_hint(*corr_to_truth, color=GREEN, curvature=-0.25)

    # Phase 7 dashed feedback: State Estimate → Camera Pointing → Camera Model.
    draw_arrow(anchor(b_est, "top"), anchor(b_point, "bottom"),
               color=VIOLET_HERO, dashed=True, curvature=0.20, lw=FEED_LW)
    draw_arrow(anchor(b_point, "top"), anchor(b_cam, "bottom"),
               color=VIOLET_HERO, dashed=True, curvature=0.20, lw=FEED_LW)

    # "Closed-loop navigation" banner sits in the open band between the
    # Zone 4 header (above the bottom row) and the Zone 3 bridge panel.
    ax.text(
        50.0, 20.8,
        "CLOSED-LOOP  NAVIGATION",
        color=TEXT, fontsize=11.5, fontweight="bold",
        ha="center", va="center", alpha=0.96, zorder=6,
    )

    # ------------------------------------------------------------------
    # 7. Title, subtitle, "You are here" note.
    # ------------------------------------------------------------------
    fig.suptitle(
        "Cislunar Optical Navigation — Software Pipeline",
        color=TEXT, fontsize=18, y=0.975, fontweight="bold",
    )
    fig.text(
        0.5, 0.936,
        "SPICE-backed IEKF with autonomous camera pointing (Phase 7)",
        color=DIM, fontsize=11.5, ha="center",
    )
    fig.text(
        0.5, 0.912,
        "This talk focuses on Phases 02–07 · Phase 08 closes as a future-facing extension",
        color=MUTED, fontsize=9.4, ha="center", style="italic", alpha=0.55,
    )

    # ------------------------------------------------------------------
    # 8. Legend strip (compact, standardized names).
    # ------------------------------------------------------------------
    legend_entries = [
        (CYAN,   "Truth"),
        (AMBER,  "Measurements"),
        (VIOLET, "Estimation"),
        (RED,    "Guidance"),
        (GREEN,  "Control"),
    ]
    lx0 = 4.0
    ly = 60.5
    sw, sh = 1.3, 1.1
    gap = 13.0
    for i, (col, label) in enumerate(legend_entries):
        lx = lx0 + i * gap
        ax.add_patch(FancyBboxPatch(
            (lx, ly - sh / 2), sw, sh,
            boxstyle="round,pad=0.04,rounding_size=0.35",
            linewidth=1.3, edgecolor=col, facecolor=PANEL, zorder=6,
        ))
        ax.text(lx + sw + 0.8, ly, label,
                color=TEXT, fontsize=8.3, va="center", zorder=6)

    # ------------------------------------------------------------------
    # 9. Bottom takeaway band — Without vs With Phase 7.
    # ------------------------------------------------------------------
    band_y = 0.35
    band_h = 3.0
    card_w = 46.0
    left_x, right_x = 2.5, 51.5

    def draw_takeaway(x: float, accent: str, heading: str, body: str) -> None:
        ax.add_patch(FancyBboxPatch(
            (x, band_y), card_w, band_h,
            boxstyle="round,pad=0.22,rounding_size=0.8",
            linewidth=1.6, edgecolor=accent, facecolor=PANEL,
            zorder=6,
        ))
        # Accent bar on the left edge.
        ax.add_patch(FancyBboxPatch(
            (x + 0.3, band_y + 0.3), 0.55, band_h - 0.6,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            linewidth=0, edgecolor="none", facecolor=accent, alpha=0.9,
            zorder=7,
        ))
        ax.text(x + 1.5, band_y + band_h * 0.68, heading,
                color=accent, fontsize=10.6, fontweight="bold",
                ha="left", va="center", zorder=7)
        ax.text(x + 1.5, band_y + band_h * 0.28, body,
                color="#D4DDEF", fontsize=9.1,
                ha="left", va="center", zorder=7)

    draw_takeaway(
        left_x, RED,
        "Without Phase 7",
        "Fixed camera → target leaves FOV → measurements drop → estimator degrades",
    )
    draw_takeaway(
        right_x, GREEN,
        "With Phase 7",
        "Estimated LOS repoints camera → target stays in FOV → updates persist → stable filter",
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.895, bottom=0.025)

    out_path = repo_path(args.out_plot)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote_plot {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
