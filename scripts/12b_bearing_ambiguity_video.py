"""Animation for Slide 06 — bearing-only ambiguity & motion-induced observability.

Three acts:
  1. A single line-of-sight from the spacecraft to the target body.
  2. Multiple candidate target positions appear along the same ray
     (range ambiguity: û fixes direction, not depth).
  3. The spacecraft moves; new bearings are drawn from each successive
     position; the intersection of rays tightens onto the true target.

Renders an MP4 at results/videos/12_bearing_ambiguity.mp4 by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import ensure_src_on_path

ensure_src_on_path()

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


# ── colour palette (matches bearing_concept_visual + animate_phases_2_3) ───
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
GREEN    = "#10B981"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=Path("results/videos/12_bearing_ambiguity.mp4"))
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seconds-per-act", type=float, default=3.0,
                   help="Approximate duration of each of the three acts.")
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--gif", action="store_true",
                   help="Also write a .gif alongside the .mp4.")
    return p.parse_args()


# ── helpers ─────────────────────────────────────────────────────────────────
def _ease(t: float) -> float:
    """Smoothstep ease-in/out on [0, 1]."""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _draw_starfield(ax, xlim, ylim, n=260, seed=3):
    rng = np.random.default_rng(seed)
    sx = rng.uniform(*xlim, n)
    sy = rng.uniform(*ylim, n)
    ax.scatter(sx, sy, s=rng.uniform(0.2, 2.4, n),
               color="white", alpha=0.18, zorder=0)


def _draw_spacecraft(ax, x, y, *, color=SC_COL, panel=PANEL, scale=1.0,
                     z=5):
    """Tiny stylised spacecraft with body, lens, and solar panels."""
    s = scale
    body = FancyBboxPatch((x - 3.2 * s, y - 2.2 * s), 6.4 * s, 4.4 * s,
                          boxstyle="round,pad=0.15,rounding_size=0.6",
                          facecolor=panel, edgecolor=color, linewidth=1.4,
                          zorder=z)
    ax.add_patch(body)
    lens = Rectangle((x + 3.2 * s, y - 1.1 * s), 2.0 * s, 2.2 * s,
                     facecolor=panel, edgecolor=color, linewidth=1.2,
                     zorder=z)
    ax.add_patch(lens)
    aperture = Circle((x + 5.1 * s, y), 0.75 * s,
                      facecolor=CYAN, edgecolor=color, linewidth=0.7,
                      alpha=0.85, zorder=z + 1)
    ax.add_patch(aperture)
    for side in (+1, -1):
        ax.add_patch(Rectangle((x - 3.2 * s - 4.0 * s,
                                y + side * 0.8 * s - 0.3 * s),
                               3.8 * s, 0.6 * s,
                               facecolor=panel, edgecolor=color,
                               linewidth=0.7, zorder=z - 1))
    ax.plot([x - 3.2 * s, x - 3.2 * s - 4.0 * s], [y, y],
            color=color, linewidth=0.5, zorder=z - 1)
    return aperture  # so caller knows where the LOS originates


def _draw_moon(ax, x, y, r, *, color=MOON_COL, z=5, alpha=0.95):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="#8891A3",
                        linewidth=0.8, zorder=z, alpha=alpha))
    for (cx, cy, cr) in [(-2.0, 1.3, 1.1), (1.6, -1.4, 0.9),
                         (0.4, 2.4, 0.6), (-1.2, -2.6, 0.7)]:
        ax.add_patch(Circle((x + cx * r / 6.8, y + cy * r / 6.8),
                            cr * r / 6.8,
                            facecolor="#B7BCC7", edgecolor="none",
                            alpha=0.55 * alpha, zorder=z + 1))


def _arrow(ax, p0, p1, *, color, lw=2.4, alpha=0.95, glow=True, z=4,
           head_length=12, head_width=7):
    if glow:
        ax.add_patch(FancyArrowPatch(
            p0, p1,
            arrowstyle=f"-|>,head_length={head_length},head_width={head_width}",
            color=color, linewidth=lw * 3.0, alpha=alpha * 0.10,
            zorder=z - 1, mutation_scale=1))
    return ax.add_patch(FancyArrowPatch(
        p0, p1,
        arrowstyle=f"-|>,head_length={head_length},head_width={head_width}",
        color=color, linewidth=lw, alpha=alpha, zorder=z, mutation_scale=1))


# ── main animation ──────────────────────────────────────────────────────────
def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fps = args.fps
    frames_per_act = int(round(args.seconds_per_act * fps))
    n_act1 = frames_per_act          # draw single LOS
    n_act2 = frames_per_act          # show range ambiguity
    n_act3 = int(round(frames_per_act * 1.4))  # motion → triangulation
    n_tail = int(round(0.8 * fps))   # final hold
    N = n_act1 + n_act2 + n_act3 + n_tail

    # ── world layout (scene units; ax is unitless 0..100 × 0..60) ──────────
    XL = (0, 100)
    YL = (0, 60)

    moon_x, moon_y, moon_r = 80.0, 36.0, 5.6
    # Spacecraft trajectory: gentle arc on the left, sweeping rightward & up
    # over the course of Act 3. We parametrise by alpha ∈ [0, 1].
    def sc_position(alpha: float) -> tuple[float, float]:
        a = float(np.clip(alpha, 0.0, 1.0))
        # x: 16 → 32, y: 36 → 22 (curves down so LOS angle changes)
        x = 16.0 + 16.0 * a
        y = 36.0 - 14.0 * (a ** 1.15)
        return x, y

    sc0 = sc_position(0.0)

    # Candidate "ghost" target positions along the û ray (Act 2)
    n_ghosts = 5
    # Distribute fractions along the LOS (excluding endpoints)
    ghost_fracs = np.linspace(0.30, 0.92, n_ghosts)

    # Sampled spacecraft positions along the trajectory used in Act 3
    n_legs = 6
    leg_alphas = np.linspace(0.0, 1.0, n_legs)

    # ── figure ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(*XL)
    ax.set_ylim(*YL)
    ax.set_aspect("equal")
    ax.axis("off")

    _draw_starfield(ax, XL, YL)

    # static title
    ax.text(50, 56.5, "Bearing-Only Geometry  ·  Range Is Hidden Until We Move",
            color=TEXT, fontsize=15, ha="center", va="center",
            fontweight="bold", zorder=10)

    # static target body
    _draw_moon(ax, moon_x, moon_y, moon_r, z=5)
    ax.text(moon_x, moon_y - moon_r - 1.4, "Target body",
            color=MOON_COL, fontsize=10, ha="center", va="top",
            fontweight="bold", zorder=6)

    # static footer (changes per act)
    footer = ax.text(50, 3.0, "",
                     color=DIM, fontsize=10.5, ha="center", va="center",
                     style="italic", zorder=8)

    # act badge in the top-left corner
    badge_box = FancyBboxPatch((2.0, 50.6), 18.0, 3.4,
                               boxstyle="round,pad=0.2,rounding_size=0.6",
                               facecolor=PANEL, edgecolor=BORDER,
                               linewidth=1.2, zorder=8)
    ax.add_patch(badge_box)
    badge = ax.text(11.0, 52.3, "", color=CYAN, fontsize=10.5,
                    ha="center", va="center", fontweight="bold", zorder=9)

    # ── dynamic artists (created once, mutated per frame) ──────────────────
    # Spacecraft: redrawn each frame as a list of patches (cheap to rebuild)
    sc_patches: list = []

    # Trajectory dotted line (revealed in Act 3)
    traj_x = np.array([sc_position(a)[0]
                       for a in np.linspace(0, 1, 200)])
    traj_y = np.array([sc_position(a)[1]
                       for a in np.linspace(0, 1, 200)])
    traj_line, = ax.plot([], [], color=DIM, linestyle=(0, (2, 3)),
                         linewidth=1.0, alpha=0.0, zorder=2)

    # Primary LOS arrow (Act 1 + Act 2)
    primary_arrow = None
    primary_glow = None
    primary_label = ax.text(0, 0, "", color=CYAN, fontsize=11,
                            ha="center", va="bottom", fontweight="bold",
                            alpha=0.0, zorder=7)
    primary_unit = ax.text(0, 0, r"$\hat{u}$", color=CYAN, fontsize=11,
                           ha="center", va="bottom", alpha=0.0, zorder=7)

    # Ghost target candidates (Act 2)
    ghost_circles: list = []
    ghost_labels: list = []
    for f in ghost_fracs:
        c = Circle((0, 0), moon_r * 0.55, facecolor=MOON_COL,
                   edgecolor="#8891A3", linewidth=0.8,
                   alpha=0.0, zorder=4)
        ax.add_patch(c)
        ghost_circles.append(c)
        ghost_labels.append(ax.text(0, 0, "?", color=AMBER, fontsize=12,
                                    ha="center", va="center",
                                    fontweight="bold", alpha=0.0, zorder=6))

    range_bracket, = ax.plot([], [], color=RED, linewidth=1.6,
                              linestyle=(0, (5, 3)), alpha=0.0, zorder=5)
    range_caps_left, = ax.plot([], [], color=RED, linewidth=1.6,
                                alpha=0.0, zorder=5)
    range_caps_right, = ax.plot([], [], color=RED, linewidth=1.6,
                                 alpha=0.0, zorder=5)
    range_label = ax.text(0, 0, r"Range  $\rho$  is unknown",
                          color=RED, fontsize=11.5, ha="center", va="top",
                          fontweight="bold", alpha=0.0, zorder=6)

    # Persistent leg-arrows from previous spacecraft positions (Act 3)
    leg_artists: list = []  # list of (arrow_patch, glow_patch, dot_patch)

    # Convergence highlight (Act 3 final)
    converge_ring = Circle((moon_x, moon_y), moon_r * 1.6,
                           facecolor="none", edgecolor=GREEN,
                           linewidth=2.2, alpha=0.0, zorder=7)
    ax.add_patch(converge_ring)
    converge_label = ax.text(moon_x, moon_y + moon_r + 2.4,
                             "rays intersect → range observable",
                             color=GREEN, fontsize=10.5, ha="center",
                             va="bottom", fontweight="bold",
                             alpha=0.0, zorder=8)

    # ── per-frame helpers ──────────────────────────────────────────────────
    def clear_spacecraft():
        nonlocal sc_patches
        for p in sc_patches:
            try:
                p.remove()
            except Exception:
                pass
        sc_patches = []

    def draw_sc_at(x, y):
        """Draws a spacecraft at (x,y) and records its patches."""
        body = FancyBboxPatch((x - 3.2, y - 2.2), 6.4, 4.4,
                              boxstyle="round,pad=0.15,rounding_size=0.6",
                              facecolor=PANEL, edgecolor=SC_COL, linewidth=1.4,
                              zorder=6)
        ax.add_patch(body)
        lens = Rectangle((x + 3.2, y - 1.1), 2.0, 2.2,
                         facecolor=PANEL, edgecolor=SC_COL, linewidth=1.2,
                         zorder=6)
        ax.add_patch(lens)
        aperture = Circle((x + 5.1, y), 0.75,
                          facecolor=CYAN, edgecolor=SC_COL, linewidth=0.7,
                          alpha=0.9, zorder=7)
        ax.add_patch(aperture)
        sp_top = Rectangle((x - 7.2, y + 0.5), 3.8, 0.6,
                           facecolor=PANEL, edgecolor=SC_COL,
                           linewidth=0.7, zorder=5)
        sp_bot = Rectangle((x - 7.2, y - 1.1), 3.8, 0.6,
                           facecolor=PANEL, edgecolor=SC_COL,
                           linewidth=0.7, zorder=5)
        ax.add_patch(sp_top)
        ax.add_patch(sp_bot)
        line = ax.plot([x - 3.2, x - 7.2], [y, y],
                       color=SC_COL, linewidth=0.5, zorder=5)[0]
        sc_patches.extend([body, lens, aperture, sp_top, sp_bot, line])
        return (x + 5.9, y)  # LOS origin (just past lens)

    def los_endpoint(origin, target, frac=1.0):
        """Endpoint at distance `frac` along the ray from origin → target."""
        dx, dy = target[0] - origin[0], target[1] - origin[1]
        return (origin[0] + frac * dx, origin[1] + frac * dy)

    # We'll draw the primary arrow as actual patches and remove/replace each
    # frame, since FancyArrowPatch isn't easily re-coordinable.
    primary_artists: list = []

    def clear_primary_arrow():
        nonlocal primary_artists
        for p in primary_artists:
            try:
                p.remove()
            except Exception:
                pass
        primary_artists = []

    def set_primary_arrow(origin, end, alpha=1.0, color=CYAN, lw=2.4):
        """Draw glow + main LOS arrow from origin to end."""
        clear_primary_arrow()
        glow = FancyArrowPatch(origin, end,
                               arrowstyle="-|>,head_length=14,head_width=8",
                               color=color, linewidth=lw * 3.5,
                               alpha=alpha * 0.10, zorder=3,
                               mutation_scale=1)
        main = FancyArrowPatch(origin, end,
                               arrowstyle="-|>,head_length=14,head_width=8",
                               color=color, linewidth=lw,
                               alpha=alpha, zorder=4, mutation_scale=1)
        ax.add_patch(glow)
        ax.add_patch(main)
        primary_artists.extend([glow, main])

    def add_leg(origin, target, *, alpha=0.55):
        """Persistently add a faint LOS leg from a past spacecraft pos."""
        glow = FancyArrowPatch(origin, target,
                               arrowstyle="-|>,head_length=10,head_width=6",
                               color=CYAN, linewidth=6.5,
                               alpha=alpha * 0.10, zorder=2,
                               mutation_scale=1)
        main = FancyArrowPatch(origin, target,
                               arrowstyle="-|>,head_length=10,head_width=6",
                               color=CYAN, linewidth=1.6,
                               alpha=alpha, zorder=3, mutation_scale=1)
        dot = Circle(origin, 0.6, facecolor=CYAN, edgecolor="white",
                     linewidth=0.6, alpha=alpha, zorder=4)
        ax.add_patch(glow)
        ax.add_patch(main)
        ax.add_patch(dot)
        leg_artists.append((main, glow, dot))

    def clear_legs():
        for (m, g, d) in leg_artists:
            try:
                m.remove(); g.remove(); d.remove()
            except Exception:
                pass
        leg_artists.clear()

    # ── per-frame update ───────────────────────────────────────────────────
    def update(frame: int):
        # Determine current act
        if frame < n_act1:
            act, t = 1, frame / max(n_act1 - 1, 1)
        elif frame < n_act1 + n_act2:
            act, t = 2, (frame - n_act1) / max(n_act2 - 1, 1)
        elif frame < n_act1 + n_act2 + n_act3:
            act, t = 3, (frame - n_act1 - n_act2) / max(n_act3 - 1, 1)
        else:
            act, t = 4, (frame - n_act1 - n_act2 - n_act3) / max(n_tail - 1, 1)

        # Always start by clearing per-frame transient artists
        clear_spacecraft()

        # ── Act 1: extend a single LOS from the spacecraft ─────────────────
        if act == 1:
            badge.set_text("ACT 1  ·  ONE BEARING")
            badge.set_color(CYAN)
            footer.set_text(
                "A single image gives a unit vector û.  "
                "Direction is fixed — depth along the ray is not.")

            origin = draw_sc_at(*sc0)
            target = (moon_x - moon_r * 0.95, moon_y)
            end = los_endpoint(origin, target, frac=_ease(t))
            set_primary_arrow(origin, end, alpha=0.95)

            mid_x = (origin[0] + target[0]) / 2
            primary_label.set_position((mid_x, moon_y + 2.6))
            primary_label.set_text("Line of sight")
            primary_label.set_alpha(_ease(t))
            primary_unit.set_position((mid_x, moon_y + 1.0))
            primary_unit.set_text(r"$\hat{u}$  (unit vector)")
            primary_unit.set_alpha(_ease(t))

            # ghosts hidden
            for c, lab in zip(ghost_circles, ghost_labels):
                c.set_alpha(0.0); lab.set_alpha(0.0)
            range_bracket.set_alpha(0.0)
            range_caps_left.set_alpha(0.0); range_caps_right.set_alpha(0.0)
            range_label.set_alpha(0.0)
            traj_line.set_alpha(0.0)
            converge_ring.set_alpha(0.0); converge_label.set_alpha(0.0)

        # ── Act 2: range is ambiguous — ghost candidates appear ────────────
        elif act == 2:
            badge.set_text("ACT 2  ·  RANGE IS HIDDEN")
            badge.set_color(AMBER)
            footer.set_text(
                "Many target positions along û produce the same image.  "
                "Range ρ is unobservable from a single bearing.")

            origin = draw_sc_at(*sc0)
            target = (moon_x - moon_r * 0.95, moon_y)
            end = los_endpoint(origin, target, frac=1.0)
            set_primary_arrow(origin, end, alpha=0.95)

            mid_x = (origin[0] + target[0]) / 2
            primary_label.set_position((mid_x, moon_y + 2.6))
            primary_label.set_text("Line of sight")
            primary_label.set_alpha(1.0)
            primary_unit.set_position((mid_x, moon_y + 1.0))
            primary_unit.set_alpha(0.0)

            # Ghost candidates fade in along the ray with a pulsing alpha
            for i, (c, lab, frac) in enumerate(zip(ghost_circles,
                                                    ghost_labels,
                                                    ghost_fracs)):
                gx = origin[0] + frac * (target[0] - origin[0])
                gy = origin[1] + frac * (target[1] - origin[1])
                c.center = (gx, gy)
                # staggered onset + gentle pulse
                onset = i / n_ghosts
                local = np.clip((t - onset * 0.5) / 0.4, 0.0, 1.0)
                pulse = 0.6 + 0.4 * np.sin(2 * np.pi * (t * 1.5 + i * 0.2))
                a = _ease(local) * pulse * 0.85
                c.set_alpha(a)
                lab.set_position((gx, gy))
                lab.set_alpha(a * 0.9)

            # Red bracket showing unknown range, fades in late in Act 2
            br_t = _ease(np.clip((t - 0.35) / 0.5, 0.0, 1.0))
            br_x0, br_x1 = origin[0] + 1.0, target[0] - 1.0
            br_y = origin[1] - 4.0
            range_bracket.set_data([br_x0, br_x1], [br_y, br_y])
            range_bracket.set_alpha(0.95 * br_t)
            range_caps_left.set_data([br_x0, br_x0], [br_y, br_y + 1.2])
            range_caps_right.set_data([br_x1, br_x1], [br_y, br_y + 1.2])
            range_caps_left.set_alpha(0.95 * br_t)
            range_caps_right.set_alpha(0.95 * br_t)
            range_label.set_position(((br_x0 + br_x1) / 2, br_y - 1.4))
            range_label.set_alpha(br_t)

            traj_line.set_alpha(0.0)
            converge_ring.set_alpha(0.0); converge_label.set_alpha(0.0)

        # ── Act 3: motion resolves the ambiguity ───────────────────────────
        elif act == 3:
            badge.set_text("ACT 3  ·  MOTION ADDS DEPTH")
            badge.set_color(GREEN)
            footer.set_text(
                "As the spacecraft moves, the parallax between successive "
                "bearings constrains range.  Observability returns.")

            # Fade out ghosts & range bracket
            fade = _ease(np.clip((t) / 0.25, 0.0, 1.0))
            for c, lab in zip(ghost_circles, ghost_labels):
                cur = c.get_alpha() if c.get_alpha() is not None else 0.0
                c.set_alpha(max(0.0, cur * (1 - fade)))
                lab.set_alpha(max(0.0, lab.get_alpha() * (1 - fade)))
            for art in (range_bracket, range_caps_left, range_caps_right,
                        range_label):
                cur = art.get_alpha() if art.get_alpha() is not None else 0.0
                art.set_alpha(max(0.0, cur * (1 - fade)))

            # Reveal trajectory dashed line
            traj_line.set_alpha(0.55 * _ease(np.clip(t / 0.3, 0, 1)))
            traj_line.set_data(traj_x, traj_y)

            # Spacecraft moves along trajectory
            alpha_traj = _ease(t)
            sc_now = sc_position(alpha_traj)
            origin_now = draw_sc_at(*sc_now)

            # Live LOS from current spacecraft to Moon
            target = (moon_x - moon_r * 0.95, moon_y)
            set_primary_arrow(origin_now, target, alpha=0.95)
            primary_label.set_alpha(0.0); primary_unit.set_alpha(0.0)

            # Drop persistent legs as we cross each leg checkpoint
            already = len(leg_artists)
            triggered = int(np.floor(alpha_traj * (n_legs - 0.001)))
            triggered = min(triggered, n_legs - 1)
            while already <= triggered:
                a_leg = leg_alphas[already]
                if a_leg >= alpha_traj - 1e-3 and already > 0:
                    break
                pos = sc_position(a_leg)
                # leg origin is just past the lens
                add_leg((pos[0] + 5.9, pos[1]), target, alpha=0.65)
                already += 1

            # Convergence ring grows in during second half of Act 3
            cv = _ease(np.clip((t - 0.55) / 0.4, 0.0, 1.0))
            converge_ring.set_alpha(0.85 * cv)
            converge_ring.set_radius(moon_r * (1.7 - 0.4 * cv))
            converge_label.set_alpha(cv)

        # ── Tail: hold final frame ─────────────────────────────────────────
        else:
            badge.set_text("ACT 3  ·  MOTION ADDS DEPTH")
            badge.set_color(GREEN)
            footer.set_text(
                "Six bearings from a moving camera intersect at the target — "
                "range becomes observable.")

            sc_now = sc_position(1.0)
            origin_now = draw_sc_at(*sc_now)
            target = (moon_x - moon_r * 0.95, moon_y)
            set_primary_arrow(origin_now, target, alpha=0.95)
            primary_label.set_alpha(0.0); primary_unit.set_alpha(0.0)
            traj_line.set_alpha(0.55)
            converge_ring.set_alpha(0.85)
            converge_ring.set_radius(moon_r * 1.3)
            converge_label.set_alpha(1.0)

        # Return iterable of artists (blit=False, so this is decorative)
        return []

    # ── render ─────────────────────────────────────────────────────────────
    print(f"Total frames: {N}  ({n_act1}+{n_act2}+{n_act3}+{n_tail})")
    print(f"Output: {args.out}")
    anim = FuncAnimation(fig, update, frames=N, interval=1000 / fps,
                         blit=False)

    writer = FFMpegWriter(fps=fps, bitrate=4500,
                          codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p",
                                      "-preset", "medium"])
    anim.save(args.out, writer=writer, dpi=args.dpi)
    print(f"  saved → {args.out}")

    if args.gif:
        gif_path = args.out.with_suffix(".gif")
        anim.save(gif_path, writer=PillowWriter(fps=min(fps, 20)),
                  dpi=max(args.dpi // 2, 100))
        print(f"  saved → {gif_path}")

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
