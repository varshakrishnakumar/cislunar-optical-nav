"""Wall-of-futures mosaic for slide 13 (V&V · WHAT BREAKS IT).

A 5 × 3 grid of mini terminal-dispersion plots tiles the (σ_pix, t_c)
robustness sweep that slide 13 shows as two 1-D curves.  Every cell
runs the same Monte-Carlo ensemble in lock-step, scaled by its own
gridpoint:

    columns →  σ_pix ∈ {1, 2, 3, 4, 5} px
    rows    →  t_c   ∈ {3.0, 2.0, 1.0} CR3BP TU   (top = best)

As the comets fly, each cell colors itself by its median terminal
miss: cyan when tight, amber as it degrades, and a red flicker once
the cell crosses the breach threshold.  The full 2-D damage pattern
of slide 07A × slide 07B reads out at a glance.

Output sized for the slide-13 placeholder (6.32" × 4.76").

Render:
    python scripts/13c_wall_of_futures_video.py [--fps 30] [--seconds 12]
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
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Circle
from matplotlib import patheffects

from dynamics.cr3bp import CR3BP
from dynamics.integrators import propagate

# ── palette (matches anim_03 / slide-13 deck) ────────────────────────────────
_BG     = "#050810"
_PANEL  = "#0A0D18"
_BORDER = "#1A2040"
_TEXT   = "#DCE0EC"
_DIM    = "#3A4060"
_CYAN   = "#22D3EE"
_AMBER  = "#F59E0B"
_GREEN  = "#10B981"
_RED    = "#F43F5E"
_VIOLET = "#8B5CF6"
_WHITE  = "#FFFFFF"

# Earth–Moon CR3BP unit conversions
L_KM  = 384_400.0
T_DAY = 4.343

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR   = Path("results/demos")

# Output canvas — exactly the slide-13 placeholder (6.32" × 4.76").
FIG_W     = 6.32
FIG_H     = 4.76

# ── grid layout ──────────────────────────────────────────────────────────────
SIGMA_PIX_GRID = [1.0, 2.0, 3.0, 4.0, 5.0]   # columns, left → right
TC_GRID        = [3.0, 2.0, 1.0]             # rows,    top → bottom

SIGMA_BASE = 1.0
TC_BASE    = 2.0

# Slide 07B's t_c sweep collapses ~6× when t_c steps from 1 → 3, so a
# (TC_BASE / tc)**1.5 factor lines up with that empirical fall-off.
TC_BETA = 1.5

# Median terminal-miss thresholds (km).
TIGHT_MAX_KM   = 80.0
DEGRADE_MAX_KM = 200.0


# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute: nominal arc + base Monte-Carlo dispersion ensemble
# ─────────────────────────────────────────────────────────────────────────────
def _setup_orbit():
    mu = 0.0121505856
    model = CR3BP(mu=mu)
    L = model.lagrange_points()
    x0 = np.array([L["L1"][0] - 1e-3, 0.0, 0.0,
                   0.0, 0.05, 0.0], dtype=float)
    return mu, model, x0


def _propagate(model, x0, t_eval):
    res = propagate(
        model.eom, (float(t_eval[0]), float(t_eval[-1])), x0,
        t_eval=t_eval, rtol=1e-11, atol=1e-13, method="DOP853",
    )
    return np.asarray(res.x)


def _build_ensemble(*, n_seeds: int, n_steps: int, tf_tu: float):
    """Run nominal + Monte-Carlo arcs once. Each cell later re-uses the
    deviation field with its own (σ, t_c) scaling — propagation is the
    expensive step, scaling is a cheap multiply."""
    _, model, x0 = _setup_orbit()
    t = np.linspace(0.0, float(tf_tu), n_steps)
    X_nom = _propagate(model, x0, t)

    rng = np.random.default_rng(11)
    sig_r = np.array([2.5e-6, 2.5e-6, 1.2e-6])     # DU
    sig_v = np.array([2.0e-5, 2.0e-5, 8.0e-6])     # DU/TU

    X_ens = np.zeros((n_seeds, n_steps, 6))
    for s in range(n_seeds):
        dx0 = np.concatenate([rng.normal(0, sig_r), rng.normal(0, sig_v)])
        X_ens[s] = _propagate(model, x0 + dx0, t)

    dX = X_ens - X_nom[None, :, :]                 # (S, N, 6) — DU
    return dict(
        t_days   = t * T_DAY,
        X_nom_km = X_nom[:, :3] * L_KM,
        dX_km    = dX[:, :, :3] * L_KM,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _smoothstep(x: float, a: float = 0.0, b: float = 1.0) -> float:
    t = float(np.clip((x - a) / (b - a), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _tier(miss_km: float) -> str:
    if miss_km <= TIGHT_MAX_KM:
        return "tight"
    if miss_km <= DEGRADE_MAX_KM:
        return "degrade"
    return "breach"


def _tier_color(tier: str) -> str:
    return {"tight": _CYAN, "degrade": _AMBER, "breach": _RED}[tier]


def _tier_glyph(tier: str) -> str:
    return {"tight": "● TIGHT", "degrade": "▲ DEGRADE",
            "breach": "✕ BREACH"}[tier]


def _try_save(ani, path: Path, fps: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        writer = FFMpegWriter(
            fps=fps, bitrate=4_200,
            extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
        )
        ani.save(str(path), writer=writer, dpi=200)
        print(f"  saved → {path}")
    except Exception as exc:
        gif = path.with_suffix(".gif")
        print(f"  ffmpeg failed ({exc}), falling back → {gif}")
        ani.save(str(gif), writer=PillowWriter(fps=fps), dpi=140)
        print(f"  saved → {gif}")


# ─────────────────────────────────────────────────────────────────────────────
# Animation
# ─────────────────────────────────────────────────────────────────────────────
def animate_wall(*, fps: int = 30, seconds: float = 12.0,
                 n_seeds: int = 8) -> None:
    n_frames = int(round(fps * seconds))

    print(f"Pre-computing {n_seeds} Monte-Carlo arcs …")
    data = _build_ensemble(n_seeds=n_seeds, n_steps=200, tf_tu=2.0)

    Xnom    = data["X_nom_km"]      # (N, 3)
    dX_base = data["dX_km"]         # (S, N, 3)
    t_d     = data["t_days"]
    N_t     = Xnom.shape[0]

    nrows, ncols = len(TC_GRID), len(SIGMA_PIX_GRID)

    # Per-row viewport scaling — every cell's worst-case dispersion fills
    # its row's box. Within-row σ_pix comparison stays visible; the
    # absolute size pattern across rows is conveyed by the miss number
    # text and tier color, not by clipping.
    base_term_max = float(np.max(np.linalg.norm(dX_base[:, -1, :2], axis=1)))
    row_half_km = []
    for tc in TC_GRID:
        row_scale_max = (max(SIGMA_PIX_GRID) / SIGMA_BASE) * \
                        ((TC_BASE / tc) ** TC_BETA)
        row_half_km.append(base_term_max * row_scale_max * 1.20)

    # Pre-compute per-cell scale
    cell_specs = []
    for r, tc in enumerate(TC_GRID):
        for c, sp in enumerate(SIGMA_PIX_GRID):
            sig_factor = sp / SIGMA_BASE
            tc_factor  = (TC_BASE / tc) ** TC_BETA
            cell_specs.append(dict(
                row=r, col=c, sigma_pix=sp, tc=tc,
                scale=sig_factor * tc_factor,
                half_km=row_half_km[r],
            ))

    # ── figure setup ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=_BG)

    # Header
    title = fig.text(0.500, 0.952, "WALL  OF  FUTURES",
                     color=_TEXT, fontsize=12.5, fontweight="bold",
                     ha="center", va="center")
    fig.text(0.500, 0.918,
             "5 × 3 robustness mosaic   ·   σ_pix × t_c   ·   shared MC seed",
             color=_DIM, fontsize=6.5, ha="center", va="center")

    # Grid frame — leave room for column header up top, row labels left,
    # and TWO clean rows of footer text below.
    grid_left, grid_right = 0.115, 0.965
    grid_top,  grid_bot   = 0.853, 0.155
    grid_w = grid_right - grid_left
    grid_h = grid_top - grid_bot
    cell_w = grid_w / ncols
    cell_h = grid_h / nrows

    # Column header — σ_pix axis
    for c, sp in enumerate(SIGMA_PIX_GRID):
        x = grid_left + (c + 0.5) * cell_w
        fig.text(x, grid_top + 0.022, f"σ_pix = {sp:.0f} px",
                 color=_TEXT, fontsize=6.4, fontweight="bold",
                 ha="center", va="center")

    # Row header — t_c axis
    for r, tc in enumerate(TC_GRID):
        y = grid_top - (r + 0.5) * cell_h
        fig.text(grid_left - 0.014, y, f"t_c = {tc:.1f}",
                 color=_TEXT, fontsize=6.4, fontweight="bold",
                 ha="right", va="center")

    # Axis-direction hints
    fig.text(grid_right + 0.000, grid_top + 0.052, "(σ_pix grows ►)",
             color=_DIM, fontsize=5.6, ha="right", va="center",
             fontstyle="italic")
    fig.text(grid_left - 0.014, grid_bot - 0.012, "(t_c shrinks ▼)",
             color=_DIM, fontsize=5.6, ha="right", va="center",
             fontstyle="italic")

    # ── per-cell artists ─────────────────────────────────────────────────
    cells = []
    inner_pad_x = cell_w * 0.06
    inner_pad_y = cell_h * 0.10
    for spec in cell_specs:
        r, c = spec["row"], spec["col"]
        ax_left = grid_left + c * cell_w + inner_pad_x
        ax_bot  = grid_top - (r + 1) * cell_h + inner_pad_y * 0.6
        ax_w    = cell_w - 2 * inner_pad_x
        ax_h    = cell_h - 1.6 * inner_pad_y
        ax = fig.add_axes([ax_left, ax_bot, ax_w, ax_h])

        ax.set_facecolor("#080B17")
        for spine in ax.spines.values():
            spine.set_edgecolor(_BORDER)
            spine.set_linewidth(0.7)
        ax.tick_params(colors=_DIM, labelsize=0, length=0)
        ax.set_xticks([]); ax.set_yticks([])
        half = spec["half_km"]
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")

        # Crosshairs
        ax.axhline(0, color=_DIM, lw=0.4, alpha=0.40, zorder=1)
        ax.axvline(0, color=_DIM, lw=0.4, alpha=0.40, zorder=1)

        # Threshold ring at the breach radius (200 km dispersion). On rows
        # where the row's viewport is bigger than the ring, you see the
        # threshold shrinking inside the cell — which is the punchline.
        ring = Circle((0, 0), radius=DEGRADE_MAX_KM, fill=False,
                      edgecolor=_VIOLET, lw=0.7, ls="--", alpha=0.55,
                      zorder=2)
        ax.add_patch(ring)

        # Soft halo behind the target star — stays visible even when
        # the dispersion is tiny (otherwise the cell looks empty).
        target_glow = ax.scatter([0], [0], marker="o",
                                 s=80, color=_AMBER, alpha=0.20,
                                 zorder=6, edgecolors="none")
        ax.scatter([0], [0], marker="*", s=34,
                   color=_AMBER, edgecolors=_WHITE, linewidths=0.40,
                   zorder=8)

        # Triple-layer trails: outer bloom + mid glow + sharp main.
        ens = []
        for _ in range(n_seeds):
            outer, = ax.plot([], [], lw=2.6, alpha=0.07, zorder=3,
                             solid_capstyle="round")
            mid,   = ax.plot([], [], lw=1.2, alpha=0.28, zorder=4,
                             solid_capstyle="round")
            main,  = ax.plot([], [], lw=0.65, alpha=0.95, zorder=5,
                             solid_capstyle="round")
            ens.append((outer, mid, main))

        # Comet heads — outer halo + sharp dot
        head_halo = ax.scatter(np.zeros(n_seeds), np.zeros(n_seeds),
                               s=42.0, color=_CYAN, alpha=0.25, zorder=8,
                               edgecolors="none")
        heads = ax.scatter(np.zeros(n_seeds), np.zeros(n_seeds),
                           s=10.0, color=_CYAN, zorder=9,
                           edgecolors=_WHITE, linewidths=0.30)

        # Tier badge — top-left corner of cell
        tier_badge = ax.text(0.045, 0.945, "", transform=ax.transAxes,
                             color=_DIM, fontsize=4.7, ha="left", va="top",
                             fontweight="bold")

        # Median miss readout — bottom-center
        miss_lbl = ax.text(0.50, 0.045, "", transform=ax.transAxes,
                           color=_DIM, fontsize=5.4, ha="center",
                           va="bottom", fontweight="bold")

        cells.append(dict(
            spec=spec, ax=ax, ens=ens, heads=heads, head_halo=head_halo,
            ring=ring, target_glow=target_glow,
            tier_badge=tier_badge, miss_lbl=miss_lbl,
        ))

    # ── footer overlay — three clean rows, no overlaps ──────────────────
    footer_y_stats   = 0.092       # row 1: live stats
    footer_y_legend  = 0.052       # row 2: tier swatches + thresholds
    footer_y_caption = 0.018       # row 3: caption / breach-ring note

    sim_text = fig.text(0.020, footer_y_stats, "",
                        color=_DIM, fontsize=6.7, fontweight="bold",
                        ha="left", va="center")
    breach_text = fig.text(0.980, footer_y_stats, "",
                           color=_TEXT, fontsize=6.7, fontweight="bold",
                           ha="right", va="center")

    # Tier swatches — single horizontal row centered, generous spacing
    # so labels don't bump.
    legend_items = [
        ("TIGHT     ≤ 80 km",     _CYAN),
        ("DEGRADING ≤ 200 km",    _AMBER),
        ("BREACHED  > 200 km",    _RED),
    ]
    n_items   = len(legend_items)
    legend_w  = 0.86
    item_w    = legend_w / n_items
    legend_x0 = (1.0 - legend_w) / 2
    swatch_w  = 0.012
    swatch_h  = 0.013
    for i, (text, col) in enumerate(legend_items):
        cx = legend_x0 + (i + 0.5) * item_w
        sx = cx - 0.090
        fig.patches.append(plt.Rectangle(
            (sx, footer_y_legend - swatch_h / 2),
            swatch_w, swatch_h,
            transform=fig.transFigure,
            linewidth=0, facecolor=col, alpha=1.00,
        ))
        fig.text(sx + swatch_w + 0.005, footer_y_legend, text,
                 color=col, fontsize=5.9, fontweight="bold",
                 ha="left", va="center")

    fig.text(0.020, footer_y_caption,
             "violet ring = breach radius (200 km)   ·   shared MC seed across cells",
             color=_VIOLET, fontsize=5.6, ha="left", va="center",
             fontstyle="italic", alpha=0.85)
    fig.text(0.980, footer_y_caption,
             "viewport autoscales per row · arcs are dispersion (Xview − Xnom)",
             color=_DIM, fontsize=5.6, ha="right", va="center",
             fontstyle="italic", alpha=0.85)

    title.set_path_effects([
        patheffects.Stroke(linewidth=2.4, foreground=_BG),
        patheffects.Normal(),
    ])

    # ── update ───────────────────────────────────────────────────────────
    def init():
        for cell in cells:
            for outer, mid, main in cell["ens"]:
                outer.set_data([], [])
                mid.set_data([], [])
                main.set_data([], [])
            cell["heads"].set_offsets(np.zeros((n_seeds, 2)))
            cell["head_halo"].set_offsets(np.zeros((n_seeds, 2)))
            cell["miss_lbl"].set_text("")
            cell["tier_badge"].set_text("")
        sim_text.set_text("")
        breach_text.set_text("")
        return ()

    def update(frame):
        # Single eased sweep — comets all launch together, all arrive
        # together, with a quiet plateau at the end so the final
        # dispersion pattern reads.
        global_phase = frame / max(n_frames - 1, 1)
        flight = _smoothstep(global_phase, 0.05, 0.86)
        head_idx = int(round(flight * (N_t - 1)))

        # Smooth pulse used by halos & breach flicker.
        pulse_slow = 0.5 + 0.5 * np.sin(2 * np.pi * frame / max(fps * 1.6, 1.0))
        flick_fast = 0.5 + 0.5 * np.sin(2 * np.pi * frame / max(fps * 0.35, 1.0))

        n_breach  = 0
        n_degrade = 0

        for cell in cells:
            scale = cell["spec"]["scale"]
            disp = scale * dX_base[:, :head_idx + 1, :2]    # (S, k, 2)

            miss_now = np.linalg.norm(disp[:, -1, :], axis=1) \
                       if disp.shape[1] > 0 else np.zeros(n_seeds)
            med_km = float(np.median(miss_now))
            tier = _tier(med_km)
            col  = _tier_color(tier)

            # Trails — three-layer bloom
            for s, (outer, mid, main) in enumerate(cell["ens"]):
                xs = disp[s, :, 0]
                ys = disp[s, :, 1]
                outer.set_data(xs, ys); outer.set_color(col)
                mid.set_data(xs, ys);   mid.set_color(col)
                main.set_data(xs, ys);  main.set_color(col)

            # Heads + halos
            head_xy = disp[:, -1, :] if disp.shape[1] > 0 \
                      else np.zeros((n_seeds, 2))
            cell["heads"].set_offsets(head_xy)
            cell["heads"].set_facecolor(col)
            cell["heads"].set_sizes(np.full(n_seeds, 8.0 + 8.0 * flight))
            cell["head_halo"].set_offsets(head_xy)
            cell["head_halo"].set_facecolor(col)
            cell["head_halo"].set_sizes(np.full(
                n_seeds, 30.0 + 22.0 * flight + 12.0 * pulse_slow))

            # Target star halo — gentle baseline pulse so the cell never
            # looks empty during the launch phase.
            cell["target_glow"].set_sizes(np.array([60.0 + 28.0 * pulse_slow]))

            # Cell tint + spine + threshold ring
            if tier == "breach":
                bg = (0.20 + 0.18 * flick_fast,
                      0.04 + 0.02 * flick_fast,
                      0.07 + 0.02 * flick_fast)
                cell["ax"].set_facecolor(bg)
                lw_spine = 1.8
                cell["ring"].set_edgecolor(_RED)
                cell["ring"].set_alpha(0.55 + 0.30 * flick_fast)
                cell["ring"].set_linewidth(0.9)
                n_breach += 1
            elif tier == "degrade":
                cell["ax"].set_facecolor("#1A1207")
                lw_spine = 1.2
                cell["ring"].set_edgecolor(_VIOLET)
                cell["ring"].set_alpha(0.55)
                cell["ring"].set_linewidth(0.7)
                n_degrade += 1
            else:
                cell["ax"].set_facecolor("#080B17")
                lw_spine = 0.9
                cell["ring"].set_edgecolor(_VIOLET)
                cell["ring"].set_alpha(0.40)
                cell["ring"].set_linewidth(0.7)
            for spine in cell["ax"].spines.values():
                spine.set_edgecolor(col)
                spine.set_linewidth(lw_spine)

            cell["miss_lbl"].set_text(f"{med_km:,.0f} km")
            cell["miss_lbl"].set_color(col)
            cell["tier_badge"].set_text(_tier_glyph(tier))
            cell["tier_badge"].set_color(col)

        sim_text.set_text(
            f"sim t = {t_d[head_idx]:4.1f} d   ·   flight progress "
            f"{100 * flight:4.0f} %"
        )

        if n_breach > 0:
            breach_text.set_text(
                f"{n_breach} cells breached   ·   {n_degrade} degrading"
            )
            breach_text.set_color(_RED)
            title.set_color(_RED if flick_fast > 0.5 else _TEXT)
        elif n_degrade > 0:
            breach_text.set_text(f"{n_degrade} cells degrading   ·   no breaches")
            breach_text.set_color(_AMBER)
            title.set_color(_TEXT)
        else:
            breach_text.set_text("all cells nominal")
            breach_text.set_color(_CYAN)
            title.set_color(_TEXT)

        return ()

    ani = FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        blit=False, interval=1000 // fps,
    )

    out = OUT_DIR / "anim_13c_wall_of_futures.mp4"
    _try_save(ani, out, fps)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fps",     type=int,   default=30)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--seeds",   type=int,   default=8)
    args = p.parse_args()
    animate_wall(fps=args.fps, seconds=args.seconds, n_seeds=args.seeds)
