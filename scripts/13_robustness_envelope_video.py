"""Cinematic robustness-envelope demo for slide 13.

A 3-D Earth–Moon CR3BP scene with a textured Moon and a Monte-Carlo
ensemble of bearing-only spacecraft "comets" flying their arcs from
near L1 toward the nominal endpoint. The scene plays five back-to-back
"flights", each frozen at a distinct σ_pix regime — calibrated baseline,
degrading envelope, high-noise stress, clean optics, graceful recovery —
so the audience sees the dispersion *re-shape* between flights, not
just spread on a static line.

Output frame is sized for the slide-13 placeholder (6.32" × 4.76").

Render:
    python scripts/13_robustness_envelope_video.py [--fps 30]
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
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib import patheffects
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

from dynamics.cr3bp import CR3BP
from dynamics.integrators import propagate

# Reuse the already-tuned textured-Moon helper from the main animation.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from animate_phases_2_3 import _draw_textured_sphere, _draw_sphere  # noqa: E402

# ── palette (matches anim_03 / slide-13 deck) ─────────────────────────────────
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
_MOON_C = "#9CA3AF"

# Earth–Moon CR3BP unit conversions
L_KM  = 384_400.0
T_DAY = 4.343

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR   = Path("results/demos")
FIG_W     = 6.32
FIG_H     = 4.76

# Each "flight" the comets take is one of these σ regimes. The labels
# match the slide-13 caption tone — calm, degrading, stressed, clean,
# recovered — so the demo doubles as a narrated walk-through.
CYCLES = [
    ("CALIBRATED BASELINE", 1.0,  _CYAN),
    ("DEGRADING ENVELOPE",  2.5,  _AMBER),
    ("STRESS · HIGH NOISE", 5.0,  _RED),
    ("CLEAN OPTICS",        0.5,  _GREEN),
    ("GRACEFUL RECOVERY",   1.0,  _CYAN),
]


# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute: nominal arc + a Monte-Carlo ensemble of perturbed 3-D arcs
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
    """Run nominal + Monte-Carlo arcs once, store full 3-D state.

    Each seed's deviation `dX[s, t, :3]` is what the animation later
    scales by σ_factor every frame, so propagation runs once and the
    σ-stress sweep becomes a multiply.
    """
    mu, model, x0 = _setup_orbit()
    t = np.linspace(0.0, float(tf_tu), n_steps)

    X_nom = _propagate(model, x0, t)               # (N, 6)

    rng = np.random.default_rng(7)
    # Calibrated against the σ_pix=1 baseline: terminal cloud is a few
    # tens of km, σ=5 stress widens it to a few hundred km. Z-axis
    # spread is intentionally smaller (real arcs are near-planar) but
    # non-zero so the 3-D view has visible depth structure.
    sig_r = np.array([2.5e-6, 2.5e-6, 1.2e-6])     # DU
    sig_v = np.array([2.0e-5, 2.0e-5, 8.0e-6])     # DU/TU

    X_ens = np.zeros((n_seeds, n_steps, 6))
    for s in range(n_seeds):
        dx0 = np.concatenate([rng.normal(0, sig_r), rng.normal(0, sig_v)])
        X_ens[s] = _propagate(model, x0 + dx0, t)

    dX = X_ens - X_nom[None, :, :]                 # (S, N, 6) — DU / DU·TU

    L_pts = model.lagrange_points()
    geom = dict(
        moon_km=np.asarray(model.primary2, dtype=float)[:3] * L_KM
                if np.asarray(model.primary2).size >= 3
                else np.array([float(model.primary2[0]),
                               float(model.primary2[1]), 0.0]) * L_KM,
        L1_km=float(L_pts["L1"][0]) * L_KM,
    )
    return dict(
        t_days=t * T_DAY,
        X_nom_km=X_nom[:, :3] * L_KM,
        dX_km=dX[:, :, :3] * L_KM,
        geom=geom,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _smoothstep(x: float, a: float = 0.0, b: float = 1.0) -> float:
    t = float(np.clip((x - a) / (b - a), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _stars_3d(ax, n=240, seed=11, alpha=0.18):
    rng = np.random.default_rng(seed)
    xl, yl, zl = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    xs = rng.uniform(*xl, n)
    ys = rng.uniform(*yl, n)
    zs = rng.uniform(*zl, n)
    ax.scatter(xs, ys, zs, s=rng.uniform(0.10, 1.4, n),
               color="white", alpha=alpha, zorder=0,
               depthshade=False, rasterized=True)


def _miss_color(miss_km: np.ndarray, vmax_km: float) -> np.ndarray:
    """Map miss → (cyan → amber → red) tier for ensemble lines."""
    f = np.clip(miss_km / max(vmax_km, 1e-9), 0.0, 1.0)
    out = np.empty((len(f), 4))
    for i, ff in enumerate(f):
        if ff < 0.5:
            t = ff / 0.5
            r = 0x22 + (0xF5 - 0x22) * t
            g = 0xD3 + (0x9E - 0xD3) * t
            b = 0xEE + (0x0B - 0xEE) * t
        else:
            t = (ff - 0.5) / 0.5
            r = 0xF5 + (0xF4 - 0xF5) * t
            g = 0x9E + (0x3F - 0x9E) * t
            b = 0x0B + (0x5E - 0x0B) * t
        out[i] = (r / 255.0, g / 255.0, b / 255.0, 1.0)
    return out


def _try_save(ani, path: Path, fps: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        writer = FFMpegWriter(
            fps=fps, bitrate=3_600,
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
# Main animation
# ─────────────────────────────────────────────────────────────────────────────
def animate_envelope(*, fps: int = 30, seconds: float = 14.0,
                     n_seeds: int = 28) -> None:
    n_frames = int(round(fps * seconds))

    print(f"Pre-computing {n_seeds} Monte-Carlo arcs …")
    data = _build_ensemble(n_seeds=n_seeds, n_steps=240, tf_tu=2.0)

    t_d   = data["t_days"]
    Xnom  = data["X_nom_km"]            # (N, 3)
    dX    = data["dX_km"]               # (S, N, 3)
    geom  = data["geom"]
    N_t   = Xnom.shape[0]

    base_miss_km = np.linalg.norm(dX[:, -1, :], axis=1)
    base_med_km  = float(np.median(base_miss_km))

    # Cycle structure: each flight gets equal frame budget.
    n_cycles = len(CYCLES)
    cycle_len = n_frames // n_cycles                # frames per cycle
    n_frames  = cycle_len * n_cycles                # snap so cycles divide evenly

    # ── figure setup ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=_BG)

    # 3-D orbit panel — fills most of the figure; minimal axis decoration
    # keeps the focus on the comets and the Moon.
    ax = fig.add_axes([-0.060, 0.000, 0.860, 0.870], projection="3d")
    ax.set_facecolor(_BG)
    pane_rgba = (0.020, 0.031, 0.063, 0.85)
    grid_rgba = (0.10, 0.13, 0.25, 0.18)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(pane_rgba)
        axis._axinfo["grid"]["color"] = grid_rgba
        axis._axinfo["grid"]["linewidth"] = 0.25
        axis.set_ticklabels([])
        axis.label.set_visible(False)
    ax.tick_params(colors=_DIM, labelsize=0, length=0)

    # Axis bounds: pad around the σ=5 envelope so worst-case still fits.
    pad_factor = 5.5
    bounds_low  = (Xnom + pad_factor * dX.min(axis=0).min(axis=0))
    bounds_high = (Xnom + pad_factor * dX.max(axis=0).max(axis=0))
    pts = np.vstack([bounds_low, bounds_high, Xnom,
                     geom["moon_km"][None, :]])
    lo = pts.min(axis=0); hi = pts.max(axis=0)
    pad = (hi - lo) * 0.08 + 2_000
    lo -= pad; hi += pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    try:
        rng = hi - lo
        ax.set_box_aspect(tuple(rng / rng.max()))
    except Exception:
        pass

    _stars_3d(ax, n=200, alpha=0.15)

    # Moon — textured if available, sized so it anchors the scene without
    # dominating the trajectory swarm. The body sits a touch off-screen
    # to the right of the comet activity, like a planetarium backdrop.
    moon_radius = 5_000.0
    moon_tex = REPO_ROOT / "results" / "seeds" / "moon_texture.jpg"
    if moon_tex.exists():
        _draw_textured_sphere(ax, geom["moon_km"], moon_radius,
                              moon_tex, n=72, alpha=1.0,
                              rotate_lon_deg=200.0)
    else:
        _draw_sphere(ax, geom["moon_km"], moon_radius, _MOON_C, alpha=0.92)
    ax.text(geom["moon_km"][0] + moon_radius * 1.3,
            geom["moon_km"][1] + moon_radius * 1.3,
            moon_radius * 1.3,
            "Moon", color=_MOON_C, fontsize=6.5, zorder=10, alpha=0.85)

    # L1 marker — small, unobtrusive; label placed above so it doesn't
    # collide with the "nominal endpoint" annotation under low-elevation
    # camera angles.
    ax.scatter([geom["L1_km"]], [0], [0], marker="x", s=22,
               color=_DIM, alpha=0.70, zorder=4, depthshade=False)
    ax.text(geom["L1_km"], -6_000, 0, "L1",
            color=_DIM, fontsize=6.0, zorder=4)

    # Faint dashed nominal arc — "where we'd be" reference.
    ax.plot(Xnom[:, 0], Xnom[:, 1], Xnom[:, 2],
            color=_GREEN, lw=0.8, alpha=0.30, ls="--", zorder=2)

    # Nominal endpoint star — label tucked far enough to clear the Moon.
    target = Xnom[-1]
    nom_star = ax.scatter([target[0]], [target[1]], [target[2]],
                          marker="*", s=160,
                          color=_AMBER, edgecolors=_WHITE, linewidths=0.5,
                          zorder=8, depthshade=False)
    ax.text(target[0] - 12_000, target[1] + 4_000, target[2] + 4_000,
            "nominal endpoint", color=_AMBER, fontsize=6.3, zorder=9,
            ha="right")

    # Live ensemble lines — comet bodies (one Line3D per seed × glow/main).
    ens_lines = []
    for _ in range(n_seeds):
        glow, = ax.plot([], [], [], lw=2.4, alpha=0.10, zorder=4,
                        solid_capstyle="round")
        main, = ax.plot([], [], [], lw=0.95, alpha=0.85, zorder=5,
                        solid_capstyle="round")
        ens_lines.append((glow, main))

    # Comet heads — bright dots that chase along each arc this cycle.
    head_xs = np.zeros(n_seeds)
    head_ys = np.zeros(n_seeds)
    head_zs = np.zeros(n_seeds)
    head_dots = ax.scatter(head_xs, head_ys, head_zs,
                           s=22, color=_CYAN, zorder=9,
                           edgecolors=_WHITE, linewidths=0.45,
                           depthshade=False)

    # ── inset: zoomed terminal-miss cloud (2-D, projected onto x-y) ───────
    inset = fig.add_axes([0.745, 0.085, 0.235, 0.265])
    inset.set_facecolor("#080B17")
    for sp in inset.spines.values():
        sp.set_edgecolor(_CYAN); sp.set_linewidth(0.9); sp.set_alpha(0.55)
    inset.tick_params(colors=_DIM, labelsize=5.5,
                      length=1.6, width=0.4, pad=1)
    inset.set_aspect("equal")
    inset.axhline(0, color=_DIM, lw=0.5, alpha=0.5)
    inset.axvline(0, color=_DIM, lw=0.5, alpha=0.5)
    inset.scatter([0], [0], marker="*", s=70,
                  color=_AMBER, edgecolors=_WHITE, linewidths=0.4, zorder=5)
    term_scatter = inset.scatter(np.zeros(n_seeds), np.zeros(n_seeds),
                                 s=20, color=_CYAN, alpha=0.85, zorder=6,
                                 edgecolors=_WHITE, linewidths=0.3)
    inset.set_xticks([]); inset.set_yticks([])

    inset.text(0.5, 1.06, "TERMINAL MISS  ·  ZOOM",
               transform=inset.transAxes,
               color=_TEXT, fontsize=6.5, ha="center", va="bottom",
               fontweight="bold", alpha=0.85)
    inset_scale_text = inset.text(
        0.97, 0.04, "", transform=inset.transAxes,
        color=_DIM, fontsize=6.0, ha="right", va="bottom", alpha=0.85,
    )
    env_ring = Circle((0, 0), radius=base_med_km, fill=False,
                      edgecolor=_VIOLET, lw=1.0, ls="--", alpha=0.55,
                      zorder=4)
    inset.add_patch(env_ring)

    # ── header overlay ────────────────────────────────────────────────────
    title_t = fig.text(0.020, 0.952, "ROBUSTNESS  ENVELOPE",
                       color=_TEXT, fontsize=10.5, fontweight="bold",
                       ha="left", va="center")
    fig.text(0.020, 0.922,
             "3-D CR3BP · 28-seed Monte Carlo · σ-stress sweep",
             color=_DIM, fontsize=6.5, ha="left", va="center")
    phase_text = fig.text(0.020, 0.898, "",
                          color=_CYAN, fontsize=7.5, ha="left", va="center",
                          fontweight="bold")

    # σ_pix gauge (top-right)
    gauge_x0, gauge_y0 = 0.560, 0.937
    gauge_w,  gauge_h  = 0.340, 0.018
    fig.patches.append(FancyBboxPatch(
        (gauge_x0 - 0.004, gauge_y0 - 0.005),
        gauge_w + 0.008, gauge_h + 0.010,
        boxstyle="round,pad=0.001,rounding_size=0.004",
        transform=fig.transFigure,
        linewidth=0.7, edgecolor=_BORDER, facecolor=_PANEL,
    ))
    sigma_track_max = 5.0
    base_x = gauge_x0 + gauge_w * (1.0 / sigma_track_max)
    fig.patches.append(Rectangle(
        (base_x - 0.0007, gauge_y0 - 0.003),
        0.0014, gauge_h + 0.006,
        transform=fig.transFigure,
        linewidth=0, facecolor=_CYAN, alpha=0.90,
    ))
    fig.text(base_x, gauge_y0 - 0.015, "σ=1",
             color=_CYAN, fontsize=5.0, ha="center", va="top",
             fontweight="bold")
    for tk in (2, 3, 4, 5):
        tx = gauge_x0 + gauge_w * (tk / sigma_track_max)
        fig.patches.append(Rectangle(
            (tx - 0.00035, gauge_y0 + gauge_h - 0.003),
            0.0007, 0.003,
            transform=fig.transFigure,
            linewidth=0, facecolor=_DIM, alpha=0.70,
        ))
    sigma_fill = Rectangle(
        (gauge_x0, gauge_y0), 0.0001, gauge_h,
        transform=fig.transFigure,
        linewidth=0, facecolor=_AMBER, alpha=0.95,
    )
    fig.patches.append(sigma_fill)

    sigma_text = fig.text(gauge_x0, gauge_y0 + gauge_h + 0.012, "",
                          color=_AMBER, fontsize=7.5, ha="left", va="bottom",
                          fontweight="bold")
    miss_text = fig.text(gauge_x0 + gauge_w, gauge_y0 - 0.030, "",
                         color=_TEXT, fontsize=7.5, ha="right", va="top",
                         fontweight="bold")

    # Flight-counter chip (lower-left of figure)
    flight_text = fig.text(0.020, 0.062, "",
                           color=_DIM, fontsize=6.5, ha="left", va="center",
                           fontweight="bold", alpha=0.85)

    for t in (title_t, sigma_text, miss_text, phase_text):
        t.set_path_effects([
            patheffects.Stroke(linewidth=2.0, foreground=_BG),
            patheffects.Normal(),
        ])

    # ── camera plan ───────────────────────────────────────────────────────
    # Slightly elevated bird's-eye angle so the trajectory pattern reads
    # over the top of the Moon. Slow continuous orbit (~22° swing across
    # the full video, gentle 3° elevation breathing) — eased ends so the
    # opening and closing frames feel still.
    elev_base = 42.0
    azim_base = -58.0
    azim_span = 22.0
    elev_amp  = 3.0

    # ── update ────────────────────────────────────────────────────────────
    def init():
        for glow, main in ens_lines:
            glow.set_data([], []); glow.set_3d_properties([])
            main.set_data([], []); main.set_3d_properties([])
        head_dots._offsets3d = ([], [], [])
        term_scatter.set_offsets(np.empty((0, 2)))
        sigma_fill.set_width(0.0001)
        sigma_text.set_text(""); miss_text.set_text("")
        phase_text.set_text(""); inset_scale_text.set_text("")
        flight_text.set_text("")
        return ()

    def update(frame):
        # Cycle book-keeping
        cyc_idx   = min(frame // cycle_len, n_cycles - 1)
        cyc_frame = frame - cyc_idx * cycle_len
        cyc_phase = cyc_frame / max(cycle_len - 1, 1)        # 0 → 1 in cycle

        label, sigma_target, regime_col = CYCLES[cyc_idx]

        # σ within cycle: a brief 12% ease-in from the previous cycle's σ
        # avoids jolts; otherwise sigma stays parked at the cycle's value.
        prev_sigma = (CYCLES[cyc_idx - 1][1] if cyc_idx > 0
                      else sigma_target)
        ramp = _smoothstep(cyc_phase / 0.12) if cyc_phase < 0.12 else 1.0
        sig_now = prev_sigma + (sigma_target - prev_sigma) * ramp

        # Comet head index along the arc this cycle.
        # Phase 0.10 → 0.85 of the cycle is the active flight; before that
        # the heads sit at L1, after that they linger at terminal so the
        # cluster reads. Eased for a cinematic launch/arrival.
        if cyc_phase < 0.10:
            flight = 0.0
        elif cyc_phase > 0.85:
            flight = 1.0
        else:
            flight = _smoothstep((cyc_phase - 0.10) / 0.75)
        head_idx = int(round(flight * (N_t - 1)))

        # σ-scaled deviations & per-seed terminal miss
        Xview = Xnom[None, :, :] + sig_now * dX
        miss_term = np.linalg.norm(Xview[:, -1, :] - Xnom[-1, :], axis=1)
        # vmax tuned so σ≈2 reads amber, σ≥4 reads red.
        cols = _miss_color(miss_term, vmax_km=base_med_km * 4.5)

        # Comet trail = arc segment up to head_idx; 28 lines updated.
        for s, (glow, main) in enumerate(ens_lines):
            xs = Xview[s, :head_idx + 1, 0]
            ys = Xview[s, :head_idx + 1, 1]
            zs = Xview[s, :head_idx + 1, 2]
            glow.set_data(xs, ys); glow.set_3d_properties(zs)
            main.set_data(xs, ys); main.set_3d_properties(zs)
            glow.set_color(cols[s]); main.set_color(cols[s])

        # Head dots
        head_dots._offsets3d = (Xview[:, head_idx, 0],
                                Xview[:, head_idx, 1],
                                Xview[:, head_idx, 2])
        head_dots.set_facecolor(cols)
        # Heads pulse (slightly larger as they accelerate)
        head_dots.set_sizes(np.full(n_seeds, 18 + 28 * (1.0 - abs(2*flight - 1.0))))

        # Inset: terminal-miss cloud (always the σ-correct cluster)
        rel = Xview[:, -1, :2] - Xnom[-1, :2]
        term_scatter.set_offsets(rel)
        term_scatter.set_facecolor(cols)
        # Reveal the cluster gradually as the comets fly toward terminal
        term_scatter.set_alpha(0.10 + 0.85 * flight)

        p95 = float(np.quantile(miss_term, 0.95))
        env_ring.set_radius(p95)
        zoom_half = max(2.5 * p95, 1.4 * base_med_km)
        inset.set_xlim(-zoom_half, zoom_half)
        inset.set_ylim(-zoom_half, zoom_half)
        inset_scale_text.set_text(f"±{zoom_half:,.0f} km")

        # σ_pix gauge fill
        frac = float(np.clip(sig_now / sigma_track_max, 0.0, 1.0))
        sigma_fill.set_width(gauge_w * frac)
        if sig_now <= 1.05:
            fill_col = _CYAN if sig_now <= 0.7 else _GREEN
        elif sig_now <= 2.5:
            fill_col = _AMBER
        else:
            fill_col = _RED
        sigma_fill.set_facecolor(fill_col)
        sigma_text.set_text(f"σ_pix = {sig_now:4.2f} px")
        sigma_text.set_color(fill_col)

        med_km = float(np.median(miss_term))
        miss_text.set_text(f"terminal miss (median) = {med_km:5.1f} km")

        phase_text.set_text(label)
        phase_text.set_color(regime_col)

        if sig_now > 3.5:
            title_t.set_color(_RED)
        elif sig_now < 0.7:
            title_t.set_color(_GREEN)
        else:
            title_t.set_color(_TEXT)

        flight_text.set_text(
            f"FLIGHT {cyc_idx + 1} / {n_cycles}   ·   "
            f"sim t = {t_d[head_idx]:4.1f} d"
        )

        # Slow camera orbit — single eased pass across all flights so the
        # parallax is continuous (no per-cycle resets).
        global_phase = frame / max(n_frames - 1, 1)
        eased = _smoothstep(global_phase)
        ax.view_init(
            elev=elev_base + elev_amp * np.sin(2 * np.pi * global_phase),
            azim=azim_base + azim_span * eased,
        )

        return ()

    ani = FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        blit=False, interval=1000 // fps,
    )

    out = OUT_DIR / "anim_13_robustness_envelope.mp4"
    _try_save(ani, out, fps)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fps",     type=int,   default=30)
    p.add_argument("--seconds", type=float, default=14.0)
    p.add_argument("--seeds",   type=int,   default=28)
    args = p.parse_args()
    animate_envelope(fps=args.fps, seconds=args.seconds, n_seeds=args.seeds)
