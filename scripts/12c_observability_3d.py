"""3D observability animation for slide 06 — driven by real IEKF run data.

Reads results/active_tracking/07_active_tracking_active.npz and tells a four-act
story in a single Moon-centered 3D scene:

  ACT 1  ·  ONE BEARING IS NOT ENOUGH
            spacecraft frozen at t0; one LOS ray from camera to Moon
            extended past the body as a fading dashed line — depth
            along the ray is ambiguous.  Filter's 1-σ position
            uncertainty drawn as a translucent sphere at the estimate.

  ACT 2  ·  MOTION ADDS GEOMETRY
            spacecraft marches along the truth trajectory; new bearings
            are sampled from each fresh vantage; older rays fade away.

  ACT 3  ·  PARALLAX CONSTRAINS RANGE
            continued motion; parallax between bearings sweeps the
            target; the 1-σ ball visibly contracts.

  ACT 4  ·  FILTER CONVERGES
            estimate trail (amber) converges toward the truth trail
            (cyan); uncertainty ball is now nearly invisible.

Run:
  python scripts/12c_observability_3d.py
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
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# CR3BP nondimensional → physical
DU_KM = 384_400.0
TU_DAY = 4.343

# Palette (consistent with bearing_concept_visual + animate_phases_2_3)
BG       = "#050B16"
PANEL    = "#0B1220"
BORDER   = "#1A2744"
TEXT     = "#EAF1FB"
DIM      = "#A9B6CA"
TRUTH_C  = "#33D1FF"   # cyan — truth
LOS_C    = "#7FE9FF"   # lighter cyan — LOS rays
EST_C    = "#F6A91A"   # amber — estimate
MOON_C   = "#D5D9E3"
UNC_C    = "#8B5CF6"   # violet — 1-σ envelope
GREEN    = "#10B981"
RED      = "#FF4D6D"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=Path,
        default=Path("results/active_tracking/07_active_tracking_active.npz"),
        help="IEKF run output to drive the animation.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/videos/12_observability_3d.mp4"),
    )
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seconds-act1", type=float, default=3.0)
    p.add_argument("--seconds-act2", type=float, default=4.5)
    p.add_argument("--seconds-act3", type=float, default=4.5)
    p.add_argument("--seconds-act4", type=float, default=3.0)
    p.add_argument("--ray-sample-stride", type=int, default=10,
                   help="Emit one persistent LOS ray every N truth samples.")
    p.add_argument("--max-rays", type=int, default=14,
                   help="Cap on persistent fan rays. Fewer rays → cleaner read.")
    p.add_argument("--dpi", type=int, default=170)
    p.add_argument("--gif", action="store_true")
    return p.parse_args()


def _ease(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _style_3d(ax):
    """Dark theme for a 3D matplotlib axis."""
    ax.set_facecolor(PANEL)
    ax.xaxis.set_pane_color((0, 0, 0, 0))
    ax.yaxis.set_pane_color((0, 0, 0, 0))
    ax.zaxis.set_pane_color((0, 0, 0, 0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color(BORDER)
        axis.label.set_color(DIM)
        axis._axinfo["grid"]["color"] = BORDER
        axis._axinfo["grid"]["linewidth"] = 0.4
        axis._axinfo["grid"]["alpha"] = 0.55
    ax.tick_params(colors=DIM, which="both", labelsize=7)


def _moon_surface(ax, center, radius, n=28):
    u, v = np.mgrid[0:2 * np.pi:n * 2j, 0:np.pi:n * 1j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=MOON_C, alpha=0.92,
                    linewidth=0, antialiased=True, zorder=5,
                    rcount=n, ccount=n * 2)


def _sphere_solid(ax, center, radius, color, alpha=0.18, n=18):
    """Translucent solid ball used as the uncertainty envelope.

    A solid surface reads cleaner than a wireframe at oblique camera angles
    (a wireframe degenerates into busy crossing curves).
    """
    u, v = np.mgrid[0:2 * np.pi:n * 2j, 0:np.pi:n * 1j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    return ax.plot_surface(x, y, z, color=color, alpha=alpha,
                           linewidth=0, antialiased=True, zorder=6,
                           rcount=n, ccount=n * 2)


def _starfield(ax, xlim, ylim, zlim, n=400, seed=7):
    rng = np.random.default_rng(seed)
    pts = np.column_stack([
        rng.uniform(*xlim, n),
        rng.uniform(*ylim, n),
        rng.uniform(*zlim, n),
    ])
    sizes = rng.uniform(0.2, 1.6, n)
    ax.scatter(*pts.T, s=sizes, color="white", alpha=0.18, zorder=0,
               depthshade=False)


# ── data loader ─────────────────────────────────────────────────────────────
def load_run(path: Path):
    if not path.exists():
        raise SystemExit(
            f"Missing IEKF data file: {path}\n"
            "Run scripts/07_active_tracking.py first."
        )
    d = np.load(path, allow_pickle=True)
    t = d["t_hist"]
    r_sc = d["r_sc_true_hist"]            # (N, 3) DU
    r_body = d["r_body_true_hist"]        # (N, 3) DU (Moon)
    los_true = d["los_true_hist"]
    xhat = d["xhat_hist"]                 # (N, 6)
    Pdiag = d["Pdiag_hist"]               # (N, 6)
    sigma_pos = np.sqrt(np.maximum(Pdiag[:, :3], 0.0)).mean(axis=1)
    return dict(t=t, r_sc=r_sc, r_body=r_body, los_true=los_true,
                xhat=xhat, sigma_pos=sigma_pos)


# ── main ────────────────────────────────────────────────────────────────────
def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    run = load_run(args.data)
    t = run["t"]                          # TU
    r_sc = run["r_sc"]                    # DU
    r_body_hist = run["r_body"]
    xhat = run["xhat"][:, :3]
    sigma_pos = run["sigma_pos"]          # DU
    n_samples = len(t)

    # Re-express everything in the Moon-centered frame so the Moon sits at
    # the origin of the scene — this keeps the visual stable as time
    # advances even if the Moon's CR3BP coordinates shift slightly.
    r_sc_m = r_sc - r_body_hist           # spacecraft, Moon-centered (DU)
    r_est_m = xhat - r_body_hist          # estimate, Moon-centered (DU)

    # Position-error history (km) for the inset
    err_km_hist = np.linalg.norm(xhat - r_sc, axis=1) * DU_KM

    # Axis bounds chosen so the whole arc sits in view, with margin
    pad = 0.012
    xs = np.concatenate([r_sc_m[:, 0], r_est_m[:, 0], [0.0]])
    ys = np.concatenate([r_sc_m[:, 1], r_est_m[:, 1], [0.0]])
    zs = np.concatenate([r_sc_m[:, 2], r_est_m[:, 2], [0.0]])
    xlim = (xs.min() - pad, xs.max() + pad)
    ylim = (ys.min() - pad, ys.max() + pad)
    span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    zlim = (-span / 4, span / 4)

    moon_radius_du = 1737.4 / DU_KM       # ≈ 0.00452 DU

    # ── frame budget ────────────────────────────────────────────────────────
    fps = args.fps
    n1 = int(round(args.seconds_act1 * fps))
    n2 = int(round(args.seconds_act2 * fps))
    n3 = int(round(args.seconds_act3 * fps))
    n4 = int(round(args.seconds_act4 * fps))
    N = n1 + n2 + n3 + n4

    # Map animation frame → truth-sample index for Acts 2/3
    # Act 1 holds at idx=0; Acts 2/3 sweep idx 0 → n_samples-1
    motion_frames = n2 + n3
    motion_idx = np.linspace(0, n_samples - 1, motion_frames).astype(int)

    # Sample which truth indices emit a persistent LOS ray
    ray_idx = list(range(0, n_samples, args.ray_sample_stride))
    if len(ray_idx) > args.max_rays:
        ray_idx = list(np.linspace(0, n_samples - 1,
                                   args.max_rays).astype(int))
    ray_idx_sorted = sorted(set(int(i) for i in ray_idx))

    # ── figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13.0, 7.3), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_facecolor(PANEL)
    fig.patch.set_facecolor(BG)
    # Constrain the 3D axis so it lives in the left ~75 % of the figure,
    # leaving the right side clear for the position-error inset.
    ax.set_position([0.04, 0.08, 0.72, 0.84])

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0],
                       zlim[1] - zlim[0]))
    _style_3d(ax)
    ax.set_xlabel("x  [DU]", color=DIM)
    ax.set_ylabel("y  [DU]", color=DIM)
    ax.set_zlabel("z  [DU]", color=DIM)

    _starfield(ax, xlim, ylim, zlim)

    # Static target body at origin
    _moon_surface(ax, (0.0, 0.0, 0.0), moon_radius_du)

    # Title and act overlays (figure-relative so they don't rotate with the
    # 3D scene).  We use fig.text so they live above the 3D axes.
    title = fig.text(0.5, 0.945,
                     "Bearing-Only Observability  ·  3D Cislunar Arc",
                     ha="center", va="center", color=TEXT,
                     fontsize=15, fontweight="bold")
    badge = fig.text(0.05, 0.91, "ACT 1  ·  ONE BEARING IS NOT ENOUGH",
                     ha="left", va="center", color=TRUTH_C,
                     fontsize=11, fontweight="bold")
    footer = fig.text(0.5, 0.05, "",
                      ha="center", va="center", color=DIM, fontsize=10.5,
                      style="italic")
    sigma_txt = fig.text(0.95, 0.91, "", ha="right", va="center",
                         color=UNC_C, fontsize=10.5, fontweight="bold")
    range_txt = fig.text(0.95, 0.875, "", ha="right", va="center",
                         color=DIM, fontsize=9.5)

    # Legend (manual, to avoid mpl placing it inside the 3D box)
    leg_ax = fig.add_axes([0.04, 0.03, 0.30, 0.07])
    leg_ax.set_facecolor((0, 0, 0, 0))
    leg_ax.axis("off")
    legend_items = [
        ("truth trajectory",        TRUTH_C, "-"),
        ("IEKF estimate",           EST_C,   "-"),
        ("LOS bearings û",          LOS_C,   "-"),
        ("1σ position uncertainty", UNC_C,   ":"),
    ]
    for i, (label, c, ls) in enumerate(legend_items):
        leg_ax.plot([0.02, 0.10], [0.85 - i * 0.27] * 2,
                    color=c, linestyle=ls, linewidth=2.0, alpha=0.9)
        leg_ax.text(0.13, 0.85 - i * 0.27, label, color=TEXT,
                    fontsize=9, va="center")
    leg_ax.set_xlim(0, 1); leg_ax.set_ylim(0, 1)

    # Position-error inset — sits in the right column carved out for it.
    err_ax = fig.add_axes([0.80, 0.20, 0.18, 0.20])
    err_ax.set_facecolor(PANEL)
    for spine in err_ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.7)
    err_ax.tick_params(colors=DIM, which="both", labelsize=6.5,
                       pad=1.5)
    err_ax.grid(True, color=BORDER, lw=0.4, linestyle="--", alpha=0.6)
    err_ax.set_yscale("log")
    err_ax.set_xlim(0, t[-1] * TU_DAY)
    y_lo = max(err_km_hist.min() * 0.6, 0.5)
    y_hi = err_km_hist.max() * 1.6
    err_ax.set_ylim(y_lo, y_hi)
    err_ax.set_xlabel("days", color=DIM, fontsize=7, labelpad=1)
    err_ax.set_ylabel("|x̂−x|  [km]", color=DIM, fontsize=7, labelpad=1)
    err_ax.set_title("position error", color=TEXT, fontsize=8.5,
                     pad=2, fontweight="bold")
    err_line, = err_ax.plot([], [], color=EST_C, lw=1.4, alpha=0.95)
    err_dot, = err_ax.plot([], [], "o", color=EST_C,
                           markeredgecolor="white",
                           markeredgewidth=0.6, markersize=4)

    # ── persistent artists, mutated per frame ───────────────────────────────
    truth_trail, = ax.plot([], [], [], color=TRUTH_C, lw=1.6,
                           alpha=0.95, zorder=4)
    truth_glow, = ax.plot([], [], [], color=TRUTH_C, lw=5.0,
                          alpha=0.10, zorder=3)

    est_trail, = ax.plot([], [], [], color=EST_C, lw=1.4,
                         alpha=0.0, zorder=4)
    est_glow, = ax.plot([], [], [], color=EST_C, lw=4.0,
                        alpha=0.0, zorder=3)

    sc_marker = ax.scatter([], [], [], s=70, color=TRUTH_C,
                           edgecolors="white", linewidths=0.8,
                           depthshade=False, zorder=8)
    est_marker = ax.scatter([], [], [], s=42, color=EST_C,
                            edgecolors=BG, linewidths=0.6,
                            depthshade=False, alpha=0.0, zorder=8)

    # Live (current-frame) primary LOS — drawn fresh each update
    primary_los_artists: list = []

    # Persistent fan of past LOS rays, kept as a Line3DCollection for speed
    fan_segments: list = []
    fan_alphas: list = []
    fan_indices_emitted: set[int] = set()
    # Seed with one tiny invisible segment so add_collection3d can autoscale.
    _dummy_seg = [[(0.0, 0.0, 0.0), (1e-9, 1e-9, 1e-9)]]
    fan_collection = Line3DCollection(_dummy_seg,
                                      colors=[(0, 0, 0, 0)],
                                      linewidths=[0.0],
                                      zorder=3)
    ax.add_collection3d(fan_collection)
    fan_collection.set_segments([])  # clear after axes have been set up

    # Uncertainty wireframe — replaced each frame (size changes)
    unc_artists: list = []

    # Phase-1 "infinite solutions" extended dashed ray
    ext_artist: list = []

    def clear(lst):
        for a in lst:
            try:
                a.remove()
            except Exception:
                pass
        lst.clear()

    # Visual amplification: actual 1-σ position uncertainty drops from
    # ~400 km to ~3 km over the run, which is microscopic vs. the ~100 000 km
    # scene span.  Render at 3-σ × visual_gain so the contraction reads.
    visual_gain = 12.0

    def draw_uncertainty(center_xyz, sigma_du):
        clear(unc_artists)
        radius = 3.0 * sigma_du * visual_gain
        if radius <= 0 or not np.isfinite(radius):
            return
        surf = _sphere_solid(ax, center_xyz, radius, UNC_C,
                             alpha=0.20, n=14)
        unc_artists.append(surf)

    def draw_primary_los(origin, target, alpha=0.95, lw=2.2, color=LOS_C):
        clear(primary_los_artists)
        # Glow: thick faint pass + bright thin pass
        g, = ax.plot([origin[0], target[0]],
                     [origin[1], target[1]],
                     [origin[2], target[2]],
                     color=color, lw=lw * 3.5, alpha=alpha * 0.10, zorder=3)
        primary_los_artists.append(g)
        m, = ax.plot([origin[0], target[0]],
                     [origin[1], target[1]],
                     [origin[2], target[2]],
                     color=color, lw=lw, alpha=alpha, zorder=4)
        primary_los_artists.append(m)

    def draw_extended_ray(origin, direction, alpha=0.7, length_du=0.2):
        clear(ext_artist)
        if alpha <= 0:
            return
        end = (origin[0] + direction[0] * length_du,
               origin[1] + direction[1] * length_du,
               origin[2] + direction[2] * length_du)
        # short dashes to differentiate from solid LOS
        line, = ax.plot([origin[0], end[0]],
                        [origin[1], end[1]],
                        [origin[2], end[2]],
                        color=RED, lw=1.4, alpha=alpha,
                        linestyle=(0, (4, 3)), zorder=4)
        ext_artist.append(line)

    def update_fan_collection():
        if not fan_segments:
            fan_collection.set_segments([])
            return
        fan_collection.set_segments(fan_segments)
        # Past rays read as a quiet trail behind the bright current bearing.
        fan_collection.set_color([(*matplotlib.colors.to_rgb(LOS_C), a)
                                  for a in fan_alphas])
        fan_collection.set_linewidths([0.6 + 0.4 * a for a in fan_alphas])

    def emit_ray(origin, target):
        fan_segments.append([tuple(origin), tuple(target)])
        fan_alphas.append(0.7)

    def age_fan(decay=0.96):
        # Aggressive decay keeps the current bearing visually dominant.
        for i in range(len(fan_alphas)):
            fan_alphas[i] = max(fan_alphas[i] * decay, 0.07)

    # Camera path: very subtle — fixed elevation, slow azimuth drift
    def camera_for(frame: int):
        a = frame / max(N - 1, 1)
        elev = 24.0 + 3.0 * np.sin(a * 2 * np.pi)
        azim = -55.0 + 18.0 * a
        return elev, azim

    # ── per-frame update ────────────────────────────────────────────────────
    def update(frame: int):
        # Resolve act + local progress
        if frame < n1:
            act, local = 1, frame / max(n1 - 1, 1)
            idx = 0
        elif frame < n1 + n2:
            act, local = 2, (frame - n1) / max(n2 - 1, 1)
            idx = motion_idx[frame - n1]
        elif frame < n1 + n2 + n3:
            act, local = 3, (frame - n1 - n2) / max(n3 - 1, 1)
            idx = motion_idx[frame - n1]
        else:
            act, local = 4, (frame - n1 - n2 - n3) / max(n4 - 1, 1)
            idx = n_samples - 1

        sc = r_sc_m[idx]
        est = r_est_m[idx]
        moon = (0.0, 0.0, 0.0)
        sigma = sigma_pos[idx]
        rng_km = float(np.linalg.norm(r_sc[idx] - r_body_hist[idx])) * DU_KM

        # truth trail up to current idx
        truth_trail.set_data_3d(r_sc_m[:idx + 1, 0],
                                 r_sc_m[:idx + 1, 1],
                                 r_sc_m[:idx + 1, 2])
        truth_glow.set_data_3d(r_sc_m[:idx + 1, 0],
                                r_sc_m[:idx + 1, 1],
                                r_sc_m[:idx + 1, 2])
        sc_marker._offsets3d = ([sc[0]], [sc[1]], [sc[2]])

        # camera dolly
        elev, azim = camera_for(frame)
        ax.view_init(elev=elev, azim=azim)

        # uncertainty sphere (always shown after the first frame)
        draw_uncertainty(est, sigma)
        sigma_txt.set_text(f"1σ pos = {sigma * DU_KM:6.0f} km")
        range_txt.set_text(f"range to Moon = {rng_km:6.0f} km")

        # position-error inset (filled progressively)
        err_line.set_data(t[:idx + 1] * TU_DAY,
                          err_km_hist[:idx + 1])
        err_dot.set_data([t[idx] * TU_DAY], [err_km_hist[idx]])

        # LOS direction (truth) at current sample
        d = moon - sc
        d_norm = d / (np.linalg.norm(d) + 1e-12)

        if act == 1:
            badge.set_text("ACT 1  ·  ONE BEARING IS NOT ENOUGH")
            badge.set_color(TRUTH_C)
            footer.set_text(
                "A single bearing fixes direction, but depth along the "
                "ray remains ambiguous.")

            # primary LOS appears via easing, then extended past the Moon
            t1 = _ease(np.clip(local / 0.55, 0, 1))
            tip = (sc[0] + d[0] * t1, sc[1] + d[1] * t1, sc[2] + d[2] * t1)
            draw_primary_los(sc, tip, alpha=0.95)

            # extended dashed beyond the Moon, fading in over second half
            t2 = _ease(np.clip((local - 0.45) / 0.5, 0, 1))
            draw_extended_ray(moon, d_norm, alpha=0.78 * t2,
                              length_du=span * 0.45)

            # estimate trail hidden in act 1
            est_trail.set_alpha(0.0); est_glow.set_alpha(0.0)
            est_marker.set_alpha(0.0)

        elif act in (2, 3):
            if act == 2:
                badge.set_text("ACT 2  ·  MOTION ADDS GEOMETRY")
                badge.set_color(GREEN)
                footer.set_text(
                    "Each new image provides a bearing from a different "
                    "vantage point.  Older rays fade as the spacecraft "
                    "moves.")
            else:
                badge.set_text("ACT 3  ·  PARALLAX CONSTRAINS RANGE")
                badge.set_color(EST_C)
                footer.set_text(
                    "Parallax between bearings constrains range, and the "
                    "1σ position uncertainty shrinks.")

            # primary LOS = current truth bearing to Moon
            draw_primary_los(sc, moon, alpha=0.95, lw=2.0)
            clear(ext_artist)

            # Emit any sampled rays we crossed this frame (threshold logic
            # handles the case where motion_idx jumps multiple steps per
            # frame — without this most rays would never fire).
            for k in ray_idx_sorted:
                if k <= idx and k not in fan_indices_emitted:
                    sc_k = r_sc_m[k]
                    emit_ray(sc_k, (0.0, 0.0, 0.0))
                    fan_indices_emitted.add(k)
            age_fan()
            update_fan_collection()

            # Estimate trail starts ghost-faint in Act 2, brighter in Act 3
            if act == 2:
                est_alpha = 0.25 * _ease(local)
            else:
                est_alpha = 0.25 + 0.45 * _ease(local)
            est_trail.set_data_3d(r_est_m[:idx + 1, 0],
                                   r_est_m[:idx + 1, 1],
                                   r_est_m[:idx + 1, 2])
            est_glow.set_data_3d(r_est_m[:idx + 1, 0],
                                  r_est_m[:idx + 1, 1],
                                  r_est_m[:idx + 1, 2])
            est_trail.set_alpha(est_alpha)
            est_glow.set_alpha(est_alpha * 0.5)
            est_marker.set_alpha(min(1.0, est_alpha + 0.35))
            est_marker._offsets3d = ([est[0]], [est[1]], [est[2]])

        else:  # act 4
            badge.set_text("ACT 4  ·  FILTER CONVERGES")
            badge.set_color(EST_C)
            footer.set_text(
                "With enough geometry, the IEKF estimate converges toward "
                "the truth.  Range becomes observable over time.")

            # Hold last frame's geometry
            draw_primary_los(sc, moon, alpha=0.95, lw=2.0)
            clear(ext_artist)
            update_fan_collection()  # fan stays as-is

            est_trail.set_data_3d(r_est_m[:idx + 1, 0],
                                   r_est_m[:idx + 1, 1],
                                   r_est_m[:idx + 1, 2])
            est_glow.set_data_3d(r_est_m[:idx + 1, 0],
                                  r_est_m[:idx + 1, 1],
                                  r_est_m[:idx + 1, 2])
            est_trail.set_alpha(1.0)
            est_glow.set_alpha(0.6)
            est_marker.set_alpha(1.0)
            est_marker._offsets3d = ([est[0]], [est[1]], [est[2]])

        return []

    # ── render ──────────────────────────────────────────────────────────────
    print(f"Total frames: {N}  ({n1}+{n2}+{n3}+{n4})  @ {fps} fps")
    print(f"Truth samples: {n_samples}  ·  rays emitted: {len(ray_idx)}")
    print(f"Output: {args.out}")

    anim = FuncAnimation(fig, update, frames=N, interval=1000 / fps,
                         blit=False)

    # libx264 + yuv420p requires even pixel dimensions; the scale filter
    # snaps to the nearest even width/height to handle odd-pixel renders.
    writer = FFMpegWriter(fps=fps, bitrate=5500, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p",
                                      "-preset", "medium",
                                      "-vf",
                                      "scale=trunc(iw/2)*2:trunc(ih/2)*2"])
    anim.save(args.out, writer=writer, dpi=args.dpi)
    print(f"  saved → {args.out}")

    if args.gif:
        gif_path = args.out.with_suffix(".gif")
        anim.save(gif_path,
                  writer=PillowWriter(fps=min(fps, 18)),
                  dpi=max(args.dpi // 2, 100))
        print(f"  saved → {gif_path}")

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
