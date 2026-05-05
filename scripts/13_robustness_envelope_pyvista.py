"""Cinematic PyVista demo for slide 13 — representative failure modes.

Renders a 4-scene 3D animation that illustrates the physical interpretation of
the robustness-envelope plots (no real MC data; representative geometry):

    Scene A · NOMINAL              filter tracks cleanly
    Scene B · sigma_pix sweep      bearing rays jitter, estimate drifts
    Scene C · t_c sweep            estimate "coasts" between sparse updates
    Scene D · 1-step delay         LOS points to a stale ghost target

Output: results/demos/anim_13_robustness_pyvista.mp4
Frame:  1280x960 (4:3, matches slide-13 placeholder 6.32" x 4.76").

Usage:
    python scripts/13_robustness_envelope_pyvista.py
    python scripts/13_robustness_envelope_pyvista.py --fps 30 --seconds 14
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
import imageio.v2 as imageio


# ── slide palette (matches slide 13) ──────────────────────────────────────────
BG_COLOR     = "#0d1320"   # close to the slide-13 panel fill, slightly lifted
COL_TRUTH    = "#FFFFFF"
COL_ESTIMATE = "#F59E0B"
COL_CHASER   = "#22D3EE"
COL_LOS      = "#22D3EE"
COL_ERROR    = "#F43F5E"
COL_GHOST    = "#F43F5E"
COL_TEXT     = "#DCE0EC"
COL_DIM      = "#7A8095"
COL_PULSE    = "#FCD34D"

# ── output ────────────────────────────────────────────────────────────────────
OUT_PATH = Path("results/demos/anim_13_robustness_pyvista.mp4")
WINDOW   = (1280, 960)            # 4:3, matches 6.32" x 4.76" slide panel

# Unicode-capable font for σ and → glyphs. VTK's built-in fonts (arial,
# courier, times) are Latin-1 only, so non-ASCII characters silently drop.
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]
UNICODE_FONT = next((p for p in _FONT_CANDIDATES if Path(p).exists()), None)


# ─────────────────────────────────────────────────────────────────────────────
# Scene plan
# ─────────────────────────────────────────────────────────────────────────────
TITLE = "FIG 13C  ·  REPRESENTATIVE ROBUSTNESS MODES"

SCENES = [
    # (label, sub-caption, duration_seconds, mode)
    ("Mode: nominal",
     "σ_pix = 1 px   t_c = 1.5",                                                  3.0, "nominal"),
    ("Mode: pixel-noise sweep",
     "σ_pix ↑   →   noisy bearing   →   miss ↑",                4.0, "sigma"),
    ("Mode: correction-time sweep",
     "t_c ↑   →   slower correction   →   estimate lags",                 4.0, "tc"),
    ("Mode: one-step measurement delay",
     "stale measurement   →   hard failure",                                       3.0, "delay"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Geometry: a representative cislunar arc for the *target*
# ─────────────────────────────────────────────────────────────────────────────
def build_trajectory(n_steps: int):
    """Compact 3D scene: target traces a small closed loop centered at origin,
    chaser sits below/behind on a roughly parallel loop.

    Keeping motion compact (and roughly periodic) means the action stays
    framed by a fixed camera; all dramatic content comes from estimate
    drift, LOS rays, pulses, and the ghost target — not from translation.

    Units are abstract scene-units; the video is captioned
    'representative seed - error visually amplified'.
    """
    s = np.linspace(0.0, 1.0, n_steps)

    # Closed-ish curving path (one slow loop across the whole video).
    theta = 2.0 * np.pi * 1.0 * s
    target = np.column_stack([
        1.6 * np.cos(theta),
        1.0 * np.sin(theta),
        0.35 * np.sin(2.0 * theta),
    ])

    # Chaser: same loop, smaller radius, offset below.
    chaser = np.column_stack([
        1.0 * np.cos(theta + 0.55),
        0.65 * np.sin(theta + 0.55),
        0.20 * np.sin(2.0 * theta + 0.55) - 1.85,
    ])

    return target, chaser


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame scene state
# ─────────────────────────────────────────────────────────────────────────────
def smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def scene_for_frame(frame_idx: int, frames_per_scene):
    """Return (scene_idx, phase_in_scene_0to1)."""
    cum = 0
    for k, n in enumerate(frames_per_scene):
        if frame_idx < cum + n:
            return k, (frame_idx - cum) / max(n - 1, 1)
        cum += n
    last = len(frames_per_scene) - 1
    return last, 1.0


# Visual error magnitude is intentionally bounded; this is a diagnostic
# illustration, not a literal MC trace. The disclaimer at the bottom of the
# frame says so.
NOMINAL_OFFSET    = 0.05   # almost overlapping
SIGMA_OFFSET_MIN  = 0.06
SIGMA_OFFSET_MAX  = 0.30


def estimate_position(target_i, target_lagged, mode, scene_phase,
                      *, sigma, tc, bias_dir, est_state):
    """Smooth, *mechanism-led* estimate position (no random spaghetti).

    nominal:  truth + tiny constant offset
    sigma  :  truth + smooth bias growing with σ (rays carry the noise)
    tc     :  estimate = truth lagged by a delay that grows with t_c
    delay  :  estimate is pulled toward the (stale) lagged truth
    """
    base = target_i

    if mode == "nominal":
        est_state["pulse"] = False
        return base + NOMINAL_OFFSET * bias_dir

    if mode == "sigma":
        # Smooth ramp with sigma; bias magnitude tracks sigma directly so
        # the audience reads "more pixel noise -> larger miss" without any
        # high-frequency content in the estimate path.
        frac = (sigma - 1.0) / 4.0   # 0 at σ=1 -> 1 at σ=5
        offset = SIGMA_OFFSET_MIN + (SIGMA_OFFSET_MAX - SIGMA_OFFSET_MIN) * frac
        est_state["pulse"] = False
        return base + offset * bias_dir

    if mode == "tc":
        # Estimate is the truth, lagged. Update pulses fire periodically;
        # spacing widens with t_c.
        update_period = max(int(round(6 * tc / 1.5)), 4)
        counter = est_state.setdefault("counter", 0)
        est_state["pulse"] = (counter % update_period == 0)
        est_state["counter"] = counter + 1
        return target_lagged

    if mode == "delay":
        # Estimate follows the stale (lagged) truth with a smoothing
        # filter so the motion is continuous; the LOS ray (drawn separately)
        # points to the same lagged target -> red ghost.
        last = est_state.get("last", target_lagged.copy())
        new_est = 0.6 * last + 0.4 * target_lagged
        est_state["last"] = new_est
        est_state["pulse"] = False
        return new_est

    return base


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: PyVista geometry
# ─────────────────────────────────────────────────────────────────────────────
def make_polyline(points: np.ndarray) -> pv.PolyData:
    if len(points) < 2:
        points = np.vstack([points, points + 1e-6])
    poly = pv.PolyData(points)
    cells = np.hstack([[len(points)], np.arange(len(points))]).astype(np.int64)
    poly.lines = cells
    return poly


def line_between(a: np.ndarray, b: np.ndarray) -> pv.PolyData:
    return pv.Line(a, b, resolution=1)


def jittered_endpoint(chaser_i, target_i, sigma, rng, gain=0.18):
    """Return a target endpoint perturbed perpendicular to the LOS direction.

    Used to draw a 'spray' of cyan rays in Scene B, replacing a physical cone.
    """
    los = target_i - chaser_i
    n = np.linalg.norm(los)
    if n < 1e-9:
        return target_i.copy()
    los_dir = los / n

    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(los_dir, up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(los_dir, up); e1 /= np.linalg.norm(e1)
    e2 = np.cross(los_dir, e1); e2 /= np.linalg.norm(e2)

    j1, j2 = rng.standard_normal(2)
    return target_i + gain * sigma * (j1 * e1 + j2 * e2)


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────
def render(*, fps: int = 30, seconds: float = 14.0,
           out_path: Path = OUT_PATH) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Scene budget
    total_frames = int(round(fps * seconds))
    raw_alloc = np.array([s[2] for s in SCENES], dtype=float)
    raw_alloc *= total_frames / raw_alloc.sum()
    frames_per_scene = [max(int(round(x)), 1) for x in raw_alloc]
    # Ensure exact total
    diff = total_frames - sum(frames_per_scene)
    frames_per_scene[-1] += diff
    n_frames = sum(frames_per_scene)

    # Pre-build trajectory
    n_steps = n_frames
    target, chaser = build_trajectory(n_steps)

    # Compute estimate trajectory frame-by-frame so per-scene state carries
    # forward (drift accumulators, coasting velocity, etc.).
    estimate = np.zeros_like(target)
    pulses = np.zeros(n_steps, dtype=bool)
    sigmas = np.zeros(n_steps)
    tcs    = np.zeros(n_steps)
    modes  = np.empty(n_steps, dtype=object)
    est_state = {}

    # Lag table for tc / delay scenes
    tc_lag_max    = max(int(0.04 * n_steps), 6)
    delay_lag     = max(int(0.06 * n_steps), 6)

    # Bias direction for the smooth estimate offset (sigma scene). Pointing
    # mostly out of the orbit plane keeps the offset visible from the
    # camera angle we'll fly.
    bias_dir = np.array([0.30, -0.45, 0.85])
    bias_dir /= np.linalg.norm(bias_dir)

    for i in range(n_steps):
        scene_idx, scene_phase = scene_for_frame(i, frames_per_scene)
        label, caption, _, mode = SCENES[scene_idx]
        modes[i] = mode

        if mode == "nominal":
            sigma = 1.0
            tc    = 1.5
        elif mode == "sigma":
            sigma = 1.0 + 4.0 * scene_phase   # 1 -> 5
            tc    = 1.5
        elif mode == "tc":
            sigma = 1.5
            tc    = 1.5 + 1.5 * scene_phase   # 1.5 -> 3.0
        else:  # delay
            sigma = 1.5
            tc    = 1.5

        sigmas[i] = sigma
        tcs[i]    = tc

        # Lagged truth used by tc / delay scenes.
        if mode == "tc":
            lag = int(round(tc_lag_max * (tc / 3.0)))
            target_lagged = target[max(i - lag, 0)]
        elif mode == "delay":
            target_lagged = target[max(i - delay_lag, 0)]
        else:
            target_lagged = target[i]

        # Scene boundary -> reset transient state (drift, counter)
        if i > 0 and modes[i] != modes[i - 1]:
            est_state = {"last": estimate[i - 1].copy(),
                         "prev": estimate[max(i - 2, 0)].copy()}

        estimate[i] = estimate_position(
            target[i], target_lagged, mode, scene_phase,
            sigma=sigma, tc=tc, bias_dir=bias_dir, est_state=est_state,
        )
        pulses[i] = bool(est_state.get("pulse", False))

    # ── plotter ──────────────────────────────────────────────────────────────
    pv.global_theme.allow_empty_mesh = True
    plotter = pv.Plotter(off_screen=True, window_size=list(WINDOW))
    plotter.set_background(BG_COLOR)
    plotter.enable_anti_aliasing("ssaa")

    # Sphere-radius constants. Truth is the reference body, so it's largest
    # and the estimate is smaller — when they overlap (nominal scene), the
    # white truth pokes out around the amber estimate so both stay legible.
    R_TRUTH  = 0.14
    R_EST    = 0.09
    R_CHASER = 0.11

    # Pre-compute camera framing (look at full scene)
    all_pts = np.vstack([target, chaser, estimate])
    center = all_pts.mean(axis=0)
    extent = np.ptp(all_pts, axis=0)
    radius = float(np.linalg.norm(extent)) * 0.55 + 4.5

    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", quality=8,
        macro_block_size=1, ffmpeg_log_level="error",
    )

    print(f"Rendering {n_frames} frames at {fps} fps -> {out_path}")
    print(f"Scene budget (frames): {dict(zip([s[0] for s in SCENES], frames_per_scene))}")

    ray_rng = np.random.default_rng(7)

    EST_TRAIL_LEN = max(int(round(0.9 * fps)), 18)   # ~0.9s of recent history

    # Scene-boundary clipping for the estimate trail: the estimator model
    # changes at each scene change (offset -> lagged truth -> ghost), so a
    # trail that crosses a boundary draws a discontinuity. Clip every
    # frame's trail to the current scene's start.
    scene_starts = np.cumsum([0] + frames_per_scene[:-1])

    for i in range(n_frames):
        scene_idx, scene_phase = scene_for_frame(i, frames_per_scene)
        label, caption, _, mode = SCENES[scene_idx]

        plotter.clear()
        plotter.set_background(BG_COLOR)

        # ── Layer 1 · faint full reference paths (context) ────────────────
        # Truth: full path, thin and dim — context, not foreground.
        plotter.add_mesh(
            make_polyline(target),
            color=COL_TRUTH, line_width=2.0, opacity=0.28,
        )
        # Chaser: faint loop guide.
        plotter.add_mesh(
            make_polyline(chaser),
            color=COL_CHASER, line_width=1.0, opacity=0.18,
        )
        # Estimate: faint full history, never disappears. Boundary jumps
        # between scenes are visible but stay subtle at this opacity.
        if i >= 1:
            plotter.add_mesh(
                make_polyline(estimate[: i + 1]),
                color=COL_ESTIMATE, line_width=2.0, opacity=0.18,
            )

        # ── Layer 2 · already-traversed truth (slightly brighter) ─────────
        if i >= 1:
            plotter.add_mesh(
                make_polyline(target[: i + 1]),
                color=COL_TRUTH, line_width=2.0, opacity=0.55,
            )

        # ── Layer 3 · bright recent estimate arc (current behavior) ──────
        # Scene-clipped so the bright segment never bridges modes.
        i0 = max(int(scene_starts[scene_idx]), i - EST_TRAIL_LEN)
        if i - i0 >= 1:
            plotter.add_mesh(
                make_polyline(estimate[i0:i + 1]),
                color=COL_ESTIMATE, line_width=4.5, opacity=0.95,
            )

        # Bodies -----------------------------------------------------------
        plotter.add_mesh(
            pv.Sphere(radius=R_TRUTH, center=target[i],
                      theta_resolution=28, phi_resolution=28),
            color=COL_TRUTH, smooth_shading=True,
            ambient=0.5, diffuse=0.65,
        )
        plotter.add_mesh(
            pv.Sphere(radius=R_EST, center=estimate[i],
                      theta_resolution=28, phi_resolution=28),
            color=COL_ESTIMATE, smooth_shading=True,
            ambient=0.55, diffuse=0.65,
        )
        plotter.add_mesh(
            pv.Sphere(radius=R_CHASER, center=chaser[i],
                      theta_resolution=28, phi_resolution=28),
            color=COL_CHASER, smooth_shading=True,
            ambient=0.55, diffuse=0.65,
        )

        # 3D labels anchored to each body. In delay mode the estimate sits
        # on top of the ghost target, so we drop the ESTIMATE label there
        # (the orange sphere is self-explanatory) to avoid stacked text;
        # STALE TARGET gets its own label below where the ghost is drawn.
        if mode == "delay":
            plotter.add_point_labels(
                points=np.array([
                    target[i]   + np.array([0.40, 0.0,  0.0]),
                    chaser[i]   + np.array([0.0,  0.0, -0.32]),
                ]),
                labels=["TRUE TARGET", "CHASER"],
                text_color=COL_TEXT, font_size=10,
                shape=None, point_size=0, always_visible=True,
                font_file=UNICODE_FONT,
            )
        else:
            plotter.add_point_labels(
                points=np.array([
                    target[i]   + np.array([0.0,  0.0,  0.32]),
                    estimate[i] + np.array([0.32, 0.0,  0.18]),
                    chaser[i]   + np.array([0.0,  0.0, -0.32]),
                ]),
                labels=["TRUE TARGET", "ESTIMATE", "CHASER"],
                text_color=COL_TEXT, font_size=10,
                shape=None, point_size=0, always_visible=True,
                font_file=UNICODE_FONT,
            )

        # Live MISS readout on the error vector — only when separation is
        # visually meaningful. Hidden in delay mode to keep that scene's
        # labels focused on TRUE / STALE.
        miss_vec = target[i] - estimate[i]
        miss_dist = float(np.linalg.norm(miss_vec))
        if miss_dist > 1e-2 and mode != "delay":
            mid = 0.5 * (target[i] + estimate[i])
            plotter.add_point_labels(
                points=np.array([mid + np.array([0.20, 0.0, 0.0])]),
                labels=[f"MISS  {miss_dist:.3f}"],
                text_color=COL_ERROR, font_size=9,
                shape=None, point_size=0, always_visible=True,
                font_file=UNICODE_FONT,
            )

        # LOS ray ----------------------------------------------------------
        if mode == "delay":
            j = max(i - delay_lag, 0)
            los_endpoint = target[j]
            # Red ghost target — the stale measurement geometry.
            plotter.add_mesh(
                pv.Sphere(radius=R_TRUTH * 1.05, center=target[j],
                          theta_resolution=24, phi_resolution=24),
                color=COL_GHOST, opacity=0.45, smooth_shading=True,
            )
            plotter.add_mesh(
                line_between(chaser[i], los_endpoint),
                color=COL_GHOST, line_width=3.5, opacity=0.90,
            )
            # Label the ghost target so the audience reads the failure mode.
            plotter.add_point_labels(
                points=np.array([target[j] + np.array([-0.32, 0.0, 0.0])]),
                labels=["STALE TARGET"],
                text_color=COL_GHOST,
                font_size=9,
                shape=None,
                point_size=0,
                always_visible=True,
                font_file=UNICODE_FONT,
            )
        else:
            los_endpoint = target[i]
            plotter.add_mesh(
                line_between(chaser[i], los_endpoint),
                color=COL_LOS, line_width=2.8, opacity=0.85,
            )

        # Pixel-noise fan: 6 faint cyan rays around the LOS. Per-frame
        # spread is gain * σ; with gain=0.04 the fan is ~0.04 wide at σ=1
        # and ~0.20 at σ=5 — visible but never overwhelming.
        if mode == "sigma":
            sigma_now = sigmas[i]
            for _ in range(6):
                ep = jittered_endpoint(chaser[i], target[i],
                                       sigma_now, ray_rng, gain=0.04)
                plotter.add_mesh(
                    line_between(chaser[i], ep),
                    color=COL_LOS, line_width=1.2, opacity=0.22,
                )

        # Error vector -----------------------------------------------------
        plotter.add_mesh(
            line_between(estimate[i], target[i]),
            color=COL_ERROR, line_width=4.0, opacity=0.95,
        )

        # Correction-update halo (only in tc scene, on update frames) ------
        if mode == "tc" and pulses[i]:
            plotter.add_mesh(
                pv.Sphere(radius=R_EST * 2.0, center=estimate[i],
                          theta_resolution=24, phi_resolution=12),
                color=COL_PULSE, opacity=0.22, style="wireframe", line_width=1.2,
            )

        # Text overlays ----------------------------------------------------
        plotter.add_text(
            TITLE,
            position="upper_left", font_size=11, color=COL_TEXT, shadow=False,
            font_file=UNICODE_FONT,
        )
        plotter.add_text(
            label,
            position=(20, WINDOW[1] - 54), font_size=9,
            color=COL_DIM, shadow=False,
            font_file=UNICODE_FONT,
        )
        plotter.add_text(
            caption,
            position=(20, WINDOW[1] - 78), font_size=9,
            color=COL_DIM, shadow=False,
            font_file=UNICODE_FONT,
        )

        # HUD (bottom-left): live σ_pix / t_c — small, single line
        if mode == "delay":
            hud = f"σ_pix = {sigmas[i]:4.2f} px    t_c = {tcs[i]:4.2f}    1-step delay"
        else:
            hud = f"σ_pix = {sigmas[i]:4.2f} px    t_c = {tcs[i]:4.2f}"
        plotter.add_text(
            hud, position="lower_left", font_size=9,
            color=COL_TEXT, shadow=False,
            font_file=UNICODE_FONT,
        )

        # Disclaimer (bottom-right)
        plotter.add_text(
            "representative seed   ·   visualized error amplified",
            position=(WINDOW[0] - 360, 14), font_size=8,
            color=COL_DIM, shadow=False,
            font_file=UNICODE_FONT,
        )

        # Camera: slow continuous orbit around scene center ----------------
        global_phase = i / max(n_frames - 1, 1)
        cam_phase = smoothstep(global_phase)
        az = -0.55 * np.pi + 0.55 * np.pi * cam_phase
        el = 0.30 + 0.05 * np.sin(2 * np.pi * global_phase)
        cam_pos = center + radius * np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ])
        plotter.camera_position = [tuple(cam_pos), tuple(center), (0.0, 0.0, 1.0)]

        img = plotter.screenshot(return_img=True)
        writer.append_data(img)

        if (i + 1) % 30 == 0 or i == n_frames - 1:
            print(f"  frame {i + 1:4d} / {n_frames}  ({label})")

    writer.close()
    plotter.close()
    print(f"Saved {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fps",     type=int,   default=30)
    p.add_argument("--seconds", type=float, default=14.0)
    p.add_argument("--out",     type=str,   default=str(OUT_PATH))
    args = p.parse_args()
    render(fps=args.fps, seconds=args.seconds, out_path=Path(args.out))
