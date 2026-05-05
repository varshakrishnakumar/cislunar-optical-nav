"""Fault-cascade cinematic for slide 13 (V&V · WHAT BREAKS IT).

A single bearing-only EKF arc runs five times back-to-back, each pass
activating one row of the slide-13 fault table:

    1. NOMINAL          — clean σ_pix=1, χ² gate strict
    2. 4% DROPOUT       — random measurements skipped (no update)
    3. PIXEL OUTLIERS   — 8σ injections rejected by the χ² gate
    4. LOOSE χ² GATE    — gate widened, outliers admitted
    5. 1-STEP DELAY     — measurements applied one step late (the breakage)

The audience sees the 3σ position ellipsoid contract on the good runs
and balloon on the broken one. A bottom timeline tracks ‖r̂ − r‖ across
all five passes — first four hug the baseline, the fifth rockets red.

Output sized for the slide-13 placeholder (6.32" × 4.76").

Render:
    python scripts/13b_fault_cascade_video.py [--fps 30] [--seconds 16]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from _common import ensure_src_on_path
ensure_src_on_path()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib import patheffects
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

from dynamics.cr3bp import CR3BP
from dynamics.integrators import propagate
from nav.ekf import ekf_propagate_cr3bp_stm
from nav.measurements.bearing import bearing_update_tangent, los_unit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from animate_phases_2_3 import _draw_textured_sphere, _draw_sphere  # noqa: E402

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
_MOON_C = "#9CA3AF"

# Earth–Moon CR3BP unit conversions
L_KM  = 384_400.0
T_DAY = 4.343

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR   = Path("results/demos")
FIG_W     = 6.32
FIG_H     = 4.76

# ── scenario descriptors ─────────────────────────────────────────────────────
# Each scenario keeps the same nominal arc; what changes is how the bearing
# stream gets corrupted and how the χ² gate responds. Tuned so the first four
# stay tight and only DELAY balloons — matches the slide-13 fault table.
SCENARIOS = [
    dict(key="NOMINAL",  label="NOMINAL",         color=_CYAN,
         dropout=0.00, outlier=0.00, kick=0.0, gate_p=0.9973, delay=False),
    dict(key="DROPOUT",  label="4% DROPOUT",      color=_AMBER,
         dropout=0.04, outlier=0.00, kick=0.0, gate_p=0.9973, delay=False),
    # 8σ kicks → NIS≈64, well past the strict-gate threshold (~12) → all rejected.
    dict(key="OUTLIERS", label="PIXEL OUTLIERS",  color=_VIOLET,
         dropout=0.00, outlier=0.10, kick=8.0, gate_p=0.9973, delay=False),
    # 4σ kicks → NIS≈16; strict gate would reject (>12), loose admits (<27.6)
    # → outliers leak through but the EKF rides them out.
    dict(key="LOOSE",    label="LOOSE χ² GATE",   color=_GREEN,
         dropout=0.00, outlier=0.10, kick=4.0, gate_p=0.999_999, delay=False),
    dict(key="DELAY",    label="1-STEP MEAS DELAY", color=_RED,
         dropout=0.00, outlier=0.00, kick=0.0, gate_p=0.9973, delay=True),
]


# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute: nominal truth + one EKF replay per scenario
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


def _ekf_run(*, mu: float, model, x_truth: np.ndarray, t_grid: np.ndarray,
             r_body_du: np.ndarray, x0_est: np.ndarray, P0: np.ndarray,
             sigma_theta: float, q_acc: float,
             scenario: dict, seed: int = 0) -> dict:
    """Replay a bearing-only EKF along the precomputed truth.

    Returns per-step traces of state estimate, covariance, the bearing-ray
    endpoints (predicted u from current estimate) and an event tag for
    'accepted' / 'rejected' / 'dropped'.
    """
    rng = np.random.default_rng(seed + hash(scenario["key"]) % 2**31)
    n = len(t_grid)

    x_est = np.zeros((n, 6))
    P_diag_pos = np.zeros((n, 3))   # 3σ extents along principal axes (DU)
    P_axes = np.zeros((n, 3, 3))    # eigenvectors of P[:3,:3] (column-wise)
    P_eig  = np.zeros((n, 3))       # eigenvalues (DU²)
    events = np.zeros(n, dtype="<U16")
    nis = np.zeros(n)

    x = x0_est.copy()
    P = P0.copy()

    # For DELAY: queue holds last step's measurement
    queued_meas: tuple[np.ndarray, np.ndarray] | None = None

    x_est[0] = x
    eig, V = np.linalg.eigh(P[:3, :3])
    eig = np.maximum(eig, 0.0)
    P_axes[0] = V
    P_eig[0]  = eig
    P_diag_pos[0] = 3.0 * np.sqrt(eig)
    events[0] = "init"

    for k in range(1, n):
        # Propagate estimate forward one step
        x, P, _ = ekf_propagate_cr3bp_stm(
            mu=mu, x=x, P=P,
            t0=float(t_grid[k - 1]), t1=float(t_grid[k]),
            q_acc=q_acc,
        )

        # Form the *truthful* bearing measurement at this step
        u_true, _ = los_unit(r_body_du, x_truth[k, :3])
        u_meas = u_true + sigma_theta * rng.normal(size=3)
        u_meas = u_meas / np.linalg.norm(u_meas)

        # Apply scenario corruption
        do_drop  = rng.random() < scenario["dropout"]
        do_outlier = rng.random() < scenario["outlier"]
        if do_outlier:
            # Rotate u_meas in a random tangent direction → controlled NIS spike
            kick = float(scenario["kick"]) * sigma_theta
            tangent = rng.normal(size=3)
            tangent = tangent - tangent.dot(u_meas) * u_meas
            tangent = tangent / max(np.linalg.norm(tangent), 1e-12)
            u_meas = u_meas + kick * tangent
            u_meas = u_meas / np.linalg.norm(u_meas)

        # Apply the measurement (or its delayed twin)
        meas_to_use: tuple[np.ndarray, np.ndarray] | None
        if scenario["delay"]:
            meas_to_use = queued_meas
            queued_meas = (u_meas.copy(), r_body_du.copy())
        else:
            meas_to_use = (u_meas.copy(), r_body_du.copy())

        if do_drop or meas_to_use is None:
            tag = "dropped"
            this_nis = float("nan")
        else:
            res = bearing_update_tangent(
                x=x, P=P,
                u_meas=meas_to_use[0],
                r_body=meas_to_use[1],
                sigma_theta=sigma_theta,
                gating_enabled=True,
                gate_probability=float(scenario["gate_p"]),
            )
            this_nis = float(res.nis)
            if res.accepted:
                x = res.x_upd
                P = res.P_upd
                tag = "outlier_accept" if do_outlier else "accepted"
            else:
                tag = "rejected"

        x_est[k] = x
        eig, V = np.linalg.eigh(0.5 * (P[:3, :3] + P[:3, :3].T))
        eig = np.maximum(eig, 0.0)
        P_axes[k] = V
        P_eig[k]  = eig
        P_diag_pos[k] = 3.0 * np.sqrt(eig)
        events[k] = tag
        nis[k] = this_nis

    return dict(
        x_est=x_est,
        P_axes=P_axes,
        P_eig=P_eig,
        P_3sigma=P_diag_pos,
        events=events,
        nis=nis,
    )


def _build_runs(*, n_steps: int, tf_tu: float):
    mu, model, x0 = _setup_orbit()
    t = np.linspace(0.0, float(tf_tu), n_steps)

    x_truth = _propagate(model, x0, t)              # (N, 6) DU
    L_pts = model.lagrange_points()

    # Bearing target = the second primary (Moon at +mu, 0, 0 in CR3BP-rot frame)
    p2 = np.asarray(model.primary2, dtype=float).reshape(-1)
    if p2.size < 3:
        p2 = np.array([float(p2[0]), 0.0, 0.0])
    r_body_du = p2[:3]

    # Initial estimate: small offset from truth + a few-thousand-km Gaussian P0.
    # Tuned so the ellipsoid is visibly wide at scenario start and contracts
    # noticeably under nominal updates.
    rng0 = np.random.default_rng(11)
    sigma_r0 = np.array([3.5e-3, 3.5e-3, 1.5e-3])   # DU  (~1300 km × 1300 km × 580 km)
    sigma_v0 = np.array([4.0e-4, 4.0e-4, 1.5e-4])   # DU/TU
    x0_est = x_truth[0] + np.concatenate([
        rng0.normal(0, sigma_r0), rng0.normal(0, sigma_v0)
    ])
    P0 = np.diag(np.concatenate([sigma_r0**2, sigma_v0**2]))

    # σ_θ ≈ 1 px on a ~5° / 1024 px detector → ~85 µrad
    sigma_theta = 1.0e-4
    q_acc = 1e-9

    runs = {}
    for sc in SCENARIOS:
        runs[sc["key"]] = _ekf_run(
            mu=mu, model=model, x_truth=x_truth, t_grid=t,
            r_body_du=r_body_du, x0_est=x0_est, P0=P0,
            sigma_theta=sigma_theta, q_acc=q_acc, scenario=sc, seed=21,
        )

    geom = dict(
        moon_km=r_body_du * L_KM,
        L1_km=float(L_pts["L1"][0]) * L_KM,
    )

    return dict(
        t_days=t * T_DAY,
        x_truth_km=x_truth[:, :3] * L_KM,
        runs=runs,
        geom=geom,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _smoothstep(x: float, a: float = 0.0, b: float = 1.0) -> float:
    t = float(np.clip((x - a) / (b - a), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _stars_3d(ax, n=180, seed=11, alpha=0.18):
    rng = np.random.default_rng(seed)
    xl, yl, zl = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    xs = rng.uniform(*xl, n)
    ys = rng.uniform(*yl, n)
    zs = rng.uniform(*zl, n)
    ax.scatter(xs, ys, zs, s=rng.uniform(0.10, 1.4, n),
               color="white", alpha=alpha, zorder=0,
               depthshade=False, rasterized=True)


def _ellipsoid_rings_3d(center, axes_eig_km, axes_evec, *, n=60, scale=3.0):
    """Return three principal-plane 3σ ring outlines for the position ellipsoid.

    Each ring lives on a plane spanned by a pair of eigenvectors of the
    position covariance — the three together trace the 3-D ellipsoid
    surface as readable wireframe rings. Cheap to redraw every frame.
    """
    order = np.argsort(axes_eig_km**2)[::-1]   # largest → smallest
    e = [axes_evec[:, k] for k in order]
    a = [scale * axes_eig_km[k] for k in order]
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    rings = []
    for j, k in ((0, 1), (0, 2), (1, 2)):
        pts = (np.outer(a[j] * np.cos(theta), e[j]) +
               np.outer(a[k] * np.sin(theta), e[k]))
        pts = pts + np.asarray(center)[None, :]
        rings.append((pts[:, 0], pts[:, 1], pts[:, 2]))
    return rings


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
def animate_cascade(*, fps: int = 30, seconds: float = 16.0,
                    n_steps: int = 90) -> None:
    n_scn = len(SCENARIOS)
    frames_per_scn = int(round(fps * seconds / n_scn))
    n_frames = frames_per_scn * n_scn

    print(f"Pre-computing {n_scn} EKF replays × {n_steps} steps …")
    data = _build_runs(n_steps=n_steps, tf_tu=2.0)

    t_d   = data["t_days"]
    Xtru  = data["x_truth_km"]                  # (N, 3)
    runs  = data["runs"]
    geom  = data["geom"]

    # Per-scenario miss timeseries (km), pre-computed for the bottom timeline.
    miss_by_key = {}
    for sc in SCENARIOS:
        x_est_km = runs[sc["key"]]["x_est"][:, :3] * L_KM
        miss_by_key[sc["key"]] = np.linalg.norm(x_est_km - Xtru, axis=1)
    miss_max = max(float(m.max()) for m in miss_by_key.values())
    miss_max = max(miss_max, 1.0)

    # ── figure & axes ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=_BG)

    # 3D scene fills the left ~62% of the figure.
    ax = fig.add_axes([-0.075, 0.140, 0.770, 0.730], projection="3d")
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

    # Bounds: enclose truth + worst-case ellipsoid extent across runs
    all_pts = [Xtru, geom["moon_km"][None, :]]
    for sc in SCENARIOS:
        x_est_km = runs[sc["key"]]["x_est"][:, :3] * L_KM
        # Add a halo of the worst 3σ extent so ellipses fit
        sigma_km = runs[sc["key"]]["P_3sigma"] * L_KM
        halo = np.linalg.norm(sigma_km, axis=1)[:, None]
        all_pts.append(x_est_km + halo * np.array([[1, 1, 1]]))
        all_pts.append(x_est_km - halo * np.array([[1, 1, 1]]))
    pts = np.vstack(all_pts)
    lo = pts.min(axis=0); hi = pts.max(axis=0)
    pad = (hi - lo) * 0.06 + 1_500
    lo -= pad; hi += pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    try:
        rng = hi - lo
        ax.set_box_aspect(tuple(rng / rng.max()))
    except Exception:
        pass

    _stars_3d(ax, n=160, alpha=0.15)

    # Moon — same texture path as the existing slide-13 video for consistency
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

    # L1 marker
    ax.scatter([geom["L1_km"]], [0], [0], marker="x", s=22,
               color=_DIM, alpha=0.70, zorder=4, depthshade=False)
    ax.text(geom["L1_km"], -6_000, 0, "L1",
            color=_DIM, fontsize=6.0, zorder=4)

    # Truth arc — faint dashed line (anchor)
    ax.plot(Xtru[:, 0], Xtru[:, 1], Xtru[:, 2],
            color=_GREEN, lw=0.8, alpha=0.30, ls="--", zorder=2)

    # Nominal endpoint star
    target = Xtru[-1]
    ax.scatter([target[0]], [target[1]], [target[2]],
               marker="*", s=140,
               color=_AMBER, edgecolors=_WHITE, linewidths=0.5,
               zorder=8, depthshade=False)
    ax.text(target[0] - 12_000, target[1] + 4_000, target[2] + 4_000,
            "nominal endpoint", color=_AMBER, fontsize=6.0, zorder=9,
            ha="right")

    # Live artists — estimated trail (3-pass glow + main)
    est_glow_outer, = ax.plot([], [], [], lw=5.5, alpha=0.06, zorder=3,
                              solid_capstyle="round", color=_CYAN)
    est_glow,       = ax.plot([], [], [], lw=3.0, alpha=0.16, zorder=4,
                              solid_capstyle="round", color=_CYAN)
    est_main,       = ax.plot([], [], [], lw=1.20, alpha=0.95, zorder=5,
                              solid_capstyle="round", color=_CYAN)
    head_dot = ax.scatter([0], [0], [0],
                          s=52, color=_CYAN, zorder=9,
                          edgecolors=_WHITE, linewidths=0.6,
                          depthshade=False)
    head_glow = ax.scatter([0], [0], [0],
                           s=180, color=_CYAN, zorder=8,
                           edgecolors="none", alpha=0.18,
                           depthshade=False)

    # 3σ position-uncertainty rings — 3 principal-plane outlines that
    # together trace the ellipsoid surface. The primary ring (largest plane)
    # gets a glow + bright main; the two secondary rings stay thin/dim so
    # the dominant uncertainty plane reads first.
    ell_primary_glow, = ax.plot([], [], [], lw=3.4, alpha=0.20, zorder=6,
                                color=_CYAN, solid_capstyle="round")
    ell_primary_main, = ax.plot([], [], [], lw=1.05, alpha=0.95, zorder=7,
                                color=_CYAN)
    ell_sec1, = ax.plot([], [], [], lw=0.55, alpha=0.40, zorder=6,
                        color=_CYAN)
    ell_sec2, = ax.plot([], [], [], lw=0.55, alpha=0.40, zorder=6,
                        color=_CYAN)

    # Bearing ray — spacecraft → Moon line (cyan accept / red reject / no-line drop)
    ray_glow, = ax.plot([], [], [], lw=2.4, alpha=0.10, zorder=5)
    ray_main, = ax.plot([], [], [], lw=0.7, alpha=0.85, zorder=6)

    # ── header ───────────────────────────────────────────────────────────────
    title_t = fig.text(0.020, 0.952, "WHAT  BREAKS  IT",
                       color=_TEXT, fontsize=10.5, fontweight="bold",
                       ha="left", va="center")
    fig.text(0.020, 0.922,
             "fault cascade · bearing-only EKF · σ_pix=1, t_c=2",
             color=_DIM, fontsize=6.5, ha="left", va="center")
    phase_text = fig.text(0.020, 0.898, "",
                          color=_CYAN, fontsize=8.0, ha="left", va="center",
                          fontweight="bold")
    for t in (title_t, phase_text):
        t.set_path_effects([
            patheffects.Stroke(linewidth=2.0, foreground=_BG),
            patheffects.Normal(),
        ])

    # ── NIS gauge ────────────────────────────────────────────────────────────
    # Live filter-health bar in the header — shows current step's NIS on a
    # log scale, with a vertical tick at the χ² gate threshold (2-DOF).
    # The bar fills cyan below threshold and flips red on overshoot, so
    # rejections register viscerally even when the bearing-ray flash is brief.
    gauge_x0, gauge_y0 = 0.385, 0.937
    gauge_w,  gauge_h  = 0.300, 0.018
    fig.patches.append(FancyBboxPatch(
        (gauge_x0 - 0.004, gauge_y0 - 0.005),
        gauge_w + 0.008, gauge_h + 0.010,
        boxstyle="round,pad=0.001,rounding_size=0.004",
        transform=fig.transFigure,
        linewidth=0.7, edgecolor=_BORDER, facecolor=_PANEL,
    ))
    nis_log_max = 2.5    # log10(NIS) display ceiling — ~316
    nis_gate_strict = float(np.log10(11.83))           # χ²(2, 0.9973)
    nis_gate_loose  = float(np.log10(27.63))           # χ²(2, 1-1e-6)
    gate_strict_x = gauge_x0 + gauge_w * (nis_gate_strict / nis_log_max)
    gate_loose_x  = gauge_x0 + gauge_w * (nis_gate_loose  / nis_log_max)
    gate_strict_tick = Rectangle(
        (gate_strict_x - 0.0008, gauge_y0 - 0.003),
        0.0016, gauge_h + 0.006,
        transform=fig.transFigure,
        linewidth=0, facecolor=_AMBER, alpha=0.95,
    )
    gate_loose_tick = Rectangle(
        (gate_loose_x - 0.0008, gauge_y0 - 0.003),
        0.0016, gauge_h + 0.006,
        transform=fig.transFigure,
        linewidth=0, facecolor=_GREEN, alpha=0.0,    # only shown on LOOSE
    )
    fig.patches.append(gate_strict_tick)
    fig.patches.append(gate_loose_tick)
    nis_fill = Rectangle(
        (gauge_x0, gauge_y0), 0.0001, gauge_h,
        transform=fig.transFigure,
        linewidth=0, facecolor=_CYAN, alpha=0.92,
    )
    fig.patches.append(nis_fill)
    fig.text(gauge_x0, gauge_y0 + gauge_h + 0.006,
             "NIS  ·  χ² FILTER HEALTH",
             color=_DIM, fontsize=5.6, ha="left", va="bottom",
             fontweight="bold", alpha=0.85)
    nis_value_text = fig.text(gauge_x0 + gauge_w + 0.006,
                              gauge_y0 + gauge_h / 2,
                              "", color=_TEXT, fontsize=6.4,
                              ha="left", va="center", fontweight="bold")

    # ── red-flash overlay (rejection visceral cue) ───────────────────────────
    # A translucent red rectangle covering the 3D panel that flares for a few
    # frames after each "rejected" event, alpha decaying.
    flash_rect = Rectangle(
        (0.0, 0.140), 0.700, 0.730,
        transform=fig.transFigure,
        linewidth=0, facecolor=_RED, alpha=0.0, zorder=20,
    )
    fig.patches.append(flash_rect)

    # ── scenario "title card" (cinematic act-break) ─────────────────────────
    # At each scenario boundary, a giant banner sweeps across the figure for
    # ~0.7 s. Two artists: a colored backing bar + an oversized label, both
    # animated. After the intro window, both fall to alpha 0 and the small
    # phase chip (top-left) carries the state.
    card_bar = Rectangle(
        (-0.20, 0.45), 1.40, 0.14,
        transform=fig.transFigure,
        linewidth=0, facecolor=_CYAN, alpha=0.0, zorder=18,
    )
    fig.patches.append(card_bar)
    card_text = fig.text(0.50, 0.520, "",
                         color=_BG, fontsize=22.0,
                         fontweight="bold", ha="center", va="center",
                         alpha=0.0, zorder=19)
    card_sub  = fig.text(0.50, 0.475, "",
                         color=_BG, fontsize=8.0,
                         fontweight="bold", ha="center", va="center",
                         alpha=0.0, zorder=19)

    # ── "DIVERGED" stamp (DELAY punchline) ──────────────────────────────────
    diverged_text = fig.text(0.34, 0.50, "FILTER  DIVERGED",
                             color=_RED, fontsize=18.0, fontweight="bold",
                             ha="center", va="center", alpha=0.0, zorder=21,
                             rotation=-8)
    diverged_text.set_path_effects([
        patheffects.Stroke(linewidth=3.0, foreground=_BG),
        patheffects.Normal(),
    ])

    # ── scenario list (right) ───────────────────────────────────────────────
    list_x = 0.700
    list_w = 0.285
    list_top = 0.880
    list_h = 0.620
    row_h = list_h / n_scn

    # Outer panel
    fig.patches.append(FancyBboxPatch(
        (list_x - 0.010, list_top - list_h - 0.012),
        list_w + 0.020, list_h + 0.040,
        boxstyle="round,pad=0.001,rounding_size=0.006",
        transform=fig.transFigure,
        linewidth=0.7, edgecolor=_BORDER, facecolor=_PANEL,
    ))
    fig.text(list_x, list_top + 0.012, "FAULT  SCENARIOS",
             color=_TEXT, fontsize=6.6, fontweight="bold", alpha=0.85,
             ha="left", va="bottom")

    row_box: list[Rectangle] = []
    row_label: list[plt.Text] = []
    row_stat:  list[plt.Text] = []
    row_stamp: list[plt.Text] = []
    for i, sc in enumerate(SCENARIOS):
        y0 = list_top - (i + 1) * row_h + 0.005
        rb = Rectangle(
            (list_x, y0), list_w, row_h - 0.010,
            transform=fig.transFigure,
            linewidth=0.7, edgecolor=_BORDER, facecolor=_PANEL, alpha=0.65,
        )
        fig.patches.append(rb)
        row_box.append(rb)

        idx_t = fig.text(list_x + 0.010, y0 + (row_h - 0.010) / 2 + 0.012,
                         f"{i + 1}", color=_DIM, fontsize=6.0,
                         fontweight="bold", ha="left", va="center")
        idx_t.set_alpha(0.7)
        lbl = fig.text(list_x + 0.030, y0 + (row_h - 0.010) / 2 + 0.012,
                       sc["label"], color=_TEXT, fontsize=7.0,
                       fontweight="bold", ha="left", va="center")
        st  = fig.text(list_x + 0.030, y0 + (row_h - 0.010) / 2 - 0.010,
                       "—", color=_DIM, fontsize=5.8, ha="left", va="center")
        stamp = fig.text(list_x + list_w - 0.010,
                         y0 + (row_h - 0.010) / 2 + 0.001,
                         "", color=_DIM, fontsize=10.0, fontweight="bold",
                         ha="right", va="center")
        row_label.append(lbl)
        row_stat.append(st)
        row_stamp.append(stamp)

    # ── bottom miss timeline ────────────────────────────────────────────────
    ax_mis = fig.add_axes([0.06, 0.04, 0.92, 0.085])
    ax_mis.set_facecolor("#080B17")
    for sp in ax_mis.spines.values():
        sp.set_edgecolor(_BORDER); sp.set_linewidth(0.7)
    ax_mis.tick_params(colors=_DIM, labelsize=5.5,
                       length=1.6, width=0.4, pad=1)
    ax_mis.set_yscale("log")
    ax_mis.set_ylim(max(1.0, 0.6 * float(min(m[1] for m in miss_by_key.values()))),
                    1.4 * miss_max)
    ax_mis.set_xlim(0, n_frames)
    ax_mis.set_xticks([frames_per_scn * (i + 0.5) for i in range(n_scn)])
    ax_mis.set_xticklabels([sc["label"].split()[0][:8] for sc in SCENARIOS],
                           color=_DIM, fontsize=5.6)
    ax_mis.set_ylabel("‖r̂−r‖  [km]", color=_DIM, fontsize=5.8, labelpad=2)
    ax_mis.grid(True, color=_BORDER, lw=0.3, alpha=0.6)
    # Per-scenario divider lines
    for i in range(1, n_scn):
        ax_mis.axvline(i * frames_per_scn, color=_BORDER, lw=0.5, alpha=0.7)
    miss_lines = []
    for sc in SCENARIOS:
        ln, = ax_mis.plot([], [], lw=1.1, color=sc["color"], alpha=0.92)
        miss_lines.append(ln)
    miss_dot = ax_mis.scatter([0], [1], s=18, color=_CYAN,
                              edgecolors=_WHITE, linewidths=0.4, zorder=5)
    miss_dot.set_offsets(np.empty((0, 2)))

    # ── camera plan ─────────────────────────────────────────────────────────
    elev_base = 38.0
    azim_base = -62.0
    azim_span = 18.0

    # Pre-compute the frame-by-frame step index inside a scenario.
    # EKF has n_steps states (0..n_steps-1), animation has frames_per_scn.
    step_lut = np.minimum(
        (np.arange(frames_per_scn) * (n_steps - 1) // (frames_per_scn - 1)).astype(int),
        n_steps - 1,
    )
    # Held frames at the end of each scenario give the eye time to read the
    # ✓/✗ verdict before the next scenario starts.
    HOLD_FRAMES = max(int(round(frames_per_scn * 0.10)), 4)
    step_lut[-HOLD_FRAMES:] = n_steps - 1

    # Cache the original axis limits so the DELAY dolly can lerp toward a
    # tight crop and back if needed.
    base_xlim = ax.get_xlim()
    base_ylim = ax.get_ylim()
    base_zlim = ax.get_zlim()

    # Title-card timing inside each scenario (in frames).
    CARD_IN  = max(int(round(frames_per_scn * 0.04)), 2)   # fade-in
    CARD_HOLD = max(int(round(frames_per_scn * 0.10)), 4)  # full opacity
    CARD_OUT = max(int(round(frames_per_scn * 0.06)), 3)   # fade-out
    CARD_LEN = CARD_IN + CARD_HOLD + CARD_OUT

    # ── update ──────────────────────────────────────────────────────────────
    def init():
        for ln in miss_lines:
            ln.set_data([], [])
        miss_dot.set_offsets(np.empty((0, 2)))
        return ()

    def update(frame):
        scn_idx = min(frame // frames_per_scn, n_scn - 1)
        scn_frame = frame - scn_idx * frames_per_scn
        scn_phase = scn_frame / max(frames_per_scn - 1, 1)
        sc = SCENARIOS[scn_idx]
        run = runs[sc["key"]]

        step = int(step_lut[scn_frame])
        x_est_km = run["x_est"][step, :3] * L_KM
        evec = run["P_axes"][step]
        eig_km = np.sqrt(np.maximum(run["P_eig"][step], 0.0)) * L_KM

        # Estimated trail up to current step
        trail_km = run["x_est"][:step + 1, :3] * L_KM
        for ln in (est_glow_outer, est_glow, est_main):
            ln.set_data(trail_km[:, 0], trail_km[:, 1])
            ln.set_3d_properties(trail_km[:, 2])
            ln.set_color(sc["color"])

        head_dot._offsets3d = ([x_est_km[0]], [x_est_km[1]], [x_est_km[2]])
        head_dot.set_facecolor([sc["color"]])
        head_glow._offsets3d = ([x_est_km[0]], [x_est_km[1]], [x_est_km[2]])
        head_glow.set_facecolor([sc["color"]])

        # 3σ position ellipsoid (3 perpendicular wireframe rings)
        try:
            rings = _ellipsoid_rings_3d(x_est_km, eig_km, evec,
                                        n=64, scale=3.0)
            (rx, ry, rz)   = rings[0]
            (sx, sy, sz)   = rings[1]
            (tx, ty, tz)   = rings[2]
            ell_primary_glow.set_data(rx, ry)
            ell_primary_glow.set_3d_properties(rz)
            ell_primary_main.set_data(rx, ry)
            ell_primary_main.set_3d_properties(rz)
            ell_sec1.set_data(sx, sy); ell_sec1.set_3d_properties(sz)
            ell_sec2.set_data(tx, ty); ell_sec2.set_3d_properties(tz)
            for ln in (ell_primary_glow, ell_primary_main,
                       ell_sec1, ell_sec2):
                ln.set_color(sc["color"])
        except Exception:
            pass

        # Bearing ray for this step → colour by event tag
        ev = str(run["events"][step])
        if ev == "dropped":
            ray_glow.set_data([], []); ray_glow.set_3d_properties([])
            ray_main.set_data([], []); ray_main.set_3d_properties([])
        else:
            moon_km = geom["moon_km"]
            xs = [x_est_km[0], moon_km[0]]
            ys = [x_est_km[1], moon_km[1]]
            zs = [x_est_km[2], moon_km[2]]
            ray_glow.set_data(xs, ys); ray_glow.set_3d_properties(zs)
            ray_main.set_data(xs, ys); ray_main.set_3d_properties(zs)
            if ev == "rejected":
                ray_main.set_color(_RED); ray_glow.set_color(_RED)
                ray_main.set_alpha(1.0); ray_glow.set_alpha(0.45)
                ray_main.set_linewidth(1.4)
            elif ev == "outlier_accept":
                ray_main.set_color(_VIOLET); ray_glow.set_color(_VIOLET)
                ray_main.set_alpha(0.95); ray_glow.set_alpha(0.30)
                ray_main.set_linewidth(0.9)
            else:
                ray_main.set_color(_CYAN); ray_glow.set_color(_CYAN)
                ray_main.set_alpha(0.55); ray_glow.set_alpha(0.10)
                ray_main.set_linewidth(0.7)

        # Header chip
        phase_text.set_text(f"#{scn_idx + 1}  ·  {sc['label']}")
        phase_text.set_color(sc["color"])

        # Scenario list — highlight active row, fade past rows, show stamps
        for i, sc_i in enumerate(SCENARIOS):
            rb  = row_box[i]
            lbl = row_label[i]
            st  = row_stat[i]
            stamp = row_stamp[i]
            if i < scn_idx:
                # Past scenario — verdict
                key = sc_i["key"]
                final_miss = miss_by_key[key][-1]
                # Survival rule: a fault "broke" the filter if the terminal
                # miss is on the same scale as the *initial* miss (≈ no
                # convergence). 5% of the prior initial miss is plenty of
                # headroom for the survivable scenarios to keep their checks.
                survive_threshold = 0.05 * miss_by_key["NOMINAL"][0]
                ok = final_miss < survive_threshold
                rb.set_facecolor(_PANEL); rb.set_alpha(0.65)
                rb.set_edgecolor(_BORDER)
                lbl.set_color(_TEXT); lbl.set_alpha(0.65)
                st.set_text(f"final ‖r̂−r‖ = {final_miss:6.0f} km")
                st.set_color(_DIM); st.set_alpha(0.85)
                stamp.set_text("✓" if ok else "✗")
                stamp.set_color(_GREEN if ok else _RED)
            elif i == scn_idx:
                # Active row — pulse highlight
                rb.set_facecolor("#10162A"); rb.set_alpha(0.95)
                rb.set_edgecolor(sc_i["color"])
                lbl.set_color(sc_i["color"]); lbl.set_alpha(1.0)
                cur_miss = miss_by_key[sc_i["key"]][step]
                st.set_text(f"step {step:3d}/{n_steps - 1}   ·   "
                            f"miss = {cur_miss:6.0f} km")
                st.set_color(_TEXT); st.set_alpha(0.85)
                # Stamp appears in last 12% of the scenario, using the same
                # "did the filter converge?" rule as the past-row branch.
                if scn_phase > 0.88:
                    final_miss = miss_by_key[sc_i["key"]][-1]
                    survive_threshold = 0.05 * miss_by_key["NOMINAL"][0]
                    ok = final_miss < survive_threshold
                    stamp.set_text("✓" if ok else "✗")
                    stamp.set_color(_GREEN if ok else _RED)
                else:
                    stamp.set_text("")
            else:
                # Upcoming — dim
                rb.set_facecolor(_PANEL); rb.set_alpha(0.40)
                rb.set_edgecolor(_BORDER)
                lbl.set_color(_DIM); lbl.set_alpha(0.85)
                st.set_text("—"); st.set_color(_DIM); st.set_alpha(0.65)
                stamp.set_text("")

        # Miss timeline — draw past scenarios full + active one growing
        for i, sc_i in enumerate(SCENARIOS):
            x0 = i * frames_per_scn
            if i < scn_idx:
                xs = np.linspace(x0, x0 + frames_per_scn - 1, n_steps)
                miss_lines[i].set_data(xs, miss_by_key[sc_i["key"]])
            elif i == scn_idx:
                xs = np.linspace(x0, x0 + scn_frame, step + 1)
                miss_lines[i].set_data(xs, miss_by_key[sc_i["key"]][:step + 1])
            else:
                miss_lines[i].set_data([], [])
        miss_dot.set_offsets([[scn_idx * frames_per_scn + scn_frame,
                               miss_by_key[sc["key"]][step]]])
        miss_dot.set_facecolor([sc["color"]])

        # Title colour reflects scenario severity
        if sc["key"] == "DELAY" and scn_phase > 0.4:
            title_t.set_color(_RED)
        else:
            title_t.set_color(_TEXT)

        # ── NIS gauge (live filter health) ───────────────────────────────
        nis_step = float(run["nis"][step])
        if not np.isfinite(nis_step) or nis_step <= 0.0:
            nis_log = 0.0
        else:
            nis_log = float(np.clip(np.log10(max(nis_step, 0.1)),
                                    0.0, nis_log_max))
        nis_fill.set_width(gauge_w * (nis_log / nis_log_max))
        gate_thresh = nis_gate_loose if sc["key"] == "LOOSE" else nis_gate_strict
        if ev == "rejected":
            nis_fill.set_facecolor(_RED)
        elif nis_log > gate_thresh:
            nis_fill.set_facecolor(_VIOLET)         # admitted-but-large
        else:
            nis_fill.set_facecolor(sc["color"])
        # Loose-gate tick is only shown during the LOOSE scenario.
        gate_loose_tick.set_alpha(0.95 if sc["key"] == "LOOSE" else 0.0)
        if np.isfinite(nis_step):
            nis_value_text.set_text(f"NIS = {nis_step:5.2f}")
        else:
            nis_value_text.set_text("NIS =  —")

        # ── red rejection flash ──────────────────────────────────────────
        # Decay-to-zero is implicit: we only set alpha > 0 when ev == "rejected"
        # and otherwise smoothly drop it (a tiny bit each frame).
        if ev == "rejected":
            flash_rect.set_alpha(0.45)
        else:
            flash_rect.set_alpha(max(0.0, flash_rect.get_alpha() - 0.18))

        # ── scenario title-card animation ────────────────────────────────
        # Frames 0..CARD_LEN of each scenario: banner sweeps in/holds/out
        # and after that the small phase chip carries the state.
        if scn_frame < CARD_LEN:
            if scn_frame < CARD_IN:
                p = scn_frame / max(CARD_IN, 1)
                a = _smoothstep(p)
            elif scn_frame < CARD_IN + CARD_HOLD:
                a = 1.0
            else:
                p = (scn_frame - CARD_IN - CARD_HOLD) / max(CARD_OUT, 1)
                a = 1.0 - _smoothstep(p)
            card_bar.set_alpha(0.92 * a)
            card_bar.set_facecolor(sc["color"])
            card_text.set_alpha(a)
            card_text.set_text(f"#{scn_idx + 1}   {sc['label']}")
            card_sub.set_alpha(a * 0.85)
            card_sub.set_text({
                "NOMINAL":  "calibrated baseline",
                "DROPOUT":  "4 % of bearings missing",
                "OUTLIERS": "8σ pixel outliers · χ² gate strict",
                "LOOSE":    "4σ outliers · χ² gate widened",
                "DELAY":    "measurements applied 1 step late",
            }[sc["key"]])
        else:
            card_bar.set_alpha(0.0)
            card_text.set_alpha(0.0)
            card_sub.set_alpha(0.0)

        # ── DELAY zoom-in & "DIVERGED" stamp ─────────────────────────────
        if sc["key"] == "DELAY":
            zoom = _smoothstep(scn_phase, 0.20, 0.95)   # 0 → 1 across run
            tighten = 0.55 * zoom                        # 0 → 0.55
            cx, cy, cz = x_est_km
            def _shrink(lo, hi, c):
                lo = lo + (c - lo) * tighten
                hi = hi - (hi - c) * tighten
                return lo, hi
            xl0, xl1 = _shrink(base_xlim[0], base_xlim[1], cx)
            yl0, yl1 = _shrink(base_ylim[0], base_ylim[1], cy)
            zl0, zl1 = _shrink(base_zlim[0], base_zlim[1], cz)
            ax.set_xlim(xl0, xl1)
            ax.set_ylim(yl0, yl1)
            ax.set_zlim(zl0, zl1)

            cur_miss_km = miss_by_key["DELAY"][step]
            if cur_miss_km > 1000.0:
                a = float(np.clip((cur_miss_km - 1000.0) / 1500.0, 0.0, 1.0))
                # Position the stamp near the diverging spacecraft head, but
                # in figure coords so it floats above the 3D axes.
                # Pulse ±5% size to draw the eye.
                pulse = 1.0 + 0.05 * np.sin(2 * np.pi * scn_phase * 4)
                diverged_text.set_alpha(a * 0.95)
                diverged_text.set_fontsize(18.0 * pulse)
            else:
                diverged_text.set_alpha(0.0)
        else:
            ax.set_xlim(base_xlim)
            ax.set_ylim(base_ylim)
            ax.set_zlim(base_zlim)
            diverged_text.set_alpha(0.0)

        # Camera — single eased orbit across all scenarios
        global_phase = frame / max(n_frames - 1, 1)
        eased = _smoothstep(global_phase)
        ax.view_init(
            elev=elev_base + 2.0 * np.sin(2 * np.pi * global_phase),
            azim=azim_base + azim_span * eased,
        )

        return ()

    ani = FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        blit=False, interval=1000 // fps,
    )

    out = OUT_DIR / "anim_13b_fault_cascade.mp4"
    _try_save(ani, out, fps)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fps",     type=int,   default=30)
    p.add_argument("--seconds", type=float, default=16.0)
    p.add_argument("--steps",   type=int,   default=90)
    args = p.parse_args()
    animate_cascade(fps=args.fps, seconds=args.seconds, n_steps=args.steps)
