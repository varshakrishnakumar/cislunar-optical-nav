"""IEKF single-step convergence visualization.

A copy of the bearing-only navigation animation (anim_03_bearings_v2.mp4),
reduced to a single measurement instant. Shows the iterated EKF (IEKF) update:
prior estimate, predicted line-of-sight, measured line-of-sight, and the
sequence of iterates pulling the LOS into alignment.

Removes from the original:
  * time-evolving trajectory animation
  * position-error panel
  * NIS panel
  * t = ... time labels
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import ensure_src_on_path
ensure_src_on_path()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from dynamics.cr3bp import CR3BP
from dynamics.integrators import propagate
from nav.measurements.bearing import (
    bearing_measurement_model,
    los_unit,
    tangent_basis,
)
from diagnostics.health import chol_solve_spd, symmetrize

# Reuse colors / helpers from the existing animation module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from animate_phases_2_3 import (  # noqa: E402
    _AMBER, _BG, _BORDER, _CYAN, _DIM, _GREEN, _MOON_C, _PANEL, _RED,
    _TEXT, _VIOLET, _WHITE, L_KM,
    _draw_sphere, _draw_textured_sphere, _try_save,
)

OUT_DIR = Path("results/demos")


# ── IEKF iteration capture ─────────────────────────────────────────────────────
def run_iekf_iterates(x_prior, P, u_meas, r_body, sigma_theta, max_iter=6,
                      step_tol=1e-15):
    """Replay the IEKF update loop and return every iterate x_0 ... x_N.

    Mirrors `bearing_update_tangent_iekf` but exposes the per-iteration state
    so we can animate the convergence path.
    """
    iters = [np.asarray(x_prior, dtype=float).copy()]
    residuals_rad = []
    x_iter = iters[0].copy()
    P = symmetrize(P)

    # Initial residual at the prior linearization point
    m0 = bearing_measurement_model(x=x_iter, u_meas=u_meas, r_body=r_body,
                                   sigma_theta=sigma_theta)
    residuals_rad.append(float(np.linalg.norm(m0.residual_3d)))

    for _ in range(max_iter):
        m = bearing_measurement_model(x=x_iter, u_meas=u_meas, r_body=r_body,
                                      sigma_theta=sigma_theta)
        S = symmetrize(m.H @ P @ m.H.T + m.R)
        PHt = P @ m.H.T
        K = chol_solve_spd(S, PHt.T).T
        correction_residual = m.residual_2d + m.H @ (x_iter - iters[0])
        x_next = iters[0] + K @ correction_residual

        step_norm = float(np.linalg.norm(x_next - x_iter))
        x_iter = np.asarray(x_next, dtype=float).reshape(6)
        iters.append(x_iter.copy())

        # Residual after this iterate's linearization point updates
        m_after = bearing_measurement_model(x=x_iter, u_meas=u_meas, r_body=r_body,
                                            sigma_theta=sigma_theta)
        residuals_rad.append(float(np.linalg.norm(m_after.residual_3d)))

        if step_norm <= step_tol:
            break

    return iters, residuals_rad


# ── scene setup ────────────────────────────────────────────────────────────────
def build_scene():
    """Build a deliberately mismatched scenario so multiple IEKF iterations
    are visibly distinct, not converged in a single step.
    """
    mu = 0.0121505856
    model = CR3BP(mu=mu)
    L = model.lagrange_points()
    x0_nom = np.array([L["L1"][0] - 1e-3, 0.0, 0.0, 0.0, 0.05, 0.0], dtype=float)

    # Truth: nominal IC plus a small perturbation, propagated to a chosen instant.
    t_eval = 1.5  # CR3BP TU; mid-way through the standard run
    x_true0 = x0_nom.copy()
    x_true0[:3] += np.array([2e-4, -1e-4, 0.0])
    x_true0[3:] += np.array([0.0,  2e-3, 0.0])
    res = propagate(model.eom, (0.0, t_eval), x_true0, t_eval=[t_eval],
                    rtol=1e-11, atol=1e-13, method="DOP853")
    x_true = res.x[-1]

    # Prior estimate: deliberately offset so the predicted LOS is ~40° off the
    # measured LOS — that puts the IEKF firmly in the nonlinear regime where
    # the linearization is meaningfully wrong, so multiple iterations are
    # visibly distinct (residuals ≈ 720 → 144 → 61 → 13 → 0.8 mrad).
    r_body = np.asarray(model.primary2, dtype=float).reshape(3)
    u_true, rng_true = los_unit(r_body, x_true[:3])
    e1, e2 = tangent_basis(u_true)
    perp_mag = float(np.tan(np.radians(40.0)) * rng_true)   # ≈ 30 700 km
    perp_dx  = (0.92 * perp_mag) * e1 + (0.40 * perp_mag) * e2
    along_dx = (-0.04 * rng_true) * u_true                   # mild along-LOS bias
    x_prior = x_true.copy()
    x_prior[:3] += perp_dx + along_dx
    x_prior[3:] += np.array([1e-4, 1e-4, 0.0])

    # Prior covariance: diffuse enough that the IEKF gain is healthy. The
    # iteration count is driven by the angular nonlinearity, not by P.
    P = np.diag([2.5e-4, 2.5e-4, 2.5e-4, 1e-6, 1e-6, 1e-6]).astype(float)

    # Noiseless measurement for clean visuals — the user sees geometric
    # convergence, not measurement scatter.
    sigma_theta = 2e-4
    u_meas = u_true.copy()

    iters, residuals = run_iekf_iterates(x_prior, P, u_meas, r_body, sigma_theta,
                                         max_iter=6)

    return dict(
        r_body_km=r_body * L_KM,
        x_true_km=x_true[:3] * L_KM,
        u_meas=u_meas,
        iter_pos_km=np.array([it[:3] for it in iters]) * L_KM,
        residuals_rad=np.array(residuals),
        rng_true_km=float(rng_true * L_KM),
    )


# ── animation ──────────────────────────────────────────────────────────────────
def _smoothstep(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _ease(t):
    t = float(np.clip(t, 0.0, 1.0))
    return 0.5 - 0.5 * np.cos(np.pi * t)


def animate_iekf(fps: int = 30) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D proj

    print("Building IEKF iteration scenario …")
    s = build_scene()
    moon = s["r_body_km"]
    truth = s["x_true_km"]
    u_meas = s["u_meas"]
    iters = s["iter_pos_km"]
    residuals = s["residuals_rad"]
    rng_true = s["rng_true_km"]

    n_iter = len(iters) - 1  # iters[0] is the prior
    print(f"  IEKF iterates: {len(iters)} (prior + {n_iter} updates)")
    print(f"  residual sequence (μrad): "
          + ", ".join(f"{r * 1e6:.1f}" for r in residuals))

    # ── frame budget ───────────────────────────────────────────────────
    F_INTRO  = 24
    F_PRIOR  = 24
    F_MEAS   = 24
    F_HOLD   = 18
    F_PER_IT = 38
    F_FINAL  = 90       # longer hold at convergence (~3 s)
    F_BEFORE_ITER = F_INTRO + F_PRIOR + F_MEAS + F_HOLD
    F_TOTAL = F_BEFORE_ITER + n_iter * F_PER_IT + F_FINAL
    print(f"  total frames: {F_TOTAL}  (≈ {F_TOTAL / fps:.1f} s @ {fps} fps)")

    # ── figure ─────────────────────────────────────────────────────────
    # Slide-shaped canvas: 6.99″ × 6.18″ × 200 dpi → 1398 × 1236 px.
    fig = plt.figure(figsize=(6.99, 6.18), facecolor=_BG, dpi=200)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d")
    ax.set_facecolor(_BG)

    # Strip 3D axis chrome — geometry only.
    pane_rgba = (0.020, 0.031, 0.063, 1.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(pane_rgba)
        axis._axinfo["grid"]["color"] = (0.10, 0.13, 0.25, 0.0)
        axis._axinfo["grid"]["linewidth"] = 0.0
        axis.line.set_color((0, 0, 0, 0))
        axis.set_ticklabels([])
        for tick in axis.get_major_ticks():
            tick.tick1line.set_visible(False); tick.tick2line.set_visible(False)
            tick.label1.set_visible(False);    tick.label2.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    try:
        ax.set_axis_off()
    except Exception:
        pass

    # Moon — demoted to context. ~40% smaller, low alpha.
    moon_radius = 5_500.0
    moon_tex = Path("results/seeds/moon_texture.jpg")
    if moon_tex.exists():
        _draw_textured_sphere(ax, moon, moon_radius, moon_tex,
                              n=64, alpha=0.55, rotate_lon_deg=180.0)
    else:
        _draw_sphere(ax, moon, moon_radius, _MOON_C, alpha=0.45)

    # Truth marker — small, optional.
    truth_dot = ax.scatter([truth[0]], [truth[1]], [truth[2]],
                           s=22, color=_CYAN, alpha=0.0, zorder=7,
                           edgecolors=_WHITE, linewidths=0.4, depthshade=False)

    # ── MEASURED LOS — promoted to "the constraint" ────────────────────
    # Two glow layers + a thick bright cyan line.
    los_len = 1.4 * rng_true
    meas_a = moon - (-u_meas) * los_len * 0.18
    meas_b = moon + (-u_meas) * los_len
    meas_glow,  = ax.plot([meas_a[0], meas_b[0]], [meas_a[1], meas_b[1]],
                          [meas_a[2], meas_b[2]],
                          color=_CYAN, lw=22.0, alpha=0.0, zorder=4)
    meas_glow2, = ax.plot([meas_a[0], meas_b[0]], [meas_a[1], meas_b[1]],
                          [meas_a[2], meas_b[2]],
                          color=_CYAN, lw=10.0, alpha=0.0, zorder=4)
    meas_line,  = ax.plot([meas_a[0], meas_b[0]], [meas_a[1], meas_b[1]],
                          [meas_a[2], meas_b[2]],
                          color=_CYAN, lw=4.2, alpha=0.0, zorder=5)

    # ── ITERATION SNAKE — connects past iterates with a continuous line ─
    snake_glow, = ax.plot([], [], [], color=_VIOLET, lw=5.0, alpha=0.0, zorder=5)
    snake,      = ax.plot([], [], [], color=_VIOLET, lw=1.6, alpha=0.0, zorder=6)
    trail_dots, = ax.plot([], [], [], "o", color=_VIOLET, ms=4.0,
                          alpha=0.0, zorder=7)

    # Past predicted-LOS rays — very faint, dotted (so the rotation toward
    # the measured LOS reads as a fan, not a tangle).
    trail_lines = []
    for _ in range(len(iters)):
        ln, = ax.plot([], [], [], color=_AMBER, lw=0.5, alpha=0.0,
                      ls=":", zorder=4)
        trail_lines.append(ln)

    # ── CURRENT estimate + predicted LOS (subordinate to measured) ─────
    cur_dot = ax.scatter([], [], [], s=90, color=_AMBER, alpha=0.0, zorder=10,
                         edgecolors=_WHITE, linewidths=0.6, depthshade=False,
                         marker="s")
    pred_glow, = ax.plot([], [], [], color=_AMBER, lw=3.0, alpha=0.0, zorder=5)
    pred_line, = ax.plot([], [], [], color=_AMBER, lw=1.7, alpha=0.0, zorder=6,
                         ls="--")

    # In-scene label tied to the orange square — shown only during the prior
    # / measurement / hold phases so the audience learns what the square is,
    # then disappears once the iteration story takes over.
    cur_label = ax.text(iters[0][0], iters[0][1], iters[0][2],
                        "  $\\hat{x}_k$  current estimate",
                        color=_AMBER, fontsize=8.2, family="monospace",
                        fontweight="bold", alpha=0.0, zorder=11)

    # Residual arc — slerps between the two LOS directions on a sphere
    # well outside the Moon silhouette so it doesn't get hidden in the LOS
    # bundle. Glow + main line so it reads at small angles too.
    res_arc_glow, = ax.plot([], [], [], color="#FFD24A", lw=6.5, alpha=0.0, zorder=7)
    res_arc,      = ax.plot([], [], [], color="#FFD24A", lw=2.4, alpha=0.0, zorder=8)

    # Convergence flash on the (now-aligned) LOS.
    flash, = ax.plot([], [], [], color=_GREEN, lw=10, alpha=0.0, zorder=4)

    # ── camera framing — tight crop on the iterate/LOS action ─────────
    # Small bias toward the iterate side of the bounding box keeps the Moon
    # nudged toward the lower-right rather than dead-center; a tight `pad`
    # keeps empty black to a minimum so the geometry fills the frame.
    bias_anchor = iters[0] + (iters[0] - moon) * 0.15
    all_pts = np.vstack([
        iters, moon[None, :], meas_b[None, :], bias_anchor[None, :],
    ])
    pad = 0.018 * (all_pts.max(axis=0) - all_pts.min(axis=0) + 1.0)
    lo = all_pts.min(axis=0) - pad
    hi = all_pts.max(axis=0) + pad
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    try:
        ranges = hi - lo
        ax.set_box_aspect(tuple(ranges / ranges.max()))
    except Exception:
        pass
    ax.view_init(elev=22.0, azim=-62.0)

    # ── tight legend (3 entries — orange square is labeled in-scene) ──
    fig.text(0.030, 0.955, "Bearing-Only IEKF Update", color=_TEXT,
             fontsize=10.5, fontweight="bold", family="monospace")
    fig.text(0.030, 0.930, "single measurement  ·  iterated linearization",
             color=_DIM, fontsize=7.4, family="monospace")
    fig.text(0.030, 0.892, "──  measured LOS  (fixed)",  color=_CYAN,
             fontsize=7.8, family="monospace", fontweight="bold")
    fig.text(0.030, 0.870, "┈  predicted LOS  (chases)", color=_AMBER,
             fontsize=7.8, family="monospace", fontweight="bold")
    fig.text(0.030, 0.848, "●  iteration path",          color=_VIOLET,
             fontsize=7.8, family="monospace", fontweight="bold")

    # Status (bottom-left).
    status_iter = fig.text(0.030, 0.075, "", color=_TEXT,
                           fontsize=11, family="monospace", fontweight="bold")
    status_res  = fig.text(0.030, 0.045, "", color=_VIOLET,
                           fontsize=9.5, family="monospace")
    status_hint = fig.text(0.030, 0.020, "", color=_DIM,
                           fontsize=8.5, family="monospace")

    # ── helpers ────────────────────────────────────────────────────────
    def _set_pred(x_pos, alpha_dot=1.0, alpha_line=0.85, alpha_glow=0.25):
        cur_dot._offsets3d = ([x_pos[0]], [x_pos[1]], [x_pos[2]])
        cur_dot.set_alpha(alpha_dot)
        pred_line.set_data([x_pos[0], moon[0]], [x_pos[1], moon[1]])
        pred_line.set_3d_properties([x_pos[2], moon[2]])
        pred_line.set_alpha(alpha_line)
        pred_glow.set_data([x_pos[0], moon[0]], [x_pos[1], moon[1]])
        pred_glow.set_3d_properties([x_pos[2], moon[2]])
        pred_glow.set_alpha(alpha_glow)

    def _set_meas(alpha):
        # Cyan LOS = the constraint. Render it brightest so "orange chases
        # cyan" reads at the first glance.
        meas_line.set_alpha(min(1.0, 1.00 * alpha))
        meas_glow2.set_alpha(0.42 * alpha)
        meas_glow.set_alpha(0.24 * alpha)

    def _residual_at(x_pos):
        rho = moon - x_pos
        u_pred = rho / np.linalg.norm(rho)
        cos_psi = float(np.clip(np.dot(u_meas, u_pred), -1.0, 1.0))
        return float(np.arccos(cos_psi))

    def _set_residual_arc(x_pos, alpha=0.85, n=48):
        rho = moon - x_pos
        u_pred = rho / np.linalg.norm(rho)
        a, b = -u_pred, -u_meas               # both point Moon → spacecraft side
        cosO = float(np.clip(np.dot(a, b), -1.0, 1.0))
        omega = float(np.arccos(cosO))
        if omega < 5e-4:                      # arc collapsed (≈ aligned)
            for art in (res_arc, res_arc_glow):
                art.set_data([], []); art.set_3d_properties([])
                art.set_alpha(0.0)
            return
        r_arc = moon_radius * 2.7             # well outside Moon silhouette
        sinO = float(np.sin(omega))
        t = np.linspace(0.0, 1.0, n)
        ca = np.sin((1.0 - t) * omega) / sinO
        cb = np.sin(t * omega) / sinO
        pts = (ca[:, None] * a[None, :] + cb[:, None] * b[None, :]) * r_arc \
              + moon[None, :]
        for art, a_alpha in ((res_arc, alpha), (res_arc_glow, 0.30 * alpha)):
            art.set_data(pts[:, 0], pts[:, 1])
            art.set_3d_properties(pts[:, 2])
            art.set_alpha(a_alpha)

    def _update_iter_path(visible_count, current_pos):
        """Draw the violet polyline through past iterates ending at current_pos.

        Purple path is supporting evidence, not the headline — old segments
        fade hard so the latest leg is the brightest part of the snake and
        the cyan/orange LOS pair stays the dominant story.
        """
        if visible_count <= 0:
            for art in (snake, snake_glow, trail_dots):
                art.set_data([], []); art.set_3d_properties([])
                art.set_alpha(0.0)
            for ln in trail_lines:
                ln.set_alpha(0.0)
            return

        seg_pts = np.vstack([iters[:visible_count], current_pos[None, :]])
        snake.set_data(seg_pts[:, 0], seg_pts[:, 1])
        snake.set_3d_properties(seg_pts[:, 2]); snake.set_alpha(0.55)
        snake_glow.set_data(seg_pts[:, 0], seg_pts[:, 1])
        snake_glow.set_3d_properties(seg_pts[:, 2]); snake_glow.set_alpha(0.10)

        past = iters[:visible_count]
        trail_dots.set_data(past[:, 0], past[:, 1])
        trail_dots.set_3d_properties(past[:, 2])
        trail_dots.set_alpha(0.35)

        for i, ln in enumerate(trail_lines):
            if i < visible_count:
                p = iters[i]
                ln.set_data([p[0], moon[0]], [p[1], moon[1]])
                ln.set_3d_properties([p[2], moon[2]])
                age = (visible_count - 1 - i) / max(1, visible_count - 1)
                ln.set_alpha(0.20 * (1.0 - 0.85 * age))
            else:
                ln.set_alpha(0.0)

    # ── frame update ───────────────────────────────────────────────────
    def init():
        return ()

    def _set_cur_label(pos, alpha):
        cur_label.set_position_3d((pos[0], pos[1], pos[2]))
        cur_label.set_alpha(alpha)

    def update(frame):
        # INTRO: fade in Moon.
        if frame < F_INTRO:
            cur_dot.set_alpha(0.0)
            pred_line.set_alpha(0.0); pred_glow.set_alpha(0.0)
            _set_meas(0.0); truth_dot.set_alpha(0.0)
            res_arc.set_alpha(0.0); res_arc_glow.set_alpha(0.0)
            _update_iter_path(0, iters[0])
            _set_cur_label(iters[0], 0.0)
            status_iter.set_text(""); status_res.set_text("")
            status_hint.set_text("")
            return ()

        # PRIOR: x⁰ + predicted LOS appear.
        f = frame - F_INTRO
        if f < F_PRIOR:
            t = _smoothstep(f / max(1, F_PRIOR - 1))
            _set_pred(iters[0],
                      alpha_dot=t, alpha_line=0.80 * t, alpha_glow=0.20 * t)
            _set_meas(0.0); truth_dot.set_alpha(0.0)
            res_arc.set_alpha(0.0); res_arc_glow.set_alpha(0.0)
            _set_cur_label(iters[0], t)
            psi = _residual_at(iters[0])
            status_iter.set_text("k = 0   prior estimate")
            status_res.set_text(f"residual ψ = {psi * 1e3:7.2f} mrad")
            status_hint.set_text("orange = predicted LOS from current estimate")
            return ()

        # MEAS: measured LOS + truth + residual arc fade in.
        f -= F_PRIOR
        if f < F_MEAS:
            t = _smoothstep(f / max(1, F_MEAS - 1))
            _set_pred(iters[0])
            _set_meas(t)
            truth_dot.set_alpha(0.40 * t)
            _set_residual_arc(iters[0], alpha=0.85 * t)
            _set_cur_label(iters[0], 1.0)
            psi = _residual_at(iters[0])
            status_iter.set_text("k = 0   prior estimate")
            status_res.set_text(f"residual ψ = {psi * 1e3:7.2f} mrad")
            status_hint.set_text("cyan = measured LOS  ·  prediction is wrong")
            return ()

        # HOLD: pause on the mismatch so the angle gap registers.
        f -= F_MEAS
        if f < F_HOLD:
            _set_pred(iters[0]); _set_meas(1.0)
            truth_dot.set_alpha(0.40)
            _set_residual_arc(iters[0], alpha=0.85)
            _set_cur_label(iters[0], 1.0)
            psi = _residual_at(iters[0])
            status_iter.set_text("k = 0   prior estimate")
            status_res.set_text(f"residual ψ = {psi * 1e3:7.2f} mrad")
            status_hint.set_text("single linearization is off  →  iterate")
            return ()

        # ITERATE: each iteration's transition gets F_PER_IT frames.
        f -= F_HOLD
        iter_idx = f // F_PER_IT
        sub      = f - iter_idx * F_PER_IT

        if iter_idx < n_iter:
            t = _ease(sub / max(1, F_PER_IT - 1))
            x_a = iters[iter_idx]
            x_b = iters[iter_idx + 1]
            x_now = (1.0 - t) * x_a + t * x_b
            _set_meas(1.0)
            truth_dot.set_alpha(0.35)
            _update_iter_path(iter_idx + 1, x_now)
            _set_pred(x_now,
                      alpha_dot=1.0, alpha_line=0.92, alpha_glow=0.22)
            _set_residual_arc(x_now, alpha=0.85)
            # Label fades out as the iteration story takes over.
            label_alpha = max(0.0, 1.0 - (iter_idx + t) / 1.5)
            _set_cur_label(x_now, label_alpha)
            psi = _residual_at(x_now)
            status_iter.set_text(f"k = {iter_idx + 1} / {n_iter}")
            status_res.set_text(f"residual ψ = {psi * 1e3:7.2f} mrad")
            status_hint.set_text(
                "re-linearize around new estimate  →  predicted LOS chases cyan")
            return ()

        # FINAL HOLD: aligned.
        f -= n_iter * F_PER_IT
        x_final = iters[-1]
        psi_final = _residual_at(x_final)
        _set_meas(1.0)
        truth_dot.set_alpha(0.35)
        _update_iter_path(n_iter, x_final)
        _set_pred(x_final, alpha_dot=1.0, alpha_line=0.95, alpha_glow=0.25)
        _set_residual_arc(x_final, alpha=0.0)
        _set_cur_label(x_final, 0.0)

        flash_t = _smoothstep(f / max(1, F_FINAL // 3))
        flash_alpha = 0.50 * (1.0 - flash_t) if f < F_FINAL // 3 else 0.0
        flash.set_data([x_final[0], moon[0]], [x_final[1], moon[1]])
        flash.set_3d_properties([x_final[2], moon[2]])
        flash.set_alpha(flash_alpha)

        status_iter.set_text(f"k = {n_iter} / {n_iter}   ✓ converged")
        status_res.set_text(f"residual ψ = {psi_final * 1e3:7.2f} mrad")
        status_hint.set_text("predicted LOS now aligns with measured LOS")
        return ()

    ani = FuncAnimation(fig, update, frames=F_TOTAL, init_func=init,
                        blit=False, interval=1000 // fps)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "anim_03_bearings_iekf_iter.mp4"
    _try_save(ani, out, fps)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    animate_iekf(fps=args.fps)
