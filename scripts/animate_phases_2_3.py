from __future__ import annotations

import argparse
from pathlib import Path

from _common import ensure_src_on_path
ensure_src_on_path()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Circle
from scipy.stats import chi2

from dynamics.cr3bp import CR3BP
from dynamics.integrators import propagate
from dynamics.variational import cr3bp_eom_with_stm
from nav.ekf import ekf_propagate_cr3bp_stm
from nav.measurements.bearing import los_unit, tangent_basis, bearing_update_tangent

# ── colour palette ─────────────────────────────────────────────────────────────
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
_EARTH_C = "#3B82F6"
_MOON_C  = "#9CA3AF"

# ── physical-unit conversions (Earth–Moon CR3BP) ───────────────────────────────
#   1 DU  = 384 400 km     (Earth–Moon distance, L*)
#   1 TU  = 4.343 days     (T_moon / 2π)
#   1 DU/TU ≈ 1 025 m/s
L_KM  = 384_400.0
T_DAY = 4.343
V_MS  = L_KM / (T_DAY * 86_400) * 1_000      # m/s per DU/TU

OUT_DIR = Path("results/demos")
TRAIL   = 220                                  # animated trail length in frames


# ── style helpers ──────────────────────────────────────────────────────────────
def _dark(ax, *, title="", xlabel="", ylabel=""):
    ax.set_facecolor(_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(_BORDER)
        sp.set_linewidth(0.7)
    ax.tick_params(colors=_DIM, which="both", labelsize=8)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    ax.grid(True, color=_BORDER, lw=0.45, linestyle="-")
    if title:  ax.set_title(title, pad=6, fontsize=9.5, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel, labelpad=3, fontsize=8.5)
    if ylabel: ax.set_ylabel(ylabel, labelpad=3, fontsize=8.5)


def _stars(ax, n=320, seed=42):
    rng = np.random.default_rng(seed)
    xl, yl = ax.get_xlim(), ax.get_ylim()
    xs = rng.uniform(*xl, n)
    ys = rng.uniform(*yl, n)
    ax.scatter(xs, ys, s=rng.uniform(0.2, 2.0, n),
               color="white", alpha=0.20, zorder=0, rasterized=True)


def _glow(ax, x, y, color, lw=1.8, zorder=3, **kw):
    """Static three-pass glow line (for fully-computed arcs)."""
    ax.plot(x, y, color=color, lw=lw * 5, alpha=0.06, zorder=zorder, **kw)
    ax.plot(x, y, color=color, lw=lw * 2.5, alpha=0.14, zorder=zorder, **kw)
    ax.plot(x, y, color=color, lw=lw, alpha=1.00, zorder=zorder, **kw)


def _try_save(ani, path: Path, fps: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        w = FFMpegWriter(fps=fps, bitrate=2_400,
                         extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
        ani.save(str(path), writer=w)
        print(f"  saved → {path}")
    except Exception as exc:
        gif = path.with_suffix(".gif")
        print(f"  ffmpeg failed ({exc}), falling back → {gif}")
        ani.save(str(gif), writer=PillowWriter(fps=fps))
        print(f"  saved → {gif}")


# ── CR3BP helpers ──────────────────────────────────────────────────────────────
def _setup():
    mu = 0.0121505856
    model = CR3BP(mu=mu)
    L = model.lagrange_points()
    x0 = np.array([L["L1"][0] - 1e-3, 0.0, 0.0, 0.0, 0.05, 0.0], dtype=float)
    return mu, model, x0


def _pack(x, phi=None):
    phi = np.eye(6) if phi is None else phi
    return np.concatenate([x, phi.reshape(-1, order="F")])


def _unpack(z):
    return z[:6].copy(), z[6:].reshape(6, 6, order="F").copy()


def _pstm(mu, t0, tf, z0):
    return propagate(
        lambda t, z: cr3bp_eom_with_stm(t, z, mu),
        (t0, tf), z0, rtol=1e-11, atol=1e-13, method="DOP853",
    )


def _ang_noise(u, sig, rng):
    u = u / np.linalg.norm(u)
    e1, e2 = tangent_basis(u)
    d = rng.normal(0, sig, 2)
    up = u + d[0] * e1 + d[1] * e2
    return up / np.linalg.norm(up)


def _smoothstep(x: float, a: float, b: float) -> float:
    t = float(np.clip((x - a) / (b - a), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _draw_sphere(ax, center, radius, color, *, alpha=0.9, n=28, zorder=5):
    u = np.linspace(0.0, 2.0 * np.pi, n)
    v = np.linspace(0.0, np.pi, n)
    cu, su = np.cos(u), np.sin(u)
    cv, sv = np.cos(v), np.sin(v)
    x = center[0] + radius * np.outer(cu, sv)
    y = center[1] + radius * np.outer(su, sv)
    z = center[2] + radius * np.outer(np.ones_like(u), cv)
    ax.plot_surface(
        x, y, z, color=color, alpha=alpha,
        linewidth=0, antialiased=True, shade=True, zorder=zorder,
    )


def _draw_textured_sphere(ax, center, radius, texture_path, *,
                          n=80, alpha=1.0, zorder=5, rotate_lon_deg=0.0):
    """Map an equirectangular texture onto a 3D sphere via plot_surface.

    Texture sampling happens once at setup; matplotlib reuses the face-
    color grid every frame, so the per-frame cost is just geometry
    projection (same as a solid sphere). Use for a static Moon where
    the view rotates but the body itself doesn't move.
    """
    import matplotlib.image as mpimg

    try:
        img = mpimg.imread(str(texture_path))
    except Exception as exc:
        print(f"  texture load failed ({exc}); falling back to solid Moon")
        _draw_sphere(ax, center, radius, _MOON_C, alpha=alpha, n=max(n // 2, 28),
                     zorder=zorder)
        return

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.dtype.kind in ("u", "i"):
        img = img.astype(np.float32) / 255.0
    H, W = img.shape[:2]

    # Sphere vertex grid  (u=longitude, v=polar angle from +z)
    u = np.linspace(0.0, 2.0 * np.pi, n)
    v = np.linspace(0.0, np.pi, n)
    cu, su = np.cos(u), np.sin(u)
    cv, sv = np.cos(v), np.sin(v)
    x = center[0] + radius * np.outer(cu, sv)
    y = center[1] + radius * np.outer(su, sv)
    z = center[2] + radius * np.outer(np.ones_like(u), cv)

    # Face-center samples for (n-1)×(n-1) quad faces
    uu_f = 0.5 * (u[:-1] + u[1:])
    vv_f = 0.5 * (v[:-1] + v[1:])
    U, V = np.meshgrid(uu_f, vv_f, indexing="ij")

    # Longitude → image column (with optional rotation of the Moon
    # about its polar axis so the familiar near-side faces the camera).
    lon = (U + np.radians(rotate_lon_deg)) % (2.0 * np.pi)
    px = np.clip((lon / (2.0 * np.pi) * W).astype(int), 0, W - 1)
    py = np.clip((V   /         np.pi * H).astype(int), 0, H - 1)

    face_rgb   = img[py, px, :3]
    face_rgba  = np.empty(face_rgb.shape[:2] + (4,), dtype=float)
    face_rgba[..., :3] = face_rgb
    face_rgba[..., 3]  = float(alpha)

    ax.plot_surface(
        x, y, z,
        facecolors=face_rgba,
        linewidth=0, antialiased=False, shade=False,
        rstride=1, cstride=1, zorder=zorder,
    )


# ── Phase 2: midcourse-correction data ────────────────────────────────────────
def _data2_3d(mu, model, x0):
    """Same targeting solve as _data2 but returns full 3D state in km."""
    t0, tf, tc = 0.0, 6.0, 2.0
    dx0 = np.array([2e-4, -1e-4, 0., 0., 2e-3, 0.])
    eom = model.eom

    res_nom = propagate(eom, (t0, tf), x0, dense_output=True, rtol=1e-11, atol=1e-13)
    r_target = res_nom.sol(tf)[:3]
    x0e = x0 + dx0

    res_tc = _pstm(mu, t0, tc, _pack(x0e))
    x_tc, _ = _unpack(res_tc.x[-1])
    dv = np.zeros(3)
    for _ in range(15):
        xb = x_tc.copy(); xb[3:6] += dv
        res_tf = _pstm(mu, tc, tf, _pack(xb))
        x_tf, phi = _unpack(res_tf.x[-1])
        err = x_tf[:3] - r_target
        if np.linalg.norm(err) < 1e-10:
            break
        try:
            dv -= np.linalg.solve(phi[:3, 3:6], err)
        except np.linalg.LinAlgError:
            dv -= np.linalg.lstsq(phi[:3, 3:6], err, rcond=None)[0]

    N = 800
    t_plot = np.linspace(t0, tf, N)
    i_post = t_plot >= tc

    X_nom = res_nom.sol(t_plot).T

    res_unc = propagate(eom, (t0, tf), x0e, dense_output=True, rtol=1e-11, atol=1e-13)
    X_unc = res_unc.sol(t_plot).T

    res_pre  = propagate(eom, (t0, tc), x0e, dense_output=True, rtol=1e-11, atol=1e-13)
    xb2 = res_pre.sol(tc).reshape(6,); xb2[3:6] += dv
    res_post = propagate(eom, (tc, tf), xb2, dense_output=True, rtol=1e-11, atol=1e-13)
    X_cor = np.full((N, 6), np.nan)
    X_cor[~i_post] = res_pre.sol(t_plot[~i_post]).T
    X_cor[i_post]  = res_post.sol(t_plot[i_post]).T

    tc_idx = int(np.argmin(np.abs(t_plot - tc)))
    dv_ms  = float(np.linalg.norm(dv)) * V_MS

    miss_unc = np.linalg.norm(X_unc[:, :3] - r_target, axis=1) * L_KM
    miss_cor = np.linalg.norm(X_cor[:, :3] - r_target, axis=1) * L_KM

    p1 = np.asarray(model.primary1, dtype=float)
    p2 = np.asarray(model.primary2, dtype=float)
    if p1.size < 3: p1 = np.array([p1[0], p1[1], 0.0])
    if p2.size < 3: p2 = np.array([p2[0], p2[1], 0.0])

    return dict(
        t=t_plot * T_DAY, tc=tc * T_DAY, tc_idx=tc_idx, tc_raw=float(tc),
        t_raw=t_plot, dv_ms=dv_ms,
        r_target=r_target[:3] * L_KM,
        X_nom=X_nom[:, :3] * L_KM,
        X_unc=X_unc[:, :3] * L_KM,
        X_cor=X_cor[:, :3] * L_KM,
        miss_unc=miss_unc, miss_cor=miss_cor,
        p1=p1[:3] * L_KM,
        p2=p2[:3] * L_KM,
    )


def _data2(mu, model, x0):
    t0, tf, tc = 0.0, 6.0, 2.0
    dx0 = np.array([2e-4, -1e-4, 0., 0., 2e-3, 0.])
    eom = model.eom

    res_nom = propagate(eom, (t0, tf), x0, dense_output=True, rtol=1e-11, atol=1e-13)
    r_target = res_nom.sol(tf)[:3]
    x0e = x0 + dx0

    res_tc = _pstm(mu, t0, tc, _pack(x0e))
    x_tc, _ = _unpack(res_tc.x[-1])
    dv = np.zeros(3)
    for _ in range(15):
        xb = x_tc.copy(); xb[3:6] += dv
        res_tf = _pstm(mu, tc, tf, _pack(xb))
        x_tf, phi = _unpack(res_tf.x[-1])
        err = x_tf[:3] - r_target
        if np.linalg.norm(err) < 1e-10:
            break
        try:
            dv -= np.linalg.solve(phi[:3, 3:6], err)
        except np.linalg.LinAlgError:
            dv -= np.linalg.lstsq(phi[:3, 3:6], err, rcond=None)[0]

    N = 800
    t_plot = np.linspace(t0, tf, N)
    i_post = t_plot >= tc

    X_nom = res_nom.sol(t_plot).T

    res_unc = propagate(eom, (t0, tf), x0e, dense_output=True, rtol=1e-11, atol=1e-13)
    X_unc = res_unc.sol(t_plot).T

    res_pre  = propagate(eom, (t0, tc), x0e, dense_output=True, rtol=1e-11, atol=1e-13)
    xb2 = res_pre.sol(tc).reshape(6,); xb2[3:6] += dv
    res_post = propagate(eom, (tc, tf), xb2, dense_output=True, rtol=1e-11, atol=1e-13)
    X_cor = np.full((N, 6), np.nan)
    X_cor[~i_post] = res_pre.sol(t_plot[~i_post]).T
    X_cor[i_post]  = res_post.sol(t_plot[i_post]).T

    tc_idx    = int(np.argmin(np.abs(t_plot - tc)))
    miss_unc  = np.linalg.norm(X_unc[:, :3] - r_target, axis=1) * L_KM    # km
    miss_cor  = np.linalg.norm(X_cor[:, :3] - r_target, axis=1) * L_KM
    dv_ms     = float(np.linalg.norm(dv)) * V_MS                            # m/s

    return dict(
        t=t_plot * T_DAY, tc=tc * T_DAY, tc_idx=tc_idx, tc_raw=float(tc),
        t_raw=t_plot, dv_ms=dv_ms,
        r_target=r_target[:2] * L_KM,
        X_nom=X_nom[:, :2] * L_KM,
        X_unc=X_unc[:, :2] * L_KM,
        X_cor=X_cor[:, :2] * L_KM,
        miss_unc=miss_unc, miss_cor=miss_cor,
        p1=model.primary1[:2] * L_KM,
        p2=model.primary2[:2] * L_KM,
    )


# ── Phase 3: bearing-only EKF data ────────────────────────────────────────────
def _data3(mu, model, x0):
    """EKF data in 3D with smooth interpolation for animation.

    Runs the filter on the sparse measurement grid (dt=0.02 TU), then
    resamples truth + estimate to a dense time grid so the animation
    pans smoothly instead of stepping through discrete measurement ticks.
    """
    from scipy.interpolate import CubicSpline

    rng    = np.random.default_rng(0)
    r_body = np.asarray(model.primary2, dtype=float)
    t0, tf, dt = 0.0, 6.0, 0.02
    t_meas = np.arange(t0, tf + 1e-12, dt)

    x_true0 = x0.copy()
    x_true0[:3] += [2e-4, -1e-4, 0.]
    x_true0[3:]  += [0., 2e-3, 0.]

    res = propagate(model.eom, (t0, tf), x_true0, t_eval=t_meas,
                    rtol=1e-11, atol=1e-13, dense_output=True)
    X_true_meas = res.x

    sig = 2e-4
    U_meas = np.array([
        _ang_noise(los_unit(r_body, X_true_meas[k, :3])[0], sig, rng)
        for k in range(len(t_meas))
    ])

    x = x0.copy()
    P = np.diag([1e-6] * 3 + [1e-8] * 3).astype(float)
    N = len(t_meas)
    X_hat_meas = np.zeros((N, 6))
    nis    = np.full(N, np.nan)
    P_diag = np.zeros((N, 6))
    X_hat_meas[0] = x; P_diag[0] = np.diag(P)
    t_prev = t_meas[0]

    for k in range(1, N):
        tk = float(t_meas[k])
        x, P, _ = ekf_propagate_cr3bp_stm(mu=mu, x=x, P=P, t0=t_prev, t1=tk, q_acc=1e-12)
        upd = bearing_update_tangent(x, P, U_meas[k], r_body, sig)
        if upd.accepted:
            x, P = upd.x_upd, upd.P_upd
        nis[k]    = upd.nis
        X_hat_meas[k]  = x
        P_diag[k] = np.diag(P)
        t_prev    = tk

    # Dense resample for smooth animation (truth via dense_output, EKF via spline)
    N_dense = 900
    t_dense = np.linspace(t0, tf, N_dense)
    X_true_dense = res.sol(t_dense).T                     # (N_dense, 6)
    spl_hat = CubicSpline(t_meas, X_hat_meas, axis=0, bc_type="natural")
    X_hat_dense = spl_hat(t_dense)

    pos_err_dense = np.linalg.norm(
        X_hat_dense[:, :3] - X_true_dense[:, :3], axis=1) * L_KM
    spl_sig = CubicSpline(t_meas, np.sqrt(np.abs(P_diag[:, 0])),
                          axis=0, bc_type="natural")
    sig3_dense = 3.0 * spl_sig(t_dense) * L_KM

    # Map each dense tick to nearest measurement for NIS look-ups
    meas_idx_per_dense = np.searchsorted(t_meas, t_dense)
    meas_idx_per_dense = np.clip(meas_idx_per_dense, 0, N - 1)

    return dict(
        # dense arrays for the animated orbit + side panels
        t=t_dense * T_DAY,
        X_true=X_true_dense[:, :3] * L_KM,
        X_hat=X_hat_dense[:, :3] * L_KM,
        pos_err=pos_err_dense,
        sig3=sig3_dense,
        # measurement-time arrays for NIS scatter + LOS pulses
        t_meas=t_meas * T_DAY,
        nis=nis,
        meas_idx_per_dense=meas_idx_per_dense,
        # geometry
        r_body=r_body,
        p2=model.primary2[:3] * L_KM if model.primary2.size >= 3
           else np.array([*model.primary2[:2], 0.0]) * L_KM,
    )


# ── Phase 2 animation ──────────────────────────────────────────────────────────
def animate_phase2(fps: int = 30, sim_speed: float = 0.77) -> None:
    print("Phase 2: computing trajectories …")
    mu, model, x0 = _setup()
    d = _data2(mu, model, x0)
    N  = len(d["t"])
    dt_f = float(d["t"][1] - d["t"][0])
    speed = max(1, round(sim_speed * T_DAY / (fps * dt_f)))

    # ── figure layout ──
    fig = plt.figure(figsize=(17, 8), facecolor=_BG)
    gs  = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[3, 2], wspace=0.30,
        left=0.07, right=0.97, bottom=0.11, top=0.91,
    )
    ax_orb = fig.add_subplot(gs[0])
    ax_mis = fig.add_subplot(gs[1])

    # ── orbit axes setup ──
    px, py = float(d["p2"][0]), float(d["p2"][1])
    all_xy = np.vstack([
        d["X_nom"],
        d["X_unc"],
        d["X_cor"][np.isfinite(d["X_cor"][:, 0])],
        np.array([[px, py]]),
        np.array([[d["r_target"][0], d["r_target"][1]]]),
    ])
    dxmin, dxmax = float(np.nanmin(all_xy[:, 0])), float(np.nanmax(all_xy[:, 0]))
    dymin, dymax = float(np.nanmin(all_xy[:, 1])), float(np.nanmax(all_xy[:, 1]))
    mx = (dxmax - dxmin) * 0.08 + 3_000
    my = (dymax - dymin) * 0.12 + 3_000
    xmin, xmax = dxmin - mx, dxmax + mx
    ymin, ymax = dymin - my, dymax + my
    ax_orb.set_xlim(xmin, xmax)
    ax_orb.set_ylim(ymin, ymax)
    _dark(ax_orb,
          title="Earth–Moon CR3BP — Midcourse Correction",
          xlabel="x  [km from barycenter]",
          ylabel="y  [km from barycenter]")
    _stars(ax_orb)

    # Moon circle (radius exaggerated for visibility)
    moon_r = (xmax - xmin) * 0.014
    ax_orb.add_patch(Circle((px, py), moon_r, color=_MOON_C, zorder=5, alpha=0.90))
    ax_orb.text(px, py + moon_r * 1.6, "Moon", color=_MOON_C,
                ha="center", fontsize=8, zorder=6)

    # Earth annotation (off-screen to the left)
    ax_orb.annotate(
        f"← Earth  ({float(d['p1'][0]):,.0f} km)",
        xy=(xmin + 500, ymin + (ymax - ymin) * 0.06),
        color=_EARTH_C, fontsize=8, ha="left", annotation_clip=False,
    )

    # L1 marker
    L1_km = float(x0[0]) * L_KM
    if xmin <= L1_km <= xmax:
        ax_orb.axvline(L1_km, color=_DIM, lw=0.6, ls=":", alpha=0.6)
        ax_orb.text(L1_km + 200, ymin + (ymax - ymin) * 0.95,
                    "L1", color=_DIM, fontsize=7.5)

    # Target
    ax_orb.scatter([d["r_target"][0]], [d["r_target"][1]],
                   s=160, color=_AMBER, marker="*", zorder=8)
    ax_orb.text(d["r_target"][0] + 400, d["r_target"][1] + 600,
                "target", color=_AMBER, fontsize=7.5, zorder=9)

    # Static faint background arcs
    ax_orb.plot(d["X_nom"][:, 0], d["X_nom"][:, 1],
                color=_GREEN, lw=0.7, alpha=0.12, zorder=1, label="Nominal arc")

    # Animated: 2-pass glow trail (uncorrected = amber, corrected = cyan)
    gu, = ax_orb.plot([], [], color=_AMBER, lw=7, alpha=0.09, zorder=3, solid_capstyle="round")
    cu, = ax_orb.plot([], [], color=_AMBER, lw=2, alpha=0.90, zorder=4, solid_capstyle="round")
    du  = ax_orb.scatter([], [], s=90, color=_AMBER, zorder=7,
                         edgecolors=_WHITE, linewidths=0.5)

    gc, = ax_orb.plot([], [], color=_CYAN, lw=7, alpha=0.09, zorder=3, solid_capstyle="round")
    cc, = ax_orb.plot([], [], color=_CYAN, lw=2, alpha=0.90, zorder=4, solid_capstyle="round")
    dc  = ax_orb.scatter([], [], s=90, color=_CYAN, zorder=7,
                         edgecolors=_WHITE, linewidths=0.5)

    burn_sc  = ax_orb.scatter([], [], s=180, color=_RED, marker="D", zorder=9,
                               edgecolors=_WHITE, linewidths=0.8)
    burn_txt = ax_orb.text(0, 0, "", color=_RED, fontsize=8, zorder=10,
                            bbox=dict(facecolor=_BG, edgecolor=_BORDER,
                                      alpha=0.85, boxstyle="round,pad=0.3"))

    legend_handles = [
        plt.Line2D([], [], color=_AMBER, lw=2, label="Off-nominal truth"),
        plt.Line2D([], [], color=_CYAN,  lw=2, label="EKF-corrected arc"),
        plt.Line2D([], [], color=_GREEN, lw=1, alpha=0.5, label="Nominal (reference)"),
        plt.scatter([], [], s=80, color=_AMBER, marker="*", label="Target"),
        plt.scatter([], [], s=80, color=_RED,   marker="D", label="Burn point"),
    ]
    ax_orb.legend(handles=legend_handles, fontsize=7.5, loc="upper left",
                  facecolor=_PANEL, edgecolor=_BORDER, labelcolor=_TEXT)

    # ── miss-distance axes ──
    _dark(ax_mis,
          title="Miss Distance to Target",
          xlabel="Elapsed time  [days]",
          ylabel="‖r(t) − r_target‖  [km]")
    ax_mis.set_yscale("log")

    ax_mis.axvline(d["tc"], color=_RED, lw=1.1, ls=":", alpha=0.65, zorder=2)
    ax_mis.text(d["tc"] + 0.04, 1.0,
                f"midcourse burn\nt = {d['tc']:.2f} days",
                color=_RED, fontsize=7.5, va="bottom", zorder=3)

    valid_cor = np.where(np.isfinite(d["miss_cor"]), d["miss_cor"], np.nan)
    ax_mis.plot(d["t"], d["miss_unc"], color=_AMBER, lw=0.8, alpha=0.22)
    ax_mis.plot(d["t"], valid_cor,     color=_CYAN,  lw=0.8, alpha=0.22)
    ax_mis.plot([], [], color=_AMBER, lw=2, label="No burn")
    ax_mis.plot([], [], color=_CYAN,  lw=2, label="After EKF burn")
    ax_mis.legend(fontsize=8, loc="upper right", facecolor=_PANEL,
                  edgecolor=_BORDER, labelcolor=_TEXT)

    vline      = ax_mis.axvline(d["t"][0], color=_WHITE, lw=1.2, alpha=0.75)
    adot_unc,  = ax_mis.plot([], [], "o", color=_AMBER, ms=7, zorder=6)
    adot_cor,  = ax_mis.plot([], [], "o", color=_CYAN,  ms=7, zorder=6)

    burn_fired = [False]
    burn_xy    = [None]

    def init2():
        for ln in [gu, cu, gc, cc]:
            ln.set_data([], [])
        for sc in [du, dc, burn_sc]:
            sc.set_offsets(np.empty((0, 2)))
        adot_unc.set_data([], [])
        adot_cor.set_data([], [])
        vline.set_xdata([d["t"][0]])
        burn_txt.set_text("")
        return gu, cu, gc, cc, du, dc, burn_sc, vline, adot_unc, adot_cor

    def update2(frame):
        i   = min(frame * speed, N - 1)
        i0  = max(0, i - TRAIL)
        t_n = float(d["t"][i])

        xs_u = d["X_unc"][i0:i+1, 0];  ys_u = d["X_unc"][i0:i+1, 1]
        gu.set_data(xs_u, ys_u);  cu.set_data(xs_u, ys_u)
        du.set_offsets([[d["X_unc"][i, 0], d["X_unc"][i, 1]]])

        cor = d["X_cor"][i0:i+1]
        mask = np.isfinite(cor[:, 0])
        if mask.any():
            gc.set_data(cor[mask, 0], cor[mask, 1])
            cc.set_data(cor[mask, 0], cor[mask, 1])
            dc.set_offsets([[cor[mask, 0][-1], cor[mask, 1][-1]]])
        else:
            gc.set_data([], []);  cc.set_data([], [])
            dc.set_offsets(np.empty((0, 2)))

        if float(d["t_raw"][i]) >= d["tc_raw"] and not burn_fired[0]:
            burn_fired[0] = True
            pre = d["X_cor"][:d["tc_idx"] + 1]
            valid = pre[np.isfinite(pre[:, 0])]
            if len(valid):
                burn_xy[0] = valid[-1]
        if burn_fired[0] and burn_xy[0] is not None:
            burn_sc.set_offsets([burn_xy[0]])
            burn_txt.set_position((burn_xy[0][0] + 500, burn_xy[0][1] + 1_500))
            burn_txt.set_text(f"|ΔV| = {d['dv_ms']:.1f} m/s")

        vline.set_xdata([t_n])
        adot_unc.set_data([t_n], [d["miss_unc"][i]])
        if np.isfinite(d["miss_cor"][i]):
            adot_cor.set_data([t_n], [d["miss_cor"][i]])
        else:
            adot_cor.set_data([], [])

        phase = "PRE-BURN" if float(d["t_raw"][i]) < d["tc_raw"] else "POST-BURN"
        fig.suptitle(
            f"Midcourse Correction  ·  t = {t_n:.2f} days  ·  {phase}",
            color=_TEXT, fontsize=13, y=0.97, fontweight="bold",
        )
        return gu, cu, gc, cc, du, dc, burn_sc, vline, adot_unc, adot_cor, burn_txt

    n_frames = int(np.ceil(N / speed)) + 40
    ani = FuncAnimation(fig, update2, frames=n_frames, init_func=init2,
                        blit=False, interval=1000 // fps)
    _try_save(ani, OUT_DIR / "anim_02_targeting_v2.mp4", fps)
    plt.close(fig)


# ── Phase 2 follow-cam (3D) animation ─────────────────────────────────────────
def animate_phase2_follow_cam(fps: int = 30, sim_speed: float = 0.77) -> None:
    print("Phase 2 follow-cam: computing trajectories …")
    mu, model, x0 = _setup()
    d = _data2_3d(mu, model, x0)
    N = len(d["t"])
    dt_f = float(d["t"][1] - d["t"][0])
    speed = max(1, round(sim_speed * T_DAY / (fps * dt_f)))

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D proj)

    fig = plt.figure(figsize=(10, 12), facecolor=_BG)
    # Stacked layout: 3D follow-cam on top, miss-distance panel below.
    ax     = fig.add_axes([0.02, 0.46, 0.96, 0.48], projection="3d")
    ax_mis = fig.add_axes([0.10, 0.07, 0.85, 0.24])
    ax.set_facecolor(_BG)

    # Dark theme for 3D axes + dimmed grid (≈75% opacity reduction)
    pane_rgba = (0.020, 0.031, 0.063, 1.0)
    grid_rgba = (0.10, 0.13, 0.25, 0.22)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(pane_rgba)
        axis.label.set_color(_TEXT)
        axis._axinfo["grid"]["color"]     = grid_rgba
        axis._axinfo["grid"]["linewidth"] = 0.35
    ax.tick_params(colors=_DIM, labelsize=7)
    ax.set_xlabel("x  [km]", labelpad=6, fontsize=8)
    ax.set_ylabel("y  [km]", labelpad=6, fontsize=8)
    ax.set_zlabel("z  [km]", labelpad=6, fontsize=8)

    # Earth & Moon (exaggerated radii for legibility). Moon uses the
    # equirectangular texture at results/seeds/moon_texture.jpg for a
    # recognisable lunar surface; Earth stays a solid blue sphere.
    _draw_sphere(ax, d["p1"], 9_000.0, _EARTH_C, alpha=0.95)
    _MOON_TEX = Path("results/seeds/moon_texture.jpg")
    if _MOON_TEX.exists():
        _draw_textured_sphere(ax, d["p2"], 5_000.0, _MOON_TEX,
                              n=72, alpha=1.0, rotate_lon_deg=180.0)
    else:
        _draw_sphere(ax, d["p2"], 5_000.0, _MOON_C, alpha=0.90)

    # Target marker
    ax.scatter([d["r_target"][0]], [d["r_target"][1]], [d["r_target"][2]],
               s=180, color=_AMBER, marker="*", zorder=10, depthshade=False)

    # Full trajectories — faint background reference
    ax.plot(d["X_nom"][:, 0], d["X_nom"][:, 1], d["X_nom"][:, 2],
            color=_GREEN, lw=0.7, alpha=0.28, zorder=2)
    ax.plot(d["X_unc"][:, 0], d["X_unc"][:, 1], d["X_unc"][:, 2],
            color=_AMBER, lw=0.6, alpha=0.18, zorder=2)
    cmask_full = np.isfinite(d["X_cor"][:, 0])
    ax.plot(d["X_cor"][cmask_full, 0], d["X_cor"][cmask_full, 1],
            d["X_cor"][cmask_full, 2],
            color=_CYAN, lw=0.6, alpha=0.18, zorder=2)

    # Animated glow trails (2-pass) — uncorrected + corrected
    glow_u, = ax.plot([], [], [], color=_AMBER, lw=7, alpha=0.10, zorder=4)
    line_u, = ax.plot([], [], [], color=_AMBER, lw=2.3, alpha=0.95, zorder=5)
    glow_c, = ax.plot([], [], [], color=_CYAN,  lw=7, alpha=0.10, zorder=4)
    line_c, = ax.plot([], [], [], color=_CYAN,  lw=2.3, alpha=0.95, zorder=5)

    dot_u = ax.scatter([], [], [], s=90, color=_AMBER, edgecolors=_WHITE,
                       linewidths=0.6, zorder=8, depthshade=False)
    dot_c = ax.scatter([], [], [], s=90, color=_CYAN,  edgecolors=_WHITE,
                       linewidths=0.6, zorder=8, depthshade=False)
    burn_sc = ax.scatter([], [], [], s=170, color=_RED, marker="D",
                         edgecolors=_WHITE, linewidths=0.8, zorder=9,
                         depthshade=False)

    # HUD overlays (figure-space text)
    hud_title = fig.text(0.025, 0.968, "", color=_TEXT, fontsize=13,
                         fontweight="bold", family="monospace")
    hud_sub   = fig.text(0.025, 0.946, "", color=_DIM,  fontsize=9.5,
                         family="monospace")
    hud_dv    = fig.text(0.025, 0.335, "", color=_RED,  fontsize=10,
                         fontweight="bold", family="monospace")

    # Static marker legend (between panels) — names the 3D glyphs
    _leg_x_sym, _leg_x_txt = 0.025, 0.050
    fig.text(_leg_x_sym, 0.440, "◆", color=_RED,   fontsize=13,
             fontweight="bold", family="monospace")
    fig.text(_leg_x_txt, 0.442, "midcourse ΔV burn point",
             color=_TEXT, fontsize=8.5, family="monospace")
    fig.text(_leg_x_sym, 0.415, "★", color=_AMBER, fontsize=13,
             fontweight="bold", family="monospace")
    fig.text(_leg_x_txt, 0.417, "target state",
             color=_TEXT, fontsize=8.5, family="monospace")
    fig.text(_leg_x_sym, 0.390, "●", color=_AMBER, fontsize=13,
             fontweight="bold", family="monospace")
    fig.text(_leg_x_txt, 0.392, "off-nominal truth",
             color=_TEXT, fontsize=8.5, family="monospace")
    fig.text(_leg_x_sym, 0.365, "●", color=_CYAN,  fontsize=13,
             fontweight="bold", family="monospace")
    fig.text(_leg_x_txt, 0.367, "EKF-corrected arc",
             color=_TEXT, fontsize=8.5, family="monospace")

    # ── Miss-distance panel (right) ──────────────────────────────────────────
    _dark(ax_mis,
          title="Miss Distance to Target",
          xlabel="Elapsed time  [days]",
          ylabel="‖r(t) − r_target‖  [km]")
    ax_mis.set_yscale("log")
    ax_mis.axvline(d["tc"], color=_RED, lw=1.1, ls=":", alpha=0.65, zorder=2)
    ax_mis.text(d["tc"] + 0.04, 1.0,
                f"midcourse burn\nt = {d['tc']:.2f} days",
                color=_RED, fontsize=7.5, va="bottom", zorder=3)

    _miss_cor_plot = np.where(np.isfinite(d["miss_cor"]), d["miss_cor"], np.nan)
    ax_mis.plot(d["t"], d["miss_unc"],   color=_AMBER, lw=0.8, alpha=0.22)
    ax_mis.plot(d["t"], _miss_cor_plot,  color=_CYAN,  lw=0.8, alpha=0.22)
    ax_mis.plot([], [], color=_AMBER, lw=2, label="No burn")
    ax_mis.plot([], [], color=_CYAN,  lw=2, label="After EKF burn")
    ax_mis.legend(fontsize=8, loc="upper right", facecolor=_PANEL,
                  edgecolor=_BORDER, labelcolor=_TEXT)

    mis_vline = ax_mis.axvline(d["t"][0], color=_WHITE, lw=1.2, alpha=0.75)
    mis_dot_u, = ax_mis.plot([], [], "o", color=_AMBER, ms=7, zorder=6)
    mis_dot_c, = ax_mis.plot([], [], "o", color=_CYAN,  ms=7, zorder=6)

    # Camera state — exponential smoothing for center / half-width
    tc_days    = float(d["tc"])
    t_total    = float(d["t"][-1])
    azim0      = 38.0
    azim_sweep = 78.0               # total azimuth travel across the shot
    elev_start = 34.0
    elev_mid   = 22.0
    elev_end   = 15.0
    D_floor    = 22_000.0           # km; hard lower bound on zoom
    D_ceiling  = 120_000.0
    D_wide     = 78_000.0           # establishing dolly-out peak
    D_tight    = 28_000.0           # post-burn intimate framing
    smooth_D   = 0.96               # EMA retention for half-width (slow zoom)
    smooth_c   = 0.935              # EMA retention for center (pan)
    preroll_frames = 8              # hold on frame 0 for an establishing beat

    cam_state = {"D": None, "center": None}

    burn_fired = [False]
    burn_xyz   = [None]

    def _apply_view(t_n, frac):
        # Elevation eases: high → mid before burn, mid → low after
        pre  = _smoothstep(t_n, 0.0, tc_days)
        post = _smoothstep(t_n, tc_days, tc_days + 1.5)
        elev = (elev_start * (1.0 - pre) + elev_mid * pre) * (1.0 - post) \
               + elev_end * post
        # Azimuth: fast sweep-in pre-burn, gentle settle post-burn
        az_pre  = _smoothstep(t_n, 0.0, tc_days)                # 0 → 1 by burn
        az_post = _smoothstep(t_n, tc_days, t_total)            # 0 → 1 over rest
        azim = azim0 + azim_sweep * (0.65 * az_pre + 0.35 * az_post)
        ax.view_init(elev=elev, azim=azim)

    def init_fc():
        for ln in (glow_u, line_u, glow_c, line_c):
            ln.set_data_3d([], [], [])
        dot_u._offsets3d   = ([], [], [])
        dot_c._offsets3d   = ([], [], [])
        burn_sc._offsets3d = ([], [], [])
        hud_title.set_text(""); hud_sub.set_text(""); hud_dv.set_text("")
        mis_dot_u.set_data([], [])
        mis_dot_c.set_data([], [])
        mis_vline.set_xdata([d["t"][0]])
        return glow_u, line_u, glow_c, line_c, dot_u, dot_c, burn_sc

    def update_fc(frame):
        # Pre-roll hold: freeze on frame 0 for the first few frames so the
        # opening reads as a settled establishing shot rather than mid-motion.
        eff = max(0, frame - preroll_frames)
        i  = min(eff * speed, N - 1)
        i0 = max(0, i - TRAIL)
        t_n = float(d["t"][i])

        # Uncorrected trail
        seg_u = d["X_unc"][i0:i+1]
        glow_u.set_data_3d(seg_u[:, 0], seg_u[:, 1], seg_u[:, 2])
        line_u.set_data_3d(seg_u[:, 0], seg_u[:, 1], seg_u[:, 2])
        cur_u = d["X_unc"][i]
        dot_u._offsets3d = ([cur_u[0]], [cur_u[1]], [cur_u[2]])

        # Corrected trail (only after it exists)
        seg_c = d["X_cor"][i0:i+1]
        cmask = np.isfinite(seg_c[:, 0])
        cur_c = None
        if cmask.any():
            xc, yc, zc = seg_c[cmask, 0], seg_c[cmask, 1], seg_c[cmask, 2]
            glow_c.set_data_3d(xc, yc, zc)
            line_c.set_data_3d(xc, yc, zc)
            cur_c = np.array([xc[-1], yc[-1], zc[-1]])
            dot_c._offsets3d = ([cur_c[0]], [cur_c[1]], [cur_c[2]])
        else:
            glow_c.set_data_3d([], [], [])
            line_c.set_data_3d([], [], [])
            dot_c._offsets3d = ([], [], [])

        # Burn event
        if float(d["t_raw"][i]) >= d["tc_raw"] and not burn_fired[0]:
            burn_fired[0] = True
            pre = d["X_cor"][:d["tc_idx"] + 1]
            valid = pre[np.isfinite(pre[:, 0])]
            if len(valid):
                burn_xyz[0] = valid[-1]
        if burn_fired[0] and burn_xyz[0] is not None:
            b = burn_xyz[0]
            burn_sc._offsets3d = ([b[0]], [b[1]], [b[2]])

        # Follow-cam: center drifts from target-weighted midpoint → near spacecraft
        target = np.asarray(d["r_target"], dtype=float)
        sc = cur_c if (burn_fired[0] and cur_c is not None) else cur_u
        mid = 0.5 * (target + sc)
        dist = float(np.linalg.norm(sc - target))

        frac = _smoothstep(t_n, tc_days - 1.2, tc_days + 0.6)

        # Scheduled zoom — decoupled from instantaneous distance so small
        # per-frame wiggles don't breathe the camera. Dolly-out early for an
        # establishing beat, then smoothly push in through the burn.
        dolly_out  = _smoothstep(t_n, 0.0, tc_days * 0.45)     # 0 → 1 over opener
        dolly_in   = _smoothstep(t_n, tc_days * 0.45, tc_days + 0.9)  # pushes in
        target_D   = (D_floor * (1.0 - dolly_out) + D_wide * dolly_out) * (1.0 - dolly_in) \
                     + D_tight * dolly_in
        # Gentle safety net in case spacecraft wanders far from target
        target_D   = max(target_D, 0.85 * dist + 6_000.0)
        target_D   = float(np.clip(target_D, D_floor, D_ceiling))

        # Camera center: bias toward target pre-burn, toward spacecraft post-burn
        target_center = (1.0 - 0.6 * frac) * mid + (0.6 * frac) * sc

        if cam_state["D"] is None:
            cam_state["D"] = target_D
            cam_state["center"] = target_center.copy()
        else:
            cam_state["D"] = smooth_D * cam_state["D"] + (1 - smooth_D) * target_D
            cam_state["center"] = (smooth_c * cam_state["center"]
                                   + (1 - smooth_c) * target_center)

        D = cam_state["D"]
        cx, cy, cz = cam_state["center"]
        ax.set_xlim(cx - D,       cx + D)
        ax.set_ylim(cy - D,       cy + D)
        ax.set_zlim(cz - D * 0.55, cz + D * 0.55)
        ax.set_box_aspect((1.0, 1.0, 0.55))

        _apply_view(t_n, frac)

        # Miss-distance panel updates
        mis_vline.set_xdata([t_n])
        mis_dot_u.set_data([t_n], [d["miss_unc"][i]])
        if np.isfinite(d["miss_cor"][i]):
            mis_dot_c.set_data([t_n], [d["miss_cor"][i]])
        else:
            mis_dot_c.set_data([], [])

        phase = "PRE-BURN" if float(d["t_raw"][i]) < d["tc_raw"] else "POST-BURN"
        hud_title.set_text(
            f"CR3BP FOLLOW-CAM   t = {t_n:5.2f} days   {phase}"
        )
        if burn_fired[0]:
            hud_sub.set_text(
                f"Spacecraft → target  ·  miss = {dist:8,.0f} km"
            )
            hud_dv.set_text(f"|ΔV| APPLIED = {d['dv_ms']:6.2f} m/s")
        else:
            hud_sub.set_text(
                f"Off-nominal drift  ·  range-to-target = {dist:8,.0f} km"
            )
            hud_dv.set_text("targeting solve in progress …")

        return (glow_u, line_u, glow_c, line_c, dot_u, dot_c, burn_sc,
                hud_title, hud_sub, hud_dv, mis_vline, mis_dot_u, mis_dot_c)

    n_frames = int(np.ceil(N / speed)) + 50 + preroll_frames
    ani = FuncAnimation(fig, update_fc, frames=n_frames, init_func=init_fc,
                        blit=False, interval=1000 // fps)
    _try_save(ani, OUT_DIR / "anim_02_targeting_follow_cam.mp4", fps)
    plt.close(fig)


# ── Phase 3 animation ──────────────────────────────────────────────────────────
def animate_phase3(fps: int = 30, sim_speed: float = 0.77) -> None:
    """3D cislunar bearing-only EKF navigation animation.

    Left panel: 3D orbit view with Moon sphere, smooth trajectory traces
    (truth + EKF estimate), animated LOS rays that pulse on measurement
    events, and a slowly-rotating camera. Data is densely interpolated
    so panning is continuous instead of stepping through sparse updates.

    Right panels: position estimation error (log, with 3σ envelope) and
    NIS with χ²(2) 95% band.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D proj

    print("Phase 3 · running EKF + building dense trajectories …")
    mu, model, x0 = _setup()
    d = _data3(mu, model, x0)
    N = len(d["t"])

    # Playback rate: advance `speed` dense ticks per video frame.
    # sim_speed here is a rate multiplier — 1.0 means real time.
    dt_f = float(d["t"][1] - d["t"][0])
    speed = max(1, round(sim_speed * T_DAY / (fps * dt_f)))

    NIS_LO = float(chi2.ppf(0.025, df=2))
    NIS_HI = float(chi2.ppf(0.975, df=2))

    # ── figure layout ────────────────────────────────────────────────
    # Explicit axes positioning keeps the 3D panel from losing space to
    # matplotlib's perspective padding (which is what made the old
    # gridspec-driven layout look empty).
    fig = plt.figure(figsize=(18, 9), facecolor=_BG)
    # 3D orbit: big, left-of-center, generously padded
    ax_orb = fig.add_axes([-0.05, 0.02, 0.78, 0.92], projection="3d")
    # Right-column 2D panels
    ax_err = fig.add_axes([0.66, 0.57, 0.31, 0.34])
    ax_nis = fig.add_axes([0.66, 0.12, 0.31, 0.34])

    # ── 3D orbit axes dark theme ─────────────────────────────────────
    ax_orb.set_facecolor(_BG)
    pane_rgba = (0.020, 0.031, 0.063, 1.0)
    grid_rgba = (0.10, 0.13, 0.25, 0.22)
    for axis in (ax_orb.xaxis, ax_orb.yaxis, ax_orb.zaxis):
        axis.set_pane_color(pane_rgba)
        axis.label.set_color(_TEXT)
        axis._axinfo["grid"]["color"] = grid_rgba
        axis._axinfo["grid"]["linewidth"] = 0.35
    ax_orb.tick_params(colors=_DIM, labelsize=7)
    ax_orb.set_xlabel("x  [km]", labelpad=6, fontsize=8)
    ax_orb.set_ylabel("y  [km]", labelpad=6, fontsize=8)
    ax_orb.set_zlabel("z  [km]", labelpad=6, fontsize=8)

    # Moon — textured sphere from the repo's moon_texture.jpg; radius is
    # exaggerated for legibility at cislunar scale. Texture is sampled once
    # at setup so per-frame render cost matches a plain sphere.
    _MOON_TEX = Path("results/seeds/moon_texture.jpg")
    if _MOON_TEX.exists():
        _draw_textured_sphere(ax_orb, d["p2"], 12_000.0, _MOON_TEX,
                              n=80, alpha=1.0, rotate_lon_deg=180.0)
    else:
        _draw_sphere(ax_orb, d["p2"], 12_000.0, _MOON_C, alpha=0.95)

    # Animated glow trails — truth (cyan) and estimate (amber)
    glow_t, = ax_orb.plot([], [], [], color=_CYAN,  lw=7, alpha=0.11, zorder=4)
    line_t, = ax_orb.plot([], [], [], color=_CYAN,  lw=2.3, alpha=0.95, zorder=5)
    glow_h, = ax_orb.plot([], [], [], color=_AMBER, lw=7, alpha=0.09, zorder=4)
    line_h, = ax_orb.plot([], [], [], color=_AMBER, lw=2.0, alpha=0.90, zorder=5, ls="--")

    # Live spacecraft / estimate markers
    dot_t = ax_orb.scatter([], [], [], s=110, color=_CYAN,  zorder=8,
                           edgecolors=_WHITE, linewidths=0.6, depthshade=False)
    dot_h = ax_orb.scatter([], [], [], s=90,  color=_AMBER, zorder=8,
                           edgecolors=_WHITE, linewidths=0.6, depthshade=False,
                           marker="s")
    # LOS ray (pulses on measurement events)
    los_line, = ax_orb.plot([], [], [], color=_VIOLET, lw=1.6, alpha=0.70,
                            ls=":", zorder=3)
    # Measurement-pulse halo (brief glow on accepted measurements)
    pulse, = ax_orb.plot([], [], [], color=_GREEN, lw=10, alpha=0.0, zorder=6)

    # Axis limits — include Moon + trajectory; preserve true physical aspect
    # (z-extent naturally much smaller than x,y for a near-planar halo, so we
    # pick a box_aspect that roughly matches the real ranges rather than
    # forcing [1,1,1], which wastes ~60% of the panel on empty z-space).
    all_pts = np.vstack([d["X_true"], d["X_hat"], d["p2"][None, :]])
    pad = 0.15 * (all_pts.max(axis=0) - all_pts.min(axis=0) + 1.0)
    lo = all_pts.min(axis=0) - pad
    hi = all_pts.max(axis=0) + pad
    ranges = hi - lo
    ax_orb.set_xlim(lo[0], hi[0]); ax_orb.set_ylim(lo[1], hi[1]); ax_orb.set_zlim(lo[2], hi[2])
    try:
        ax_orb.set_box_aspect(tuple(ranges / ranges.max()))
    except Exception:
        pass

    # Static 3D legend (figure-space so it doesn't rotate with axes)
    fig.text(0.015, 0.88, "● truth trajectory", color=_CYAN,
             fontsize=10, family="monospace", fontweight="bold")
    fig.text(0.015, 0.855, "■ IEKF estimate", color=_AMBER,
             fontsize=10, family="monospace", fontweight="bold")
    fig.text(0.015, 0.830, "┈ LOS to Moon", color=_VIOLET,
             fontsize=10, family="monospace", fontweight="bold")
    fig.text(0.015, 0.805, "● Moon target", color=_MOON_C,
             fontsize=10, family="monospace", fontweight="bold")

    # ── position-error panel ─────────────────────────────────────────
    _dark(ax_err,
          title="Position Estimation Error",
          xlabel="Elapsed time  [days]",
          ylabel="‖r̂ − r_true‖  [km]")
    ax_err.set_yscale("log")
    ax_err.plot(d["t"], d["pos_err"], color=_CYAN, lw=0.7, alpha=0.18)
    ax_err.fill_between(d["t"], 1e-8, d["sig3"], color=_VIOLET, alpha=0.10, zorder=1)
    ax_err.plot([], [], color=_CYAN,   lw=2, label="‖r̂ − r‖")
    ax_err.plot([], [], color=_VIOLET, lw=6, alpha=0.35, label="3σ bound (x)")
    ax_err.legend(fontsize=7.5, loc="upper right",
                  facecolor=_PANEL, edgecolor=_BORDER, labelcolor=_TEXT)
    # Use a fixed-y log range so the trace doesn't rescale mid-animation
    pe_finite = d["pos_err"][np.isfinite(d["pos_err"]) & (d["pos_err"] > 0)]
    if pe_finite.size:
        ax_err.set_ylim(max(pe_finite.min() * 0.5, 1e-3), pe_finite.max() * 3.0)
    e_vline = ax_err.axvline(d["t"][0], color=_WHITE, lw=1.1, alpha=0.7)
    e_line, = ax_err.plot([], [], color=_CYAN, lw=2.0, zorder=5)
    e_dot,  = ax_err.plot([], [], "o", color=_CYAN, ms=6, zorder=6)

    # ── NIS panel ────────────────────────────────────────────────────
    _dark(ax_nis,
          title="Filter Consistency  ·  NIS vs χ²(2) gate",
          xlabel="Elapsed time  [days]",
          ylabel="NIS")
    ax_nis.fill_between([d["t"][0], d["t"][-1]], NIS_LO, NIS_HI,
                        color=_GREEN, alpha=0.14, zorder=1,
                        label=f"95% χ²(2) band  [{NIS_LO:.2f}, {NIS_HI:.2f}]")
    ax_nis.axhline(2.0, color=_GREEN, lw=0.9, ls="--", alpha=0.55)
    nis_fin = d["nis"][np.isfinite(d["nis"])]
    ax_nis.set_ylim(-0.2, max(14.0, float(nis_fin.max()) * 1.10) if nis_fin.size else 14.0)
    ax_nis.set_xlim(d["t"][0], d["t"][-1])
    ax_nis.legend(fontsize=7.5, loc="upper right",
                  facecolor=_PANEL, edgecolor=_BORDER, labelcolor=_TEXT)
    n_vline = ax_nis.axvline(d["t"][0], color=_WHITE, lw=1.1, alpha=0.7)
    nis_dots_in,  = ax_nis.plot([], [], "o", color=_GREEN, ms=4.5, alpha=0.92, zorder=4)
    nis_dots_out, = ax_nis.plot([], [], "o", color=_RED,   ms=4.5, alpha=0.92, zorder=4)

    nis_in_t,  nis_in_v  = [], []
    nis_out_t, nis_out_v = [], []
    last_meas_idx = [-1]
    pulse_timer   = [0]

    # Camera orbit — slow parallax rotation for depth perception
    elev_base = 22.0
    azim_base = -55.0
    azim_span = 45.0

    def init3():
        for ln in (glow_t, line_t, glow_h, line_h, los_line, pulse):
            ln.set_data([], []); ln.set_3d_properties([])
        for sc in (dot_t, dot_h):
            sc._offsets3d = ([], [], [])
        e_vline.set_xdata([d["t"][0]])
        n_vline.set_xdata([d["t"][0]])
        e_line.set_data([], [])
        e_dot.set_data([], [])
        nis_dots_in.set_data([], [])
        nis_dots_out.set_data([], [])
        return (glow_t, line_t, glow_h, line_h, los_line, pulse,
                dot_t, dot_h, e_vline, n_vline, e_line, e_dot,
                nis_dots_in, nis_dots_out)

    def update3(frame):
        i  = min(frame * speed, N - 1)
        i0 = max(0, i - TRAIL)
        t_n = float(d["t"][i])

        # Trails
        xs_t = d["X_true"][i0:i+1, 0]; ys_t = d["X_true"][i0:i+1, 1]; zs_t = d["X_true"][i0:i+1, 2]
        glow_t.set_data(xs_t, ys_t);  glow_t.set_3d_properties(zs_t)
        line_t.set_data(xs_t, ys_t);  line_t.set_3d_properties(zs_t)

        xs_h = d["X_hat"][i0:i+1, 0];  ys_h = d["X_hat"][i0:i+1, 1];  zs_h = d["X_hat"][i0:i+1, 2]
        glow_h.set_data(xs_h, ys_h);  glow_h.set_3d_properties(zs_h)
        line_h.set_data(xs_h, ys_h);  line_h.set_3d_properties(zs_h)

        # Markers
        dot_t._offsets3d = ([d["X_true"][i, 0]], [d["X_true"][i, 1]], [d["X_true"][i, 2]])
        dot_h._offsets3d = ([d["X_hat"][i, 0]],  [d["X_hat"][i, 1]],  [d["X_hat"][i, 2]])

        # LOS ray (truth → Moon)
        rx, ry, rz = d["X_true"][i]
        mx, my, mz = d["p2"]
        los_line.set_data([rx, mx], [ry, my]); los_line.set_3d_properties([rz, mz])

        # Measurement pulse: fade-in for ~4 frames each time meas idx advances
        meas_idx = int(d["meas_idx_per_dense"][i])
        if meas_idx > last_meas_idx[0] and np.isfinite(d["nis"][meas_idx]):
            last_meas_idx[0] = meas_idx
            pulse_timer[0] = 6
            # accumulate NIS scatter
            tk = float(d["t_meas"][meas_idx])
            nv = float(d["nis"][meas_idx])
            if NIS_LO <= nv <= NIS_HI:
                nis_in_t.append(tk);  nis_in_v.append(nv)
            else:
                nis_out_t.append(tk); nis_out_v.append(nv)

        if pulse_timer[0] > 0:
            pulse.set_data([rx, mx], [ry, my]); pulse.set_3d_properties([rz, mz])
            pulse.set_alpha(0.05 + 0.06 * pulse_timer[0])
            pulse_timer[0] -= 1
        else:
            pulse.set_alpha(0.0)

        # Slow camera rotation (smooth sinusoid)
        tau = t_n / d["t"][-1]
        ax_orb.view_init(
            elev=elev_base + 3.0 * np.sin(2 * np.pi * tau),
            azim=azim_base + azim_span * tau,
        )

        # Right panels
        e_vline.set_xdata([t_n]); n_vline.set_xdata([t_n])
        e_line.set_data(d["t"][:i+1], d["pos_err"][:i+1])
        e_dot.set_data([t_n], [d["pos_err"][i]])
        nis_dots_in.set_data(nis_in_t, nis_in_v)
        nis_dots_out.set_data(nis_out_t, nis_out_v)

        fig.suptitle(
            f"Bearing-Only IEKF Navigation  ·  3D Cislunar Arc  ·  "
            f"t = {t_n:5.2f} days   pos error = {d['pos_err'][i]:,.0f} km",
            color=_TEXT, fontsize=13, y=0.965, fontweight="bold",
        )
        return (glow_t, line_t, glow_h, line_h, los_line, pulse,
                dot_t, dot_h, e_vline, n_vline, e_line, e_dot,
                nis_dots_in, nis_dots_out)

    n_frames = int(np.ceil(N / speed)) + 40
    ani = FuncAnimation(fig, update3, frames=n_frames, init_func=init3,
                        blit=False, interval=1000 // fps)
    _try_save(ani, OUT_DIR / "anim_03_bearings_v2.mp4", fps)
    plt.close(fig)


# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["2", "2f", "3"], default=None,
                        help="Render only phase 2, 2f (3D follow-cam), or 3 "
                             "(default: phases 2 and 3)")
    parser.add_argument("--fps",        type=int,   default=30)
    parser.add_argument("--sim-speed",  type=float, default=0.77,
                        help="Simulated CR3BP TU per real second")
    args = parser.parse_args()

    if args.phase in (None, "2"):
        animate_phase2(fps=args.fps, sim_speed=args.sim_speed)
    if args.phase == "2f":
        animate_phase2_follow_cam(fps=args.fps, sim_speed=args.sim_speed)
    if args.phase in (None, "3"):
        animate_phase3(fps=args.fps, sim_speed=args.sim_speed)
