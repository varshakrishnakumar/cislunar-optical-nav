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
from matplotlib.lines import Line2D
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

REPO_ROOT_PATH = Path(__file__).resolve().parent.parent
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


def _try_save(ani, path: Path, fps: int, *, bitrate: int = 2_400):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        w = FFMpegWriter(
            fps=fps,
            bitrate=bitrate,
            extra_args=[
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ],
        )
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


# ── SPICE↔CR3BP comparison helpers ───────────────────────────────────────────
def _propagate_cr3bp_from_spice_ic(arc):
    """Re-propagate the SPICE arc's initial state under CR3BP dynamics.

    Returns the CR3BP-propagated trajectory in the same Moon-centered
    synodic-km frame as the SPICE input. The two arcs share an IC and
    diverge over the 6.56-day window — that visible separation is the
    point of overlaying them: SPICE = ephemeris truth (DE442s), CR3BP =
    rotating-frame approximation with constant n.

    Velocity at t=0 is taken from a cubic spline of the SPICE synodic
    position; tiny inconsistencies between SPICE-synodic and CR3BP-synodic
    rotation rates show up as additional drift, which is honest for the
    audience: it's exactly the modeling error the navigation work has to
    absorb downstream.
    """
    from scipy.interpolate import CubicSpline
    from dynamics.cr3bp import CR3BP
    from dynamics.integrators import propagate

    mu = 0.0121505856
    em_mean = float(np.mean(arc["em_distance"]))
    moon_x_bary = (1.0 - mu) * em_mean      # Moon's mean synodic-x position

    # Re-center to barycenter (CR3BP convention): + Moon's synodic position
    X_bary_km = arc["X"] + np.array([moon_x_bary, 0.0, 0.0])

    spl = CubicSpline(arc["t_days"], X_bary_km, axis=0)
    pos0_km = spl(0.0)              # km
    vel0_kmpd = spl(0.0, 1)         # km / day
    vel0_kmps = vel0_kmpd / 86_400.0

    L = 384_400.0
    V = L / (T_DAY * 86_400.0)      # ≈ 1.024 km/s per DU/TU
    s0 = np.concatenate([pos0_km / L, vel0_kmps / V])

    model = CR3BP(mu=mu)
    t_tu = arc["t_days"] / T_DAY
    res = propagate(model.eom, (float(t_tu[0]), float(t_tu[-1])), s0,
                    t_eval=t_tu, rtol=1e-11, atol=1e-13, method="DOP853")
    X_cr3bp_bary_km = res.x[:, :3] * L
    return X_cr3bp_bary_km - np.array([moon_x_bary, 0.0, 0.0])


def _occlusion_mask(X, view_dir, occ_center, occ_radius):
    """Boolean mask: True where each row of X is hidden behind a sphere
    of radius `occ_radius` centered at `occ_center`, as seen from the
    camera direction `view_dir` (unit vector pointing from focus toward
    camera; positive = in front)."""
    rel = X - occ_center
    z_along = rel @ view_dir
    perp_sq = (rel * rel).sum(axis=1) - z_along * z_along
    return (z_along < 0.0) & (perp_sq < occ_radius * occ_radius)


def _camera_dir(elev_deg, azim_deg):
    el = np.radians(elev_deg)
    az = np.radians(azim_deg)
    return np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])


# ── SPICE-seeded mission-arc loader ──────────────────────────────────────────
def _load_spice_mission_arc():
    """Load the 6.56-day NRHO arc (synodic frame, km) seeded from SPICE.

    Returns Moon-centered synodic coordinates so the geometry is read
    relative to the body the audience is looking at. The Moon's mean
    synodic position over the arc is subtracted out — the residual y/z
    motion of the Moon (a few hundred km from EM-distance variation) is
    well below the trajectory's own scale and reads as static.
    """
    csvp = REPO_ROOT_PATH / "results" / "seeds" / "spice_nrho_seed_relative.csv"
    if not csvp.exists():
        return None
    import csv as _csv
    with csvp.open() as f:
        rows = list(_csv.DictReader(f))
    t_days = np.array([float(r["t_days"]) for r in rows])
    syn = np.array([
        [float(r["synodic_x_km"]), float(r["synodic_y_km"]), float(r["synodic_z_km"])]
        for r in rows
    ])
    em_dist = np.array([float(r["earth_moon_distance_km"]) for r in rows])
    moon_range = np.array([float(r["moon_range_km"]) for r in rows])
    # Moon synodic position ≈ (1-mu) * EM_distance on +x; barycenter at origin.
    mu = 0.0121505856
    moon_syn = np.column_stack([(1.0 - mu) * em_dist,
                                np.zeros_like(em_dist),
                                np.zeros_like(em_dist)])
    moon_mean = moon_syn.mean(axis=0)
    X_moon = syn - moon_mean[None, :]    # spacecraft, Moon-centered synodic km
    earth_mean = -mu * em_dist.mean()    # barycenter→Earth on -x; subtract moon_mean
    earth_pos = np.array([earth_mean - moon_mean[0], 0.0, 0.0])  # Moon-centered
    return dict(
        t_days=t_days,
        X=X_moon,                     # (N, 3) Moon-centered synodic km
        moon_range=moon_range,
        em_distance=em_dist,
        earth_pos=earth_pos,
    )


# ── targeting arcs (uncorrected vs corrected) from a SPICE-seeded IC ──────────
def _targeting_arcs_from_spice_ic(arc, *, tc_days: float = 4.5,
                                   dx0_du=(2e-4, -1e-4, 0.0,
                                           0.0, 2e-3, 0.0)):
    """Build (uncorrected, corrected) trajectory pairs from the SPICE IC.

    The SPICE-seeded initial state is perturbed by a small position +
    velocity error (DU, DU/TU).  Two arcs are then propagated under
    CR3BP dynamics:

      • uncorrected  — perturbed IC, no burn (the "miss" arc)
      • corrected    — perturbed IC + a single ΔV at t = tc that returns
                       x(tf) to the unperturbed endpoint

    All arrays are returned in the same Moon-centered synodic-km frame
    used by the rest of the follow-cam, sampled on `arc['t_days']`.
    """
    from scipy.interpolate import CubicSpline

    from dynamics.cr3bp import CR3BP
    from dynamics.integrators import propagate
    from dynamics.variational import cr3bp_eom_with_stm

    mu = 0.0121505856
    L = 384_400.0
    V = L / (T_DAY * 86_400.0)            # km/s per DU/TU
    em_mean = float(np.mean(arc["em_distance"]))
    moon_x_bary = (1.0 - mu) * em_mean
    moon_offset = np.array([moon_x_bary, 0.0, 0.0])

    # Recover the IC in barycentric synodic km, then convert to DU/TU.
    X_bary_km = arc["X"] + moon_offset
    spl = CubicSpline(arc["t_days"], X_bary_km, axis=0)
    pos0_km = spl(0.0)
    vel0_kmps = spl(0.0, 1) / 86_400.0
    s0_du = np.concatenate([pos0_km / L, vel0_kmps / V])

    model = CR3BP(mu=mu)
    eom = model.eom

    t_days = arc["t_days"]
    t_tu   = t_days / T_DAY
    t0_tu, tf_tu = float(t_tu[0]), float(t_tu[-1])
    tc_tu = float(tc_days) / T_DAY

    # Nominal arc — what the spacecraft "should" follow without error.
    res_nom = propagate(eom, (t0_tu, tf_tu), s0_du, dense_output=True,
                        rtol=1e-11, atol=1e-13, method="DOP853")
    r_target_du = res_nom.sol(tf_tu)[:3]

    s0_err = s0_du + np.asarray(dx0_du, dtype=float)

    # Uncorrected (no-burn) arc — the persistent ghost showing the miss.
    res_unc = propagate(eom, (t0_tu, tf_tu), s0_err, dense_output=True,
                        rtol=1e-11, atol=1e-13, method="DOP853")

    # Single-impulse targeting at tc: solve dv such that x(tf) ≈ r_target.
    res_tc_stm = propagate(
        lambda t, z: cr3bp_eom_with_stm(t, z, mu),
        (t0_tu, tc_tu), _pack(s0_err),
        rtol=1e-11, atol=1e-13, method="DOP853",
    )
    x_tc, _ = _unpack(res_tc_stm.x[-1])
    dv = np.zeros(3)
    for _ in range(15):
        xb = x_tc.copy(); xb[3:6] += dv
        res_tf_stm = propagate(
            lambda t, z: cr3bp_eom_with_stm(t, z, mu),
            (tc_tu, tf_tu), _pack(xb),
            rtol=1e-11, atol=1e-13, method="DOP853",
        )
        x_tf, phi = _unpack(res_tf_stm.x[-1])
        err = x_tf[:3] - r_target_du
        if np.linalg.norm(err) < 1e-10:
            break
        try:
            dv -= np.linalg.solve(phi[:3, 3:6], err)
        except np.linalg.LinAlgError:
            dv -= np.linalg.lstsq(phi[:3, 3:6], err, rcond=None)[0]

    # Corrected arc — perturbed IC, applies dv at tc, then re-propagates.
    res_pre = propagate(eom, (t0_tu, tc_tu), s0_err, dense_output=True,
                        rtol=1e-11, atol=1e-13, method="DOP853")
    xb_post = res_pre.sol(tc_tu).reshape(6,)
    xb_post[3:6] += dv
    res_post = propagate(eom, (tc_tu, tf_tu), xb_post, dense_output=True,
                         rtol=1e-11, atol=1e-13, method="DOP853")

    X_unc_du = res_unc.sol(t_tu).T[:, :3]
    X_cor_du = np.empty((len(t_tu), 3))
    pre_mask = t_tu <= tc_tu
    X_cor_du[pre_mask]  = res_pre.sol(t_tu[pre_mask]).T[:, :3]
    X_cor_du[~pre_mask] = res_post.sol(t_tu[~pre_mask]).T[:, :3]

    X_unc_km = X_unc_du * L - moon_offset
    X_cor_km = X_cor_du * L - moon_offset
    r_target_km = r_target_du * L - moon_offset

    miss_unc_km = np.linalg.norm(X_unc_km - r_target_km, axis=1)
    miss_cor_km = np.linalg.norm(X_cor_km - r_target_km, axis=1)

    dv_kmps = float(np.linalg.norm(dv)) * V
    return dict(
        X_unc=X_unc_km, X_cor=X_cor_km,
        miss_unc_km=miss_unc_km, miss_cor_km=miss_cor_km,
        r_target_km=r_target_km,
        dv_ms=dv_kmps * 1000.0,
        tc_days=float(tc_days),
        # Velocity direction of the corrected arc just after the burn —
        # used to draw a notional ΔV cue at burn time.
        v_post=(X_cor_km[(np.abs(t_days - tc_days)).argmin() + 4]
                - X_cor_km[(np.abs(t_days - tc_days)).argmin()]),
    )


# ── Phase 2 follow-cam (3D) animation — mission-geometry version ──────────────
def animate_phase2_follow_cam(fps: int = 30, sim_speed: float = 0.18) -> None:
    """Mission-geometry video for the L2 Southern Halo / NRHO scenario.

    The arc includes three phases — insertion, tracking, and a midcourse
    correction — read directly off the trail's color.  The persistent
    dashed violet ghost shows the uncorrected (no-burn) arc; the bright
    phase-coloured trail shows the trajectory the single midcourse burn
    actually flies.  A live miss-distance overlay gives the burn its
    quantitative payoff: ~thousands of km without correction, ~kilometres
    with.

    Slide 03 audience cues:
      1. Where: Moon-centered, Earth–Moon CR3BP rotating frame.
      2. What: a 6.56-day halo arc, SPICE-seeded IC.
      3. Mission flow: insertion → tracking → midcourse correction.
    """
    print("Phase 2 mission-arc: loading SPICE-seeded NRHO …")
    arc = _load_spice_mission_arc()
    if arc is None:
        print("  spice_nrho_seed_relative.csv not found; falling back to CR3BP propagation")
        mu, model, x0 = _setup()
        d_fb = _data2_3d(mu, model, x0)
        moon_pos = np.asarray(d_fb["p2"], dtype=float)
        arc = dict(
            t_days=d_fb["t"],
            X=d_fb["X_nom"] - moon_pos[None, :],
            moon_range=np.linalg.norm(d_fb["X_nom"] - moon_pos[None, :], axis=1),
            em_distance=np.full_like(d_fb["t"], 384_400.0),
            earth_pos=np.asarray(d_fb["p1"]) - moon_pos,
        )

    # Real targeting story: perturbed IC, single midcourse burn at tc,
    # plus the no-burn miss arc as a persistent ghost.
    print("  solving single-impulse targeting on SPICE IC …")
    targ = _targeting_arcs_from_spice_ic(arc, tc_days=4.5)
    end_cor_miss = "≈ 0" if abs(targ["miss_cor_km"][-1]) < 0.5 \
                   else f"{targ['miss_cor_km'][-1]:,.0f}"
    print(f"  midcourse ΔV = {targ['dv_ms']:6.1f} m/s  ·  "
          f"end miss: {targ['miss_unc_km'][-1]:,.0f} km  →  "
          f"{end_cor_miss} km")

    # Densify for smooth playback.  Three trajectory layers participate:
    #   X_dense       — corrected CR3BP arc (bright animated trail)
    #   X_unc_dense   — uncorrected (no-burn) CR3BP arc (miss ghost)
    #   X_spice_dense — SPICE-truth arc (model-error ghost; CR3BP vs SPICE)
    from scipy.interpolate import CubicSpline
    N_dense = 1200
    t_dense = np.linspace(arc["t_days"][0], arc["t_days"][-1], N_dense)
    X_dense       = CubicSpline(arc["t_days"], targ["X_cor"], axis=0)(t_dense)
    X_unc_dense   = CubicSpline(arc["t_days"], targ["X_unc"], axis=0)(t_dense)
    X_spice_dense = CubicSpline(arc["t_days"], arc["X"],      axis=0)(t_dense)
    range_dense   = CubicSpline(arc["t_days"], arc["moon_range"])(t_dense)
    miss_unc_d    = CubicSpline(arc["t_days"], targ["miss_unc_km"])(t_dense)
    miss_cor_d    = CubicSpline(arc["t_days"], targ["miss_cor_km"])(t_dense)
    r_target_km   = targ["r_target_km"]
    drift_km = np.linalg.norm(X_dense - X_spice_dense, axis=1)
    print(f"  CR3BP↔SPICE drift over {arc['t_days'][-1]:.2f} d: "
          f"max = {drift_km.max():,.0f} km, end = {drift_km[-1]:,.0f} km")

    t_total = float(t_dense[-1])
    N = N_dense
    # Playback rate: simulated days per real second.
    sim_dps = sim_speed * T_DAY
    dt_f = t_total / (N - 1)
    speed = max(1, round(sim_dps / (fps * dt_f)))

    # Mission phase boundaries (days). Trajectory color changes at each
    # boundary — insertion = amber, tracking = cyan, post-burn = green.
    # Midcourse sits in post-perilune cruise (~t=4.5 d) so it doesn't get
    # crowded by the close-approach geometry around perilune (~3.4 d).
    t_insertion_end = 0.50           # insertion → tracking handoff
    t_burn          = float(targ["tc_days"])  # midcourse correction event (days)
    t_burn_label    = 0.45           # banner fade-in widths (days)

    burn_idx = int(np.argmin(np.abs(t_dense - t_burn)))
    burn_pos = X_dense[burn_idx].copy()

    # Subtle ΔV vector at burn — direction along velocity, length scaled to
    # be visually subtle (≈3% of camera frame). This is a notional cue, not
    # a targeting solve.
    if burn_idx + 5 < N:
        v_hat = X_dense[burn_idx + 5] - X_dense[burn_idx]
    else:
        v_hat = X_dense[burn_idx] - X_dense[burn_idx - 5]
    v_norm = np.linalg.norm(v_hat)
    v_hat = v_hat / v_norm if v_norm > 1e-9 else np.array([1.0, 0.0, 0.0])

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D proj)

    # ── figure layout: full-bleed 3D, no side panels ─────────────────────────
    # Target placement: a 9.8" × 8.83" near-square slot in the slide.
    fig = plt.figure(figsize=(9.8, 8.83), dpi=200, facecolor=_BG)
    ax  = fig.add_axes([0.0, 0.04, 1.0, 0.94], projection="3d",
                       computed_zorder=False)
    ax.set_facecolor(_BG)

    # Dark theme: hide axis ticks/labels entirely so the geometry breathes.
    pane_rgba = (0.020, 0.031, 0.063, 1.0)
    grid_rgba = (0.10, 0.13, 0.25, 0.18)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(pane_rgba)
        axis._axinfo["grid"]["color"]     = grid_rgba
        axis._axinfo["grid"]["linewidth"] = 0.30
        axis.set_ticklabels([])
        axis.set_ticks([])
        axis.line.set_color((0, 0, 0, 0))

    # ── Moon (Moon-centered, so place at origin) ─────────────────────────────
    # Keep the Moon above the trajectory artists so the lunar disk is a hard
    # visual occluder in the slide render. The far/near masks still create the
    # depth dimming; the surface itself prevents bright lines crossing the Moon.
    MOON_R_KM = 1737.4
    MOON_R_DRAW = 4_000.0   # exaggerated for legibility at cislunar zoom
    MOON_Z = 30
    _MOON_TEX = REPO_ROOT_PATH / "results" / "seeds" / "moon_texture.jpg"
    if _MOON_TEX.exists():
        _draw_textured_sphere(ax, np.zeros(3), MOON_R_DRAW, _MOON_TEX,
                              n=80, alpha=1.0, rotate_lon_deg=180.0,
                              zorder=MOON_Z)
    else:
        _draw_sphere(ax, np.zeros(3), MOON_R_DRAW, _MOON_C, alpha=0.92,
                     zorder=MOON_Z)

    # ── SPICE truth arc — faint dotted reference (CR3BP vs SPICE drift) ────
    spice_near, = ax.plot([], [], [], color=_WHITE, lw=0.9, alpha=0.32,
                           ls=(0, (1, 3)), zorder=6)
    spice_far,  = ax.plot([], [], [], color=_WHITE, lw=0.9, alpha=0.10,
                           ls=(0, (1, 3)), zorder=2)

    # ── uncorrected (no-burn) miss arc — persistent dashed ghost ────────────
    # Two layers split by occlusion so the back of the loop dims when it
    # tucks behind the lunar limb.  This is the path the spacecraft would
    # have flown WITHOUT the midcourse correction.
    miss_near, = ax.plot([], [], [], color=_VIOLET, lw=1.0, alpha=0.62,
                          ls=(0, (5, 3)), zorder=6)
    miss_far,  = ax.plot([], [], [], color=_VIOLET, lw=1.0, alpha=0.14,
                          ls=(0, (5, 3)), zorder=2)

    # ── corrected trail — three phase-coloured segments share one path ──────
    # Phase 1 (insertion)   → amber   · Phase 2 (tracking) → cyan
    # Phase 3 (post-burn)   → green
    # Each phase uses an independent (near, far, glow) triple so we can
    # both color-by-phase AND keep proper occlusion against the Moon.
    PHASE_COLORS = (_AMBER, _CYAN, _GREEN)

    glow_near_p, line_near_p = [], []
    glow_far_p,  line_far_p  = [], []
    for c in PHASE_COLORS:
        # Near/far split controls alpha and depth cues. The Moon surface is
        # drawn above both layers so lines never visually cross the lunar disk.
        gn, = ax.plot([], [], [], color=c, lw=8, alpha=0.10, zorder=7)
        ln, = ax.plot([], [], [], color=c, lw=2.4, alpha=0.95, zorder=8)
        gf, = ax.plot([], [], [], color=c, lw=5, alpha=0.04, zorder=1)
        lf, = ax.plot([], [], [], color=c, lw=1.6, alpha=0.30, zorder=2)
        glow_near_p.append(gn); line_near_p.append(ln)
        glow_far_p.append(gf);  line_far_p.append(lf)

    sc_dot = ax.scatter([], [], [], s=110, color=_CYAN, edgecolors=_WHITE,
                        linewidths=0.7, zorder=9, depthshade=False)

    # ── burn event glyph + ΔV vector (hidden until burn) ─────────────────────
    burn_sc = ax.scatter([], [], [], s=180, color=_RED, marker="D",
                         edgecolors=_WHITE, linewidths=0.9, zorder=9,
                         depthshade=False)
    dv_line, = ax.plot([], [], [], color=_RED, lw=2.0, alpha=0.0, zorder=9)
    dv_tip   = ax.scatter([], [], [], s=60, color=_RED, marker="^",
                          edgecolors=_WHITE, linewidths=0.5, zorder=10,
                          depthshade=False, alpha=0.0)

    # ── HUD: persistent phase chips, top-left, monospace ────────────────────
    # Three chips stay on screen the whole time — the active one brightens,
    # the others sit at low alpha.  Color matches the trail color of each
    # phase so the path itself is keyed to the chips.
    chip_dim, chip_bright = 0.28, 1.00
    chip_y0, chip_dy = 0.952, 0.030
    PHASE_CHIPS = [
        ("01 · ORANGE INSERTION", _AMBER),
        ("02 · CYAN TRACKING",    _CYAN),
        ("03 · GREEN POST-BURN",  _GREEN),
    ]
    chips = []
    for i, (txt, color) in enumerate(PHASE_CHIPS):
        chip = fig.text(0.035, chip_y0 - i * chip_dy, txt,
                        color=color, alpha=chip_dim, fontsize=10.0,
                        fontweight="bold", family="monospace")
        chips.append(chip)

    hud_clock = fig.text(0.035, chip_y0 - 3 * chip_dy - 0.005,
                         "", color=_TEXT, fontsize=9.0,
                         family="monospace")
    hud_sub   = fig.text(0.035, chip_y0 - 4 * chip_dy + 0.003,
                         "", color=_DIM, fontsize=8.0,
                         family="monospace")

    # ── Miss-distance overlay (top-right) — the burn's quantitative payoff ──
    fig.text(0.965, 0.952, "PREDICTED TERMINAL MISS",
             color=_DIM, fontsize=8.5, ha="right", family="monospace",
             fontweight="bold")
    miss_unc_txt = fig.text(0.965, 0.922, "no-burn:    ---  km",
                             color=_VIOLET, fontsize=9.5, ha="right",
                             family="monospace")
    miss_cor_txt = fig.text(0.965, 0.898, "corrected:   ---  km",
                             color=_GREEN, fontsize=9.5, ha="right",
                             family="monospace")
    dv_txt       = fig.text(0.965, 0.870,
                             f"midcourse ΔV = {targ['dv_ms']:5.1f} m/s",
                             color=_RED, fontsize=9.0, ha="right",
                             family="monospace", alpha=0.0)

    def _fmt_miss_line(label: str, value_km: float, *, approx_zero=False) -> str:
        if approx_zero and abs(value_km) < 0.5:
            value = "≈ 0"
        else:
            value = f"{value_km:6,.0f}"
        return f"{label:<11} {value:>6}  km"

    # SPICE credibility cue — small, dim, bottom-left.
    fig.text(0.035, 0.060,
             "EARTH–MOON ROTATING · SPICE-SEEDED IC · DE442s",
             color=_DIM, fontsize=7.5, family="monospace")
    fig.text(0.035, 0.040,
             "L2 SOUTHERN HALO · INSERTION → TRACKING → MIDCOURSE",
             color=_DIM, fontsize=7.5, family="monospace")
    # Explicit line-style legend (right footer).
    legend_specs = [
        (0.088, _WHITE,  0.36, (0, (1, 3)), "gray dotted = SPICE truth"),
        (0.068, _VIOLET, 0.70, (0, (5, 3)), "violet dashed = no-burn continuation"),
        (0.048, _CYAN,   0.95, "-",         "cyan/green solid = flown / corrected"),
    ]
    for y, color, alpha, ls, label in legend_specs:
        fig.add_artist(Line2D([0.675, 0.755], [y, y],
                              transform=fig.transFigure, color=color,
                              alpha=alpha, lw=1.5, ls=ls))
        fig.text(0.765, y - 0.006, label, color=_TEXT, alpha=0.82,
                 fontsize=7.2, ha="left", family="monospace")

    # Off-screen Earth direction cue.
    fig.text(0.035, 0.085, "← Earth (off-frame, 380,000 km)",
             color=_EARTH_C, fontsize=7.5, family="monospace", alpha=0.80)

    # ── time progress bar ─────────────────────────────────────────────────────
    ax_pb = fig.add_axes([0.030, 0.018, 0.940, 0.012])
    ax_pb.set_facecolor((0.03, 0.05, 0.10, 1.0))
    for sp in ax_pb.spines.values():
        sp.set_edgecolor(_BORDER); sp.set_linewidth(0.6)
    ax_pb.set_xlim(0, t_total); ax_pb.set_ylim(0, 1)
    ax_pb.set_xticks([]); ax_pb.set_yticks([])

    # phase tick marks on the bar
    for tx, lbl in [(t_insertion_end, "INSERTION"),
                    (t_burn,          "MIDCOURSE")]:
        ax_pb.axvline(tx, color=_DIM, lw=0.6, alpha=0.55)
    from matplotlib.patches import Rectangle
    pb_fill = Rectangle((0, 0), 0, 1, color=_CYAN, alpha=0.55, zorder=2)
    ax_pb.add_patch(pb_fill)

    # ── camera framing ────────────────────────────────────────────────────────
    # Bound the scene around all three trajectories + the Moon sphere.  We
    # then enforce an isotropic cube so the lunar surface stays round and
    # geometric occlusion lines up with what's drawn on screen.
    pts = np.vstack([X_dense, X_unc_dense, X_spice_dense,
                     np.array([[ MOON_R_DRAW, 0, 0],
                               [-MOON_R_DRAW, 0, 0],
                               [0,  MOON_R_DRAW, 0],
                               [0, -MOON_R_DRAW, 0],
                               [0, 0,  MOON_R_DRAW],
                               [0, 0, -MOON_R_DRAW]])])
    pad = 0.08 * (pts.max(axis=0) - pts.min(axis=0))
    lo  = pts.min(axis=0) - pad
    hi  = pts.max(axis=0) + pad
    base_center = 0.5 * (lo + hi)
    raw_half    = 0.5 * (hi - lo)
    # Isotropic cube — equal half-extent on each axis so the Moon stays
    # spherical and box_aspect can be (1,1,1).  Horizontal extents driven
    # by the trajectory; vertical extent inherits the same scale.
    cube_half = float(np.max(raw_half))
    base_half = np.array([cube_half, cube_half, cube_half])
    half_max  = cube_half                  # for ΔV arrow length scaling
    box_aspect = (1.0, 1.0, 1.0)

    # View angle: the L2 Southern Halo's major axis runs nearly along +z
    # (~70 000 km, vs ~14 000 km in-plane minor and ~2 600 km out-of-plane).
    # Looking near-horizontally puts that whole +z extent on screen-vertical
    # and the loop reads as a tall ellipse.  A high elevation rotates the
    # +z axis away from screen-up into screen-depth, so the orbit projects
    # as a tilted-loop arc with visible 3D character.
    azim_base = 54.0
    azim_span = 36.0
    elev_base = 42.0
    elev_amp  = 6.0

    preroll_frames = int(round(0.25 * fps))   # brief establishing beat only

    # Occlusion bookkeeping — Moon at origin, slight inflation to soften
    # the silhouette edge so points right on the limb don't pop in/out.
    OCC_R = MOON_R_DRAW * 1.08
    cam_state = {"elev": elev_base, "azim": azim_base}

    def _split_xyz(X, mask_occluded):
        """Return ((x_near, y_near, z_near), (x_far, y_far, z_far)) with
        NaN at the masked-out points so each line draws only its half."""
        Xn = X.copy(); Xf = X.copy()
        Xn[mask_occluded] = np.nan
        Xf[~mask_occluded] = np.nan
        return ((Xn[:, 0], Xn[:, 1], Xn[:, 2]),
                (Xf[:, 0], Xf[:, 1], Xf[:, 2]))

    moon_origin = np.zeros(3)

    def _frame_camera(t_n):
        # Slow azimuth sweep across the arc (parallax cue, not vertigo).
        tau = t_n / t_total
        az = azim_base + azim_span * _smoothstep(tau, 0.0, 1.0)
        el = elev_base + elev_amp * np.sin(np.pi * tau)

        # Brief push-in around the burn (~15% tighter through the event)
        push = _smoothstep(t_n, t_burn - 0.7, t_burn) \
               - _smoothstep(t_n, t_burn + 0.2, t_burn + 1.0)
        zoom = 1.0 - 0.15 * push

        cx, cy, cz = base_center
        hx, hy, hz = base_half * zoom
        ax.set_xlim(cx - hx, cx + hx)
        ax.set_ylim(cy - hy, cy + hy)
        ax.set_zlim(cz - hz, cz + hz)
        ax.set_box_aspect(box_aspect)
        ax.view_init(elev=el, azim=az)
        cam_state["elev"] = el
        cam_state["azim"] = az

    # Pre-compute phase-segment masks on the dense grid.  The corrected
    # trail is sliced into three windows; each phase artist only ever
    # draws within its window, so colors don't bleed into one another.
    phase_t_lo = (0.0, t_insertion_end, t_burn)
    phase_t_hi = (t_insertion_end, t_burn, t_total + 1.0)
    phase_in_window = [
        (t_dense >= lo) & (t_dense <= hi)
        for lo, hi in zip(phase_t_lo, phase_t_hi)
    ]

    def init_fc():
        for ln in (*line_near_p, *line_far_p, *glow_near_p, *glow_far_p,
                   miss_near, miss_far, spice_near, spice_far):
            ln.set_data_3d([], [], [])
        sc_dot._offsets3d = ([], [], [])
        burn_sc._offsets3d = ([], [], [])
        dv_line.set_data_3d([], [], [])
        dv_line.set_alpha(0.0)
        dv_tip._offsets3d = ([], [], [])
        dv_tip.set_alpha(0.0)
        for chip in chips:
            chip.set_alpha(chip_dim)
        hud_clock.set_text(""); hud_sub.set_text("")
        miss_unc_txt.set_text("no-burn:    ---  km")
        miss_cor_txt.set_text("corrected:   ---  km")
        dv_txt.set_alpha(0.0)
        pb_fill.set_width(0)
        return (*line_near_p, *line_far_p, *glow_near_p, *glow_far_p,
                miss_near, miss_far, spice_near, spice_far,
                sc_dot, burn_sc, dv_line, dv_tip)

    def update_fc(frame):
        eff = max(0, frame - preroll_frames)
        i  = min(eff * speed, N - 1)
        t_n = float(t_dense[i])

        # Update camera first so view_dir reflects this frame's pose.
        _frame_camera(t_n)
        view_dir = _camera_dir(cam_state["elev"], cam_state["azim"])

        # Occlusion masks: True where a point sits behind the Moon sphere.
        miss_mask  = _occlusion_mask(X_unc_dense,   view_dir, moon_origin, OCC_R)
        cor_mask   = _occlusion_mask(X_dense,       view_dir, moon_origin, OCC_R)
        spice_mask = _occlusion_mask(X_spice_dense, view_dir, moon_origin, OCC_R)

        # Persistent SPICE-truth ghost — model-error reference (full arc).
        (sn, sf) = _split_xyz(X_spice_dense, spice_mask)
        spice_near.set_data_3d(*sn)
        spice_far.set_data_3d(*sf)

        # Persistent miss ghost — uncorrected (no-burn) arc (full arc).
        (mn, mf) = _split_xyz(X_unc_dense, miss_mask)
        miss_near.set_data_3d(*mn)
        miss_far.set_data_3d(*mf)

        # Phase-coloured corrected trail — three artists, each masked to
        # its own time window AND truncated at the current frame.
        seen_mask = np.zeros(N, dtype=bool)
        seen_mask[: i + 1] = True
        for p in range(3):
            keep = phase_in_window[p] & seen_mask
            X_p = np.where(keep[:, None], X_dense, np.nan)
            occ_p = cor_mask | ~keep      # treat outside-window as occluded
            (sn, sf) = _split_xyz(X_p, occ_p)
            glow_near_p[p].set_data_3d(*sn); line_near_p[p].set_data_3d(*sn)
            glow_far_p[p].set_data_3d(*sf);  line_far_p[p].set_data_3d(*sf)

        # Spacecraft dot — colour matches the active phase.
        cur = X_dense[i]
        if t_n < t_insertion_end:
            sc_color = _AMBER
        elif t_n < t_burn:
            sc_color = _CYAN
        else:
            sc_color = _GREEN
        sc_dot.set_color(sc_color)
        if cor_mask[i]:
            sc_dot._offsets3d = ([], [], [])
        else:
            sc_dot._offsets3d = ([cur[0]], [cur[1]], [cur[2]])

        # Burn event: glyph at burn point, pulses around the burn moment,
        # ΔV arrow fades in then out.
        if t_n >= t_burn - 0.30:
            burn_sc._offsets3d = ([burn_pos[0]], [burn_pos[1]], [burn_pos[2]])
            # 4-Hz pulse for ±0.6 days around the burn
            dt_burn = t_n - t_burn
            if abs(dt_burn) < 0.7:
                pulse = 1.0 + 0.55 * np.sin(2 * np.pi * dt_burn / 0.30)
            else:
                pulse = 1.0
            burn_sc.set_sizes([180 * pulse * pulse])
            # ΔV arrow: fades in over 0.25 days, fades out 0.85 days after
            f_in  = _smoothstep(t_n, t_burn - 0.05, t_burn + 0.20)
            f_out = _smoothstep(t_n, t_burn + 0.50, t_burn + 1.10)
            dv_alpha = 0.95 * f_in * (1.0 - f_out)
            arrow_len = half_max * 0.12
            tip = burn_pos + v_hat * arrow_len
            dv_line.set_data_3d([burn_pos[0], tip[0]],
                                [burn_pos[1], tip[1]],
                                [burn_pos[2], tip[2]])
            dv_line.set_alpha(dv_alpha)
            dv_tip._offsets3d = ([tip[0]], [tip[1]], [tip[2]])
            dv_tip.set_alpha(dv_alpha)
            dv_txt.set_alpha(min(1.0, dv_alpha + 0.10))
        else:
            burn_sc._offsets3d = ([], [], [])
            burn_sc.set_sizes([180])
            dv_line.set_alpha(0.0)
            dv_tip.set_alpha(0.0)
            dv_txt.set_alpha(0.0)

        # Persistent phase chips — brighten the active one.
        if t_n < t_insertion_end:
            active = 0
        elif t_n < t_burn:
            active = 1
        else:
            active = 2
        for k, chip in enumerate(chips):
            chip.set_alpha(chip_bright if k == active else chip_dim)

        hud_clock.set_text(f"t = {t_n:5.2f} / {t_total:.2f} days")
        hud_sub.set_text(
            f"Moon range = {range_dense[i]:6,.0f} km    "
            f"perilune ≈ {arc['moon_range'].min():,.0f} km"
        )

        # Live miss-distance overlay — the burn's quantitative payoff.
        miss_unc_txt.set_text(_fmt_miss_line("no-burn:", miss_unc_d[i]))
        miss_cor_txt.set_text(
            _fmt_miss_line("corrected:", miss_cor_d[i], approx_zero=True))

        # Time progress bar
        pb_fill.set_width(t_n)

        return (*line_near_p, *line_far_p, *glow_near_p, *glow_far_p,
                miss_near, miss_far, spice_near, spice_far,
                sc_dot, burn_sc, dv_line, dv_tip,
                *chips, hud_clock, hud_sub, miss_unc_txt, miss_cor_txt,
                dv_txt, pb_fill)

    n_frames = int(np.ceil(N / speed)) + preroll_frames + int(round(0.8 * fps))
    ani = FuncAnimation(fig, update_fc, frames=n_frames, init_func=init_fc,
                        blit=False, interval=1000 // fps)
    _try_save(ani, OUT_DIR / "anim_02_targeting_follow_cam.mp4", fps,
              bitrate=6_000)
    plt.close(fig)


# ── Phase 3: Active vs Fixed pointing comparison ─────────────────────────────
def _load_active_fixed_npz():
    """Load the matched active/fixed runs produced by 07_active_tracking.py.

    Returns a dict with both cases keyed under 'active' and 'fixed', each
    holding the time grid, truth state, EKF estimate, Moon position,
    visibility flags, NIS, and pre-computed RMS metrics. The two runs
    share an arc (6 TU, dt=0.02, 301 steps); only the camera pointing
    policy differs — active slews boresight to the estimated Moon LOS,
    fixed holds a constant body-frame boresight.
    """
    base = REPO_ROOT_PATH / "results" / "active_tracking"
    paths = {
        "active": base / "07_active_tracking_active.npz",
        "fixed":  base / "07_active_tracking_fixed.npz",
    }
    out: dict = {}
    for case, p in paths.items():
        if not p.exists():
            return None
        d = np.load(p, allow_pickle=True)
        out[case] = dict(
            t=np.asarray(d["t_hist"], dtype=float),
            r_sc_true=np.asarray(d["r_sc_true_hist"], dtype=float),
            r_body=np.asarray(d["r_body_true_hist"], dtype=float),
            xhat=np.asarray(d["xhat_hist"], dtype=float),
            visible=np.asarray(d["visible_hist"], dtype=bool),
            update_used=np.asarray(d["update_used_hist"], dtype=bool),
            nis=np.asarray(d["nis_hist"], dtype=float),
            los_cmd=np.asarray(d["los_cmd_hist"], dtype=float),
            los_true=np.asarray(d["los_true_hist"], dtype=float),
            off_boresight=np.asarray(d["off_boresight_hist"], dtype=float),
            rms_pos=float(d["rms_position_error"]),
            visibility_frac=float(d["visibility_fraction"]),
        )
    return out


def animate_phase3(fps: int = 30, sim_speed: float = 0.18) -> None:
    """Active-Pointing vs Fixed-Camera comparison animation.

    Same filter, same trajectory, same measurement model — only the
    camera pointing policy differs. The active case slews the boresight
    toward the estimated Moon LOS each step (subject to slew-rate
    limits). The fixed case keeps a body-frame-fixed boresight; the
    Moon walks out of FOV around t≈1.1 and the filter dead-reckons from
    there. Result: ~1300× RMS position-error ratio across the arc.

    Layout:
      • Left 3D panel: Moon-centered, isotropic. Truth trail + both
        estimate trails + LOS rays drawn only when the Moon is inside
        each camera's FOV. Visibility difference reads as glowing rays
        for active vs intermittent rays for fixed.
      • Right column (3 stacked):
          – Visibility timeline (binary on/off bars per case)
          – Position error norm, log-y (the four-orders-of-magnitude split)
          – NIS scatter against the χ²(2) gate
    """
    print("Phase 3 · loading active/fixed pointing comparison …")
    data = _load_active_fixed_npz()
    if data is None:
        print("  active/fixed npz not found — falling back to legacy _data3()")
        return _animate_phase3_legacy(fps=fps, sim_speed=sim_speed)
    A = data["active"]; F = data["fixed"]

    # Sanity: the two runs share the same time grid.
    t = A["t"]
    N = len(t)
    t_total = float(t[-1])
    rms_ratio = F["rms_pos"] / max(A["rms_pos"], 1e-12)

    # Densify positions for smoother motion (npz is 301 samples).
    from scipy.interpolate import CubicSpline
    N_dense = 1200
    t_dense = np.linspace(t[0], t[-1], N_dense)

    def _spline_xyz(arr_n3):
        return CubicSpline(t, arr_n3, axis=0)(t_dense)

    truth_dense    = _spline_xyz(A["r_sc_true"])     # both cases share truth
    moon_dense     = _spline_xyz(A["r_body"])
    active_dense   = _spline_xyz(A["xhat"][:, :3])
    fixed_dense    = _spline_xyz(F["xhat"][:, :3])
    # boresight directions densified (renormalized after spline)
    los_cmd_a_dense = _spline_xyz(A["los_cmd"])
    los_cmd_f_dense = _spline_xyz(F["los_cmd"])
    los_cmd_a_dense /= np.linalg.norm(los_cmd_a_dense, axis=1, keepdims=True) + 1e-12
    los_cmd_f_dense /= np.linalg.norm(los_cmd_f_dense, axis=1, keepdims=True) + 1e-12

    # Visibility & NIS stay on the original 301-grid (binary / measurement-tick).
    pos_err_a = np.linalg.norm(A["xhat"][:, :3] - A["r_sc_true"], axis=1)
    pos_err_f = np.linalg.norm(F["xhat"][:, :3] - F["r_sc_true"], axis=1)

    # Map each dense tick to nearest measurement-grid index.
    meas_idx_per_dense = np.clip(
        np.searchsorted(t, t_dense), 0, N - 1
    )

    # ── playback rate ────────────────────────────────────────────────────────
    dt_dense = float(t_dense[1] - t_dense[0])
    sim_tu_per_sec = sim_speed * T_DAY                        # TU per real sec
    speed = max(1, round(sim_tu_per_sec / (fps * dt_dense)))

    # ── figure layout ────────────────────────────────────────────────────────
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D proj)

    # Aspect tuned to the 7.92" × 4.69" slide cell so the mp4 fills the
    # placeholder without distortion (matches PowerPoint Height/Width).
    fig = plt.figure(figsize=(15.84, 9.38), facecolor=_BG)
    # 3D orbit takes the left ~60% of the canvas, with breathing room.
    ax3d = fig.add_axes([0.00, 0.05, 0.62, 0.88], projection="3d")
    # Right-column stack (3 panels). Generous gaps so each panel's title
    # never crashes into the panel above it.
    ax_vis = fig.add_axes([0.66, 0.80, 0.31, 0.10])
    ax_err = fig.add_axes([0.66, 0.46, 0.31, 0.26])
    ax_nis = fig.add_axes([0.66, 0.12, 0.31, 0.26])

    ax3d.set_facecolor(_BG)
    pane_rgba = (0.020, 0.031, 0.063, 1.0)
    grid_rgba = (0.10, 0.13, 0.25, 0.18)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.set_pane_color(pane_rgba)
        axis._axinfo["grid"]["color"]     = grid_rgba
        axis._axinfo["grid"]["linewidth"] = 0.30
        axis.set_ticklabels([])
        axis.set_ticks([])
        axis.line.set_color((0, 0, 0, 0))

    # Moon at its mean synodic position (the body barely moves in synodic
    # over 6 TU; treating it as fixed keeps the camera framing stable).
    moon_pos = moon_dense.mean(axis=0)
    MOON_R = 0.012   # CR3BP-ND visual radius (Moon physical radius is much
                     # smaller; exaggerate for legibility at this zoom).

    _MOON_TEX = REPO_ROOT_PATH / "results" / "seeds" / "moon_texture.jpg"
    if _MOON_TEX.exists():
        _draw_textured_sphere(ax3d, moon_pos, MOON_R, _MOON_TEX,
                              n=72, alpha=1.0, rotate_lon_deg=180.0)
    else:
        _draw_sphere(ax3d, moon_pos, MOON_R, _MOON_C, alpha=0.92)

    # Faint full-arc reference (truth) so geometry reads from the start.
    ax3d.plot(truth_dense[:, 0], truth_dense[:, 1], truth_dense[:, 2],
              color=_DIM, lw=0.8, alpha=0.45, ls=(0, (4, 3)), zorder=2)

    # Animated lines: cumulative trails for truth + both estimates.
    truth_glow,  = ax3d.plot([], [], [], color=_WHITE, lw=6, alpha=0.06, zorder=3)
    truth_line,  = ax3d.plot([], [], [], color=_WHITE, lw=1.5, alpha=0.55, zorder=4)
    active_glow, = ax3d.plot([], [], [], color=_CYAN,  lw=7, alpha=0.10, zorder=4)
    active_line, = ax3d.plot([], [], [], color=_CYAN,  lw=2.4, alpha=0.95, zorder=5)
    fixed_glow,  = ax3d.plot([], [], [], color=_AMBER, lw=7, alpha=0.08, zorder=4)
    fixed_line,  = ax3d.plot([], [], [], color=_AMBER, lw=2.0, alpha=0.85, zorder=5,
                              ls=(0, (5, 2)))

    # Spacecraft + estimate dots
    truth_dot  = ax3d.scatter([], [], [], s=70, color=_WHITE, edgecolors=_BG,
                              linewidths=0.5, zorder=8, depthshade=False)
    active_dot = ax3d.scatter([], [], [], s=110, color=_CYAN, edgecolors=_WHITE,
                              linewidths=0.7, zorder=8, depthshade=False)
    fixed_dot  = ax3d.scatter([], [], [], s=110, color=_AMBER, marker="s",
                              edgecolors=_WHITE, linewidths=0.7, zorder=8,
                              depthshade=False)

    # LOS rays — drawn only when each case's camera SEES the Moon.
    los_active, = ax3d.plot([], [], [], color=_CYAN,  lw=1.6, alpha=0.0,
                              ls=":", zorder=6)
    los_fixed,  = ax3d.plot([], [], [], color=_AMBER, lw=1.6, alpha=0.0,
                              ls=":", zorder=6)

    # Boresight cones: short cyan/amber rays in the commanded direction.
    bore_active, = ax3d.plot([], [], [], color=_CYAN,  lw=2.2, alpha=0.85, zorder=7)
    bore_fixed,  = ax3d.plot([], [], [], color=_AMBER, lw=2.2, alpha=0.85, zorder=7)

    # ── 3D framing (isotropic cube around the Moon + arc) ────────────────────
    pts = np.vstack([truth_dense, moon_pos[None, :] + MOON_R * np.eye(3),
                     moon_pos[None, :] - MOON_R * np.eye(3)])
    pad = 0.08 * (pts.max(axis=0) - pts.min(axis=0) + 0.02)
    lo  = pts.min(axis=0) - pad
    hi  = pts.max(axis=0) + pad
    base_center = 0.5 * (lo + hi)
    half        = float(0.5 * np.max(hi - lo))
    ax3d.set_xlim(base_center[0] - half, base_center[0] + half)
    ax3d.set_ylim(base_center[1] - half, base_center[1] + half)
    ax3d.set_zlim(base_center[2] - half, base_center[2] + half)
    ax3d.set_box_aspect((1.0, 1.0, 1.0))

    # ── Right panels ────────────────────────────────────────────────────────
    # Visibility timeline (binary, two horizontal lanes). xlabel suppressed
    # because the bottom panel (NIS) carries it for all three.
    _dark(ax_vis, title="Visibility — fixed loses the Moon at t ≈ 1.1",
          ylabel="")
    ax_vis.set_yticks([0.25, 0.75])
    ax_vis.set_yticklabels(["FIXED", "ACTIVE"], fontsize=8)
    ax_vis.tick_params(axis="x", labelbottom=False)
    ax_vis.set_xlim(t[0], t[-1])
    ax_vis.set_ylim(0, 1)
    # Pre-computed visibility bands (drawn once)
    for case, lane_y, color in [(F, 0.10, _AMBER), (A, 0.60, _CYAN)]:
        vis = case["visible"].astype(int)
        # Find runs of "visible"
        edges = np.diff(np.concatenate([[0], vis, [0]]))
        starts = np.where(edges == 1)[0]
        ends   = np.where(edges == -1)[0]
        for s, e in zip(starts, ends):
            ax_vis.axvspan(t[s], t[min(e, len(t) - 1)],
                            ymin=lane_y, ymax=lane_y + 0.30,
                            color=color, alpha=0.65, lw=0)
    vis_vline = ax_vis.axvline(t[0], color=_WHITE, lw=1.2, alpha=0.7)

    # Position error log. xlabel suppressed (NIS carries it).
    _dark(ax_err, title="Position error — four orders of magnitude apart",
          ylabel="‖r̂ − r_true‖  [ND, CR3BP]")
    ax_err.tick_params(axis="x", labelbottom=False)
    ax_err.set_yscale("log")
    ax_err.plot(t, pos_err_a + 1e-12, color=_CYAN,  lw=0.7, alpha=0.18)
    ax_err.plot(t, pos_err_f + 1e-12, color=_AMBER, lw=0.7, alpha=0.18)
    ax_err.plot([], [], color=_CYAN,  lw=2, label="active")
    ax_err.plot([], [], color=_AMBER, lw=2, label="fixed")
    ax_err.set_xlim(t[0], t[-1])
    err_lo = max(min(pos_err_a[pos_err_a > 0].min() * 0.3, 1e-7), 1e-9)
    err_hi = max(pos_err_f.max() * 3.0, 10.0)
    ax_err.set_ylim(err_lo, err_hi)
    ax_err.legend(fontsize=8, loc="upper left",
                   facecolor=_PANEL, edgecolor=_BORDER, labelcolor=_TEXT)
    err_vline = ax_err.axvline(t[0], color=_WHITE, lw=1.0, alpha=0.7)
    err_active_dot, = ax_err.plot([], [], "o", color=_CYAN,  ms=6, zorder=6)
    err_fixed_dot,  = ax_err.plot([], [], "s", color=_AMBER, ms=6, zorder=6)
    err_active_trace, = ax_err.plot([], [], color=_CYAN,  lw=2.0, zorder=5)
    err_fixed_trace,  = ax_err.plot([], [], color=_AMBER, lw=2.0, zorder=5,
                                      ls=(0, (5, 2)))

    # NIS w/ χ²(2) gate
    NIS_LO = float(chi2.ppf(0.025, df=2))
    NIS_HI = float(chi2.ppf(0.975, df=2))
    _dark(ax_nis, title="NIS — active stays inside χ²(2) gate",
          xlabel="t  [dimensionless TU]", ylabel="NIS")
    ax_nis.fill_between([t[0], t[-1]], NIS_LO, NIS_HI,
                         color=_GREEN, alpha=0.14, zorder=1,
                         label=f"χ²(2) 95% gate [{NIS_LO:.2f}, {NIS_HI:.2f}]")
    ax_nis.axhline(2.0, color=_GREEN, lw=0.8, ls="--", alpha=0.55)
    nis_a_finite = np.isfinite(A["nis"])
    nis_f_finite = np.isfinite(F["nis"])
    ax_nis.set_xlim(t[0], t[-1])
    nis_max = max(
        float(A["nis"][nis_a_finite].max()) if nis_a_finite.any() else 10.0,
        float(F["nis"][nis_f_finite].max()) if nis_f_finite.any() else 10.0,
    )
    ax_nis.set_ylim(-0.5, min(nis_max * 1.15, 14.0))
    ax_nis.legend(fontsize=8, loc="upper right",
                   facecolor=_PANEL, edgecolor=_BORDER, labelcolor=_TEXT)
    nis_vline = ax_nis.axvline(t[0], color=_WHITE, lw=1.0, alpha=0.7)
    nis_active_pts, = ax_nis.plot([], [], "o", color=_CYAN,  ms=4.0, alpha=0.85, zorder=4)
    nis_fixed_pts,  = ax_nis.plot([], [], "s", color=_AMBER, ms=4.0, alpha=0.85, zorder=4)
    nis_a_t, nis_a_v = [], []
    nis_f_t, nis_f_v = [], []

    # ── HUD ──────────────────────────────────────────────────────────────────
    hud_title = fig.text(0.005, 0.965,
                          "Active-Pointing  vs  Fixed-Camera   ·   Same filter, same trajectory",
                          color=_TEXT, fontsize=14, fontweight="bold",
                          family="monospace")
    hud_clock = fig.text(0.005, 0.935, "", color=_DIM, fontsize=11,
                          family="monospace")
    hud_rms   = fig.text(0.005, 0.040,
                          f"RMS position-error ratio  fixed : active  =  {rms_ratio:,.0f}×",
                          color=_GREEN, fontsize=11.5, fontweight="bold",
                          family="monospace")
    hud_active = fig.text(0.005, 0.080,
                          f"ACTIVE  visibility = {A['visibility_frac']*100:5.1f}%   "
                          f"RMS = {A['rms_pos']:.2e} ND",
                          color=_CYAN, fontsize=10, family="monospace")
    hud_fixed  = fig.text(0.005, 0.060,
                          f"FIXED   visibility = {F['visibility_frac']*100:5.1f}%   "
                          f"RMS = {F['rms_pos']:.2e} ND",
                          color=_AMBER, fontsize=10, family="monospace")

    # Camera rotation parameters — kept gentle so the orbit reads as the
    # primary motion, not the camera. Span is the total azimuth sweep over
    # the full arc.
    azim_base = -55.0
    azim_span = 28.0
    elev_base = 22.0
    elev_amp  = 2.5
    preroll_frames = int(round(1.0 * fps))

    last_meas_idx = [-1]

    def init3():
        for ln in (truth_glow, truth_line, active_glow, active_line,
                   fixed_glow, fixed_line, los_active, los_fixed,
                   bore_active, bore_fixed, err_active_trace, err_fixed_trace):
            ln.set_data([], [])
            try:
                ln.set_3d_properties([])
            except AttributeError:
                pass
        for sc in (truth_dot, active_dot, fixed_dot):
            sc._offsets3d = ([], [], [])
        err_active_dot.set_data([], [])
        err_fixed_dot.set_data([], [])
        nis_active_pts.set_data([], [])
        nis_fixed_pts.set_data([], [])
        vis_vline.set_xdata([t[0]])
        err_vline.set_xdata([t[0]])
        nis_vline.set_xdata([t[0]])
        return (truth_glow, truth_line, active_glow, active_line,
                fixed_glow, fixed_line)

    def update3(frame):
        eff = max(0, frame - preroll_frames)
        i_dense = min(eff * speed, N_dense - 1)
        t_n = float(t_dense[i_dense])
        i_meas = int(meas_idx_per_dense[i_dense])

        # Trails (cumulative)
        seg_t = truth_dense[:i_dense + 1]
        truth_glow.set_data(seg_t[:, 0], seg_t[:, 1])
        truth_glow.set_3d_properties(seg_t[:, 2])
        truth_line.set_data(seg_t[:, 0], seg_t[:, 1])
        truth_line.set_3d_properties(seg_t[:, 2])

        seg_a = active_dense[:i_dense + 1]
        active_glow.set_data(seg_a[:, 0], seg_a[:, 1])
        active_glow.set_3d_properties(seg_a[:, 2])
        active_line.set_data(seg_a[:, 0], seg_a[:, 1])
        active_line.set_3d_properties(seg_a[:, 2])

        seg_f = fixed_dense[:i_dense + 1]
        fixed_glow.set_data(seg_f[:, 0], seg_f[:, 1])
        fixed_glow.set_3d_properties(seg_f[:, 2])
        fixed_line.set_data(seg_f[:, 0], seg_f[:, 1])
        fixed_line.set_3d_properties(seg_f[:, 2])

        # Markers
        cur_t = truth_dense[i_dense]
        cur_a = active_dense[i_dense]
        cur_f = fixed_dense[i_dense]
        truth_dot._offsets3d  = ([cur_t[0]], [cur_t[1]], [cur_t[2]])
        active_dot._offsets3d = ([cur_a[0]], [cur_a[1]], [cur_a[2]])
        fixed_dot._offsets3d  = ([cur_f[0]], [cur_f[1]], [cur_f[2]])

        # LOS rays — alpha tied to visibility flags from the measurement grid
        vis_a = bool(A["visible"][i_meas])
        vis_f = bool(F["visible"][i_meas])
        if vis_a:
            los_active.set_data([cur_t[0], moon_pos[0]],
                                 [cur_t[1], moon_pos[1]])
            los_active.set_3d_properties([cur_t[2], moon_pos[2]])
            los_active.set_alpha(0.75)
        else:
            los_active.set_alpha(0.0)
        if vis_f:
            los_fixed.set_data([cur_t[0], moon_pos[0]],
                                [cur_t[1], moon_pos[1]])
            los_fixed.set_3d_properties([cur_t[2], moon_pos[2]])
            los_fixed.set_alpha(0.65)
        else:
            los_fixed.set_alpha(0.0)

        # Boresight glyphs — short rays from spacecraft along commanded LOS
        bore_len = half * 0.10
        bore_a_tip = cur_t + los_cmd_a_dense[i_dense] * bore_len
        bore_f_tip = cur_t + los_cmd_f_dense[i_dense] * bore_len
        bore_active.set_data([cur_t[0], bore_a_tip[0]],
                              [cur_t[1], bore_a_tip[1]])
        bore_active.set_3d_properties([cur_t[2], bore_a_tip[2]])
        bore_fixed.set_data([cur_t[0], bore_f_tip[0]],
                             [cur_t[1], bore_f_tip[1]])
        bore_fixed.set_3d_properties([cur_t[2], bore_f_tip[2]])

        # Camera orbit
        tau = t_n / t_total
        ax3d.view_init(
            elev=elev_base + elev_amp * np.sin(np.pi * tau),
            azim=azim_base + azim_span * tau,
        )

        # Right-panel updates — sweep the time cursor + draw growing traces
        vis_vline.set_xdata([t_n])
        err_vline.set_xdata([t_n])
        nis_vline.set_xdata([t_n])

        # Error traces up to current measurement index (sparse grid is fine)
        err_active_trace.set_data(t[:i_meas + 1], pos_err_a[:i_meas + 1] + 1e-12)
        err_fixed_trace.set_data(t[:i_meas + 1],  pos_err_f[:i_meas + 1] + 1e-12)
        err_active_dot.set_data([t[i_meas]], [pos_err_a[i_meas] + 1e-12])
        err_fixed_dot.set_data([t[i_meas]],  [pos_err_f[i_meas] + 1e-12])

        # Accumulate NIS scatter only on freshly-encountered measurement ticks
        if i_meas > last_meas_idx[0]:
            for k in range(last_meas_idx[0] + 1, i_meas + 1):
                if np.isfinite(A["nis"][k]):
                    nis_a_t.append(t[k]); nis_a_v.append(float(A["nis"][k]))
                if np.isfinite(F["nis"][k]):
                    nis_f_t.append(t[k]); nis_f_v.append(float(F["nis"][k]))
            last_meas_idx[0] = i_meas
        nis_active_pts.set_data(nis_a_t, nis_a_v)
        nis_fixed_pts.set_data(nis_f_t, nis_f_v)

        # HUD
        hud_clock.set_text(
            f"t = {t_n:5.2f} / {t_total:.2f} TU    "
            f"{301} steps · σ_px = 1 px    "
            f"|  active vis: {'●' if vis_a else '○'}    "
            f"fixed vis: {'●' if vis_f else '○'}"
        )

        return (truth_glow, truth_line, active_glow, active_line,
                fixed_glow, fixed_line, los_active, los_fixed,
                truth_dot, active_dot, fixed_dot,
                err_active_trace, err_fixed_trace,
                err_active_dot, err_fixed_dot,
                nis_active_pts, nis_fixed_pts,
                vis_vline, err_vline, nis_vline,
                hud_clock, bore_active, bore_fixed)

    n_frames = int(np.ceil(N_dense / speed)) + preroll_frames + int(round(0.6 * fps))
    ani = FuncAnimation(fig, update3, frames=n_frames, init_func=init3,
                        blit=False, interval=1000 // fps)
    _try_save(ani, OUT_DIR / "anim_03_bearings_v2.mp4", fps)
    plt.close(fig)


def _animate_phase3_legacy(fps: int = 30, sim_speed: float = 0.77) -> None:
    """Legacy bearing-only EKF animation kept as a fallback for when the
    07_active_tracking npz files aren't on disk. 3D orbit + position-error
    log + NIS scatter; single-case version superseded by the active/fixed
    comparison in animate_phase3."""
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
    parser.add_argument("--sim-speed",  type=float, default=None,
                        help="Playback speed; uses a phase-specific default "
                             "when omitted")
    args = parser.parse_args()

    phase_kwargs = {"fps": args.fps}
    if args.sim_speed is not None:
        phase_kwargs["sim_speed"] = args.sim_speed

    if args.phase in (None, "2"):
        animate_phase2(**phase_kwargs)
    if args.phase == "2f":
        animate_phase2_follow_cam(**phase_kwargs)
    if args.phase in (None, "3"):
        animate_phase3(**phase_kwargs)
