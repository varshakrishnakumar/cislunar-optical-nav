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

OUT_DIR = Path("results/plots")
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
    rng    = np.random.default_rng(0)
    r_body = np.asarray(model.primary2, dtype=float)
    t0, tf, dt = 0.0, 6.0, 0.02
    t_meas = np.arange(t0, tf + 1e-12, dt)

    x_true0 = x0.copy()
    x_true0[:3] += [2e-4, -1e-4, 0.]
    x_true0[3:]  += [0., 2e-3, 0.]

    res = propagate(model.eom, (t0, tf), x_true0, t_eval=t_meas, rtol=1e-11, atol=1e-13)
    X_true = res.x

    sig = 2e-4
    U_meas = np.array([
        _ang_noise(los_unit(r_body, X_true[k, :3])[0], sig, rng)
        for k in range(len(t_meas))
    ])

    x = x0.copy()
    P = np.diag([1e-6] * 3 + [1e-8] * 3).astype(float)
    N = len(t_meas)
    X_hat  = np.zeros((N, 6))
    nis    = np.full(N, np.nan)
    P_diag = np.zeros((N, 6))
    X_hat[0] = x; P_diag[0] = np.diag(P)
    t_prev = t_meas[0]

    for k in range(1, N):
        tk = float(t_meas[k])
        x, P, _ = ekf_propagate_cr3bp_stm(mu=mu, x=x, P=P, t0=t_prev, t1=tk, q_acc=1e-12)
        upd = bearing_update_tangent(x, P, U_meas[k], r_body, sig)
        if upd.accepted:
            x, P = upd.x_upd, upd.P_upd
        nis[k]    = upd.nis
        X_hat[k]  = x
        P_diag[k] = np.diag(P)
        t_prev    = tk

    pos_err = np.linalg.norm(X_hat[:, :3] - X_true[:, :3], axis=1) * L_KM   # km
    sig3    = 3.0 * np.sqrt(np.abs(P_diag[:, 0])) * L_KM                    # km

    return dict(
        t=t_meas * T_DAY,
        X_true=X_true[:, :2] * L_KM,
        X_hat=X_hat[:, :2] * L_KM,
        X_true_raw=X_true,
        pos_err=pos_err, sig3=sig3, nis=nis,
        r_body=r_body,
        p2=model.primary2[:2] * L_KM,
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

    fig = plt.figure(figsize=(17, 8.5), facecolor=_BG)
    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[3, 2], wspace=0.22,
        left=0.04, right=0.97, bottom=0.09, top=0.91,
    )
    ax = fig.add_subplot(gs[0], projection="3d")
    ax_mis = fig.add_subplot(gs[1])
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

    # Earth & Moon (exaggerated radii for legibility)
    _draw_sphere(ax, d["p1"], 9_000.0, _EARTH_C, alpha=0.95)
    _draw_sphere(ax, d["p2"], 5_000.0, _MOON_C,  alpha=0.90)

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
    hud_title = fig.text(0.025, 0.955, "", color=_TEXT, fontsize=13,
                         fontweight="bold", family="monospace")
    hud_sub   = fig.text(0.025, 0.922, "", color=_DIM,  fontsize=9.5,
                         family="monospace")
    hud_dv    = fig.text(0.025, 0.060, "", color=_RED,  fontsize=10,
                         fontweight="bold", family="monospace")
    hud_foot  = fig.text(0.025, 0.035, "Follow-cam · orbital sweep tightens at burn",
                         color=_DIM, fontsize=8, family="monospace")

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
    azim0      = 38.0
    azim_rate  = 0.28               # deg per frame → ~85° over 300 frames
    elev_start = 32.0
    elev_end   = 16.0
    D_floor    = 22_000.0           # km; hard lower bound on zoom
    D_ceiling  = 120_000.0
    smooth_a   = 0.88               # EMA retention for center / D

    cam_state = {"D": None, "center": None}

    burn_fired = [False]
    burn_xyz   = [None]

    def _apply_view(frame_i, frac):
        elev = elev_start * (1.0 - frac) + elev_end * frac
        azim = azim0 + azim_rate * frame_i
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
        i  = min(frame * speed, N - 1)
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

        # Target camera center: lerp toward spacecraft as burn fires
        target_center = (1.0 - 0.6 * frac) * mid + (0.6 * frac) * sc
        # Target half-width: auto-fit to sc↔target distance, tightened by frac
        target_D = max(D_floor, dist * (1.55 - 0.55 * frac) + 9_000.0)
        target_D = min(target_D, D_ceiling)

        if cam_state["D"] is None:
            cam_state["D"] = target_D
            cam_state["center"] = target_center.copy()
        else:
            cam_state["D"] = smooth_a * cam_state["D"] + (1 - smooth_a) * target_D
            cam_state["center"] = (smooth_a * cam_state["center"]
                                   + (1 - smooth_a) * target_center)

        D = cam_state["D"]
        cx, cy, cz = cam_state["center"]
        ax.set_xlim(cx - D,       cx + D)
        ax.set_ylim(cy - D,       cy + D)
        ax.set_zlim(cz - D * 0.55, cz + D * 0.55)
        ax.set_box_aspect((1.0, 1.0, 0.55))

        _apply_view(frame, frac)

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

    n_frames = int(np.ceil(N / speed)) + 50
    ani = FuncAnimation(fig, update_fc, frames=n_frames, init_func=init_fc,
                        blit=False, interval=1000 // fps)
    _try_save(ani, OUT_DIR / "anim_02_targeting_follow_cam.mp4", fps)
    plt.close(fig)


# ── Phase 3 animation ──────────────────────────────────────────────────────────
def animate_phase3(fps: int = 30, sim_speed: float = 0.77) -> None:
    print("Phase 3: running EKF …")
    mu, model, x0 = _setup()
    d = _data3(mu, model, x0)
    N  = len(d["t"])
    dt_f = float(d["t"][1] - d["t"][0])
    speed = max(1, round(sim_speed * T_DAY / (fps * dt_f)))

    NIS_LO = float(chi2.ppf(0.025, df=2))   # ≈ 0.05
    NIS_HI = float(chi2.ppf(0.975, df=2))   # ≈ 7.38

    # ── figure layout: orbit | (error / NIS) ──
    fig = plt.figure(figsize=(17, 8.5), facecolor=_BG)
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        width_ratios=[3, 2], height_ratios=[1, 1],
        wspace=0.28, hspace=0.38,
        left=0.07, right=0.97, bottom=0.10, top=0.91,
    )
    ax_orb = fig.add_subplot(gs[:, 0])      # full left column
    ax_err = fig.add_subplot(gs[0, 1])      # top right: position error
    ax_nis = fig.add_subplot(gs[1, 1])      # bottom right: NIS

    # ── orbit axes (zoom-follow) ──
    ZOOM_KM = 22_000
    cx0, cy0 = float(d["X_true"][0, 0]), float(d["X_true"][0, 1])
    ax_orb.set_xlim(cx0 - ZOOM_KM, cx0 + ZOOM_KM)
    ax_orb.set_ylim(cy0 - ZOOM_KM, cy0 + ZOOM_KM)
    _dark(ax_orb,
          title="Bearing-Only EKF — Near-L1 State Estimation",
          xlabel="x  [km from barycenter]",
          ylabel="y  [km from barycenter]")
    _stars(ax_orb, n=280,
           seed=7)  # stars drawn after setting initial limits

    # Moon label (may be off-screen, shown as annotation)
    ax_orb.text(0.98, 0.96, f"Moon at x = {float(d['p2'][0]):,.0f} km →",
                transform=ax_orb.transAxes, color=_MOON_C,
                fontsize=7.5, ha="right", va="top")

    # Animated trails
    gt, = ax_orb.plot([], [], color=_CYAN,  lw=7, alpha=0.09, zorder=3, solid_capstyle="round")
    ct, = ax_orb.plot([], [], color=_CYAN,  lw=2, alpha=0.90, zorder=4, solid_capstyle="round")
    dt  = ax_orb.scatter([], [], s=90, color=_CYAN, zorder=7,
                          edgecolors=_WHITE, linewidths=0.5)

    gh, = ax_orb.plot([], [], color=_AMBER, lw=7, alpha=0.09, zorder=3,
                       solid_capstyle="round", ls="--")
    ch, = ax_orb.plot([], [], color=_AMBER, lw=2, alpha=0.90, zorder=4,
                       solid_capstyle="round", ls="--")
    dh  = ax_orb.scatter([], [], s=70, color=_AMBER, zorder=7,
                          edgecolors=_WHITE, linewidths=0.5, marker="s")

    los_line, = ax_orb.plot([], [], color=_VIOLET, lw=1.0, alpha=0.55,
                              ls=":", zorder=2)

    orb_legend = [
        plt.Line2D([], [], color=_CYAN,  lw=2, label="True trajectory"),
        plt.Line2D([], [], color=_AMBER, lw=2, ls="--", label="EKF estimate"),
        plt.Line2D([], [], color=_VIOLET, lw=1, ls=":", label="LOS to Moon"),
    ]
    ax_orb.legend(handles=orb_legend, fontsize=7.5, loc="upper left",
                  facecolor=_PANEL, edgecolor=_BORDER, labelcolor=_TEXT)

    # NIS indicator (live "MEAS" text)
    meas_txt = ax_orb.text(0.02, 0.96, "", transform=ax_orb.transAxes,
                            color=_GREEN, fontsize=8.5, va="top", fontweight="bold")

    # ── position error axes ──
    _dark(ax_err,
          title="Position Estimation Error",
          xlabel="Elapsed time  [days]",
          ylabel="‖r̂ − r_true‖  [km]")
    ax_err.set_yscale("log")
    ax_err.plot(d["t"], d["pos_err"], color=_CYAN, lw=0.7, alpha=0.20)
    ax_err.fill_between(d["t"], 1e-8, d["sig3"], color=_VIOLET, alpha=0.10, zorder=1)
    ax_err.plot([], [], color=_CYAN,   lw=2, label="‖r̂ − r‖")
    ax_err.plot([], [], color=_VIOLET, lw=6, alpha=0.35, label="3σ bound (x)")
    ax_err.legend(fontsize=7.5, loc="upper right",
                  facecolor=_PANEL, edgecolor=_BORDER, labelcolor=_TEXT)

    e_vline = ax_err.axvline(d["t"][0], color=_WHITE, lw=1.1, alpha=0.7)
    e_dot,  = ax_err.plot([], [], "o", color=_CYAN, ms=6, zorder=5)

    # ── NIS axes ──
    _dark(ax_nis,
          title=f"Filter Consistency: Normalised Innovation Squared (NIS)",
          xlabel="Elapsed time  [days]",
          ylabel="NIS  [χ²(2) distributed]")
    ax_nis.fill_between(d["t"], NIS_LO, NIS_HI, color=_GREEN, alpha=0.10, zorder=1,
                         label=f"95% χ²(2) band  [{NIS_LO:.2f}, {NIS_HI:.2f}]")
    ax_nis.axhline(2.0, color=_GREEN, lw=0.8, ls="--", alpha=0.5)
    ax_nis.set_ylim(-0.2, max(14.0, float(np.nanmax(d["nis"][np.isfinite(d["nis"])])) * 1.15))
    ax_nis.legend(fontsize=7.5, loc="upper right",
                  facecolor=_PANEL, edgecolor=_BORDER, labelcolor=_TEXT)

    n_vline = ax_nis.axvline(d["t"][0], color=_WHITE, lw=1.1, alpha=0.7)

    # pre-allocate scatter collection (grows each frame)
    nis_dots_in,  = ax_nis.plot([], [], "o", color=_GREEN, ms=4, alpha=0.85, zorder=4)
    nis_dots_out, = ax_nis.plot([], [], "o", color=_RED,   ms=4, alpha=0.85, zorder=4)

    nis_in_t,  nis_in_v  = [], []
    nis_out_t, nis_out_v = [], []

    def init3():
        for ln in [gt, ct, gh, ch, los_line]:
            ln.set_data([], [])
        for sc in [dt, dh]:
            sc.set_offsets(np.empty((0, 2)))
        e_vline.set_xdata([d["t"][0]])
        n_vline.set_xdata([d["t"][0]])
        e_dot.set_data([], [])
        nis_dots_in.set_data([], [])
        nis_dots_out.set_data([], [])
        meas_txt.set_text("")
        return gt, ct, gh, ch, los_line, dt, dh, e_vline, n_vline, e_dot, nis_dots_in, nis_dots_out

    def update3(frame):
        i  = min(frame * speed, N - 1)
        i0 = max(0, i - TRAIL)
        t_n = float(d["t"][i])

        # True trajectory trail
        xs_t = d["X_true"][i0:i+1, 0];  ys_t = d["X_true"][i0:i+1, 1]
        gt.set_data(xs_t, ys_t);  ct.set_data(xs_t, ys_t)
        dt.set_offsets([[d["X_true"][i, 0], d["X_true"][i, 1]]])

        # EKF estimate trail
        xs_h = d["X_hat"][i0:i+1, 0];  ys_h = d["X_hat"][i0:i+1, 1]
        gh.set_data(xs_h, ys_h);  ch.set_data(xs_h, ys_h)
        dh.set_offsets([[d["X_hat"][i, 0], d["X_hat"][i, 1]]])

        # LOS ray (true position → Moon, converted to km)
        rx, ry = d["X_true"][i, 0], d["X_true"][i, 1]
        mx, my = float(d["p2"][0]), float(d["p2"][1])
        los_line.set_data([rx, mx], [ry, my])

        # Follow spacecraft (zoom pan)
        ax_orb.set_xlim(rx - ZOOM_KM, rx + ZOOM_KM)
        ax_orb.set_ylim(ry - ZOOM_KM, ry + ZOOM_KM)

        # Measurement flash indicator
        if np.isfinite(d["nis"][i]):
            meas_txt.set_text("● MEAS")
        else:
            meas_txt.set_text("")

        # Right panels
        e_vline.set_xdata([t_n])
        n_vline.set_xdata([t_n])
        e_dot.set_data([t_n], [d["pos_err"][i]])

        # Accumulate NIS scatter
        if np.isfinite(d["nis"][i]):
            if NIS_LO <= d["nis"][i] <= NIS_HI:
                nis_in_t.append(t_n);  nis_in_v.append(d["nis"][i])
            else:
                nis_out_t.append(t_n); nis_out_v.append(d["nis"][i])

        nis_dots_in.set_data(nis_in_t, nis_in_v)
        nis_dots_out.set_data(nis_out_t, nis_out_v)

        fig.suptitle(
            f"Bearing-Only EKF Navigation  ·  t = {t_n:.2f} days"
            f"    pos error = {d['pos_err'][i]:,.0f} km",
            color=_TEXT, fontsize=12.5, y=0.97, fontweight="bold",
        )
        return gt, ct, gh, ch, los_line, dt, dh, e_vline, n_vline, e_dot, nis_dots_in, nis_dots_out, meas_txt

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
