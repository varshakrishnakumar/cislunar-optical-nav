from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from scipy.stats import chi2

from dynamics.cr3bp import CR3BP
from dynamics.integrators import propagate
from dynamics.variational import cr3bp_eom_with_stm
from nav.ekf import ekf_propagate_cr3bp_stm
from nav.measurements.bearing import los_unit, tangent_basis, bearing_update_tangent

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

OUT_DIR = Path("results/plots")



def _try_save(ani, path: Path, fps: int):
    try:
        w = FFMpegWriter(fps=fps, bitrate=2200,
                         extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
        ani.save(str(path), writer=w)
        print(f"  saved → {path}")
    except Exception as e:
        gif = path.with_suffix(".gif")
        print(f"  ffmpeg failed ({e}), falling back → {gif}")
        ani.save(str(gif), writer=PillowWriter(fps=fps))
        print(f"  saved → {gif}")


def _add_stars(ax, n=220, rng=None):
    rng = rng or np.random.default_rng(42)
    xs = rng.uniform(0.78, 1.16, n)
    ys = rng.uniform(-0.16, 0.16, n)
    zs = np.full(n, -0.005)
    sizes = rng.uniform(0.4, 2.5, n)
    ax.scatter(xs, ys, zs, s=sizes, color="white", alpha=0.35, zorder=0,
               depthshade=False)


def _setup_cr3bp():
    mu = 0.0121505856
    model = CR3BP(mu=mu)
    L = model.lagrange_points()
    L1x = L["L1"][0]
    x0_nom = np.array([L1x - 1e-3, 0.0, 0.0, 0.0, 0.05, 0.0], dtype=float)
    return mu, model, L, x0_nom


def _dark_right_ax(ax):
    ax.set_facecolor(_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(_BORDER)
    ax.tick_params(colors=_TEXT)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    ax.grid(True, color=_BORDER, lw=0.5)


def _style_3d(ax):
    ax.set_facecolor(_BG)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(_BORDER)
    ax.yaxis.pane.set_edgecolor(_BORDER)
    ax.zaxis.pane.set_edgecolor(_BORDER)
    ax.tick_params(colors=_DIM, labelsize=7)
    ax.xaxis.label.set_color(_DIM)
    ax.yaxis.label.set_color(_DIM)
    ax.zaxis.label.set_color(_DIM)
    ax.set_xlabel("x [ND]", labelpad=2)
    ax.set_ylabel("y [ND]", labelpad=2)
    ax.set_zlabel("z", labelpad=1)
    ax.grid(True, color=_BORDER, lw=0.3, alpha=0.5)



def _pack(x, phi=None):
    phi = np.eye(6) if phi is None else phi
    return np.concatenate([x, phi.reshape(-1, order="F")])

def _unpack(z):
    return z[:6].copy(), z[6:].reshape(6, 6, order="F").copy()

def _prop_stm(mu, t0_, tf_, z0):
    return propagate(
        lambda t, z: cr3bp_eom_with_stm(t, z, mu),
        (t0_, tf_), z0, rtol=1e-11, atol=1e-13, method="DOP853"
    )



def _compute_phase2(mu, model, x0_nom):
    t0, tf, tc = 0.0, 6.0, 2.0
    dx0 = np.array([2e-4, -1e-4, 0.0, 0.0, 2e-3, 0.0])
    eom = model.eom

    res_nom = propagate(eom, (t0, tf), x0_nom, dense_output=True,
                        rtol=1e-11, atol=1e-13, method="DOP853")
    r_target = res_nom.sol(tf)[:3]

    x0_err = x0_nom + dx0
    res_unc = propagate(eom, (t0, tf), x0_err, dense_output=True,
                        rtol=1e-11, atol=1e-13, method="DOP853")

    z0_tc = _pack(x0_err)
    res_tc = _prop_stm(mu, t0, tc, z0_tc)
    x_tc, _ = _unpack(res_tc.x[-1])
    dv = np.zeros(3)
    for _ in range(15):
        xb = x_tc.copy(); xb[3:6] += dv
        res_tf = _prop_stm(mu, tc, tf, _pack(xb))
        x_tf, phi = _unpack(res_tf.x[-1])
        err = x_tf[:3] - r_target
        if np.linalg.norm(err) < 1e-10: break
        try:    dv -= np.linalg.solve(phi[:3, 3:6], err)
        except: dv -= np.linalg.lstsq(phi[:3, 3:6], err, rcond=None)[0]

    N = 700
    t_plot = np.linspace(t0, tf, N)
    idx_pre  = t_plot <= tc
    idx_post = t_plot >= tc

    X_nom = res_nom.sol(t_plot).T
    X_unc = res_unc.sol(t_plot).T

    res_pre = propagate(eom, (t0, tc), x0_err, dense_output=True,
                        rtol=1e-11, atol=1e-13, method="DOP853")
    xb_full = res_pre.sol(tc).reshape(6,); xb_full[3:6] += dv
    res_post = propagate(eom, (tc, tf), xb_full, dense_output=True,
                         rtol=1e-11, atol=1e-13, method="DOP853")

    X_cor = np.full((N, 6), np.nan)
    X_cor[idx_pre]  = res_pre.sol(t_plot[idx_pre]).T
    X_cor[idx_post] = res_post.sol(t_plot[idx_post]).T

    miss_unc = np.linalg.norm(X_unc[:, :3] - r_target, axis=1)
    miss_cor = np.linalg.norm(X_cor[:, :3] - r_target, axis=1)
    tc_idx   = int(np.argmin(np.abs(t_plot - tc)))

    return dict(
        t=t_plot, tc=tc, tc_idx=tc_idx, dv=dv,
        r_target=r_target,
        X_nom=X_nom, X_unc=X_unc, X_cor=X_cor,
        miss_unc=miss_unc, miss_cor=miss_cor,
        p1=model.primary1, p2=model.primary2,
    )



def _add_ang_noise(u, sig, rng):
    u = u / np.linalg.norm(u)
    e1, e2 = tangent_basis(u)
    d = rng.normal(0, sig, 2)
    up = u + d[0]*e1 + d[1]*e2
    return up / np.linalg.norm(up)


def _compute_phase3(mu, model, x0_nom):
    rng = np.random.default_rng(0)
    r_body = np.asarray(model.primary2, dtype=float)
    t0, tf, dt = 0.0, 6.0, 0.02
    t_meas = np.arange(t0, tf + 1e-12, dt)

    x_true0 = x0_nom.copy()
    x_true0[:3] += [2e-4, -1e-4, 0.0]
    x_true0[3:] += [0.0,  2e-3, 0.0]

    res = propagate(model.eom, (t0, tf), x_true0, t_eval=t_meas,
                    rtol=1e-11, atol=1e-13, method="DOP853")
    X_true = res.x

    sig = 2e-4
    U_meas = np.array([
        _add_ang_noise(los_unit(r_body, X_true[k, :3])[0], sig, rng)
        for k in range(len(t_meas))
    ])

    x = x0_nom.copy()
    P = np.diag([1e-6]*3 + [1e-8]*3).astype(float)
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
        if upd.accepted: x, P = upd.x_upd, upd.P_upd
        nis[k] = upd.nis
        X_hat[k] = x; P_diag[k] = np.diag(P)
        t_prev = tk

    pos_err = np.linalg.norm(X_hat[:, :3] - X_true[:, :3], axis=1)
    sig3    = 3.0 * np.sqrt(np.abs(P_diag[:, 0]))

    return dict(
        t=t_meas, X_true=X_true, X_hat=X_hat,
        pos_err=pos_err, sig3=sig3, nis=nis,
        r_body=r_body, p1=model.primary1, p2=model.primary2,
    )



def animate_phase2(fps=30, sim_speed=0.77):
    print("Phase 2: computing…")
    mu, model, L, x0_nom = _setup_cr3bp()
    d = _compute_phase2(mu, model, x0_nom)
    t, N = d["t"], len(d["t"])
    dt    = float(t[1] - t[0])
    speed = max(1, round(sim_speed / (fps * dt)))

    fig = plt.figure(figsize=(16, 7), facecolor=_BG)
    ax3 = fig.add_axes([0.02, 0.05, 0.54, 0.88], projection="3d")
    axR = fig.add_axes([0.60, 0.12, 0.37, 0.75])

    _style_3d(ax3)
    _dark_right_ax(axR)
    _add_stars(ax3)

    ax3.set_title("", color=_TEXT)

    ax3.plot(d["X_nom"][:, 0], d["X_nom"][:, 1], np.zeros(N),
             color=_GREEN, lw=0.6, alpha=0.15)

    p1, p2 = d["p1"], d["p2"]
    ax3.scatter([p1[0]], [p1[1]], [0], s=80,  color="#4466FF", zorder=6, depthshade=False)
    ax3.scatter([p2[0]], [p2[1]], [0], s=50,  color="#BBBBBB", zorder=6, depthshade=False)
    ax3.scatter([d["r_target"][0]], [d["r_target"][1]], [0],
                s=120, color=_AMBER, marker="*", zorder=7, depthshade=False)

    ax3.view_init(elev=22, azim=-55)

    legend_txt = (
        "● Earth    ● Moon    ★ Target\n"
        "─ nominal  ─ truth   ─ corrected"
    )
    ax3.text2D(0.01, 0.01, legend_txt, transform=ax3.transAxes,
               color=_TEXT, fontsize=7.5, va="bottom",
               bbox=dict(facecolor=_PANEL, edgecolor=_BORDER, alpha=0.8, pad=4))

    axR.plot(t, d["miss_unc"], color=_AMBER, lw=1.0, alpha=0.30)
    valid_cor = np.where(np.isfinite(d["miss_cor"]), d["miss_cor"], np.nan)
    axR.plot(t, valid_cor,    color=_CYAN,  lw=1.0, alpha=0.30)
    axR.axvline(d["tc"], color=_RED, lw=1.0, ls=":", alpha=0.5)
    axR.text(d["tc"] + 0.05, axR.get_ylim()[0] if axR.get_ylim()[0] > 0 else 1e-5,
             "burn tc", color=_RED, fontsize=8)
    axR.set_yscale("log")
    axR.set_xlabel("t [ND]"); axR.set_ylabel("‖r(t) − r_target‖  [ND]")
    axR.set_title("Miss Distance vs Time")
    axR.plot([], [], color=_AMBER, label="uncorrected")
    axR.plot([], [], color=_CYAN,  label="corrected")
    axR.legend(fontsize=8, loc="upper right")

    trail_unc, = ax3.plot([], [], [], color=_AMBER, lw=1.8, alpha=0.85)
    trail_cor, = ax3.plot([], [], [], color=_CYAN,  lw=1.8, alpha=0.85)
    sc_unc     = ax3.scatter([], [], [], s=60,  color=_AMBER, zorder=9, depthshade=False)
    sc_cor     = ax3.scatter([], [], [], s=60,  color=_CYAN,  zorder=10, depthshade=False)
    burn_dot   = ax3.scatter([], [], [], s=110, color=_RED, marker="D", zorder=11, depthshade=False)

    vline = axR.axvline(t[0], color=_WHITE, lw=1.2, alpha=0.7)
    dot_unc, = axR.plot([], [], "o", color=_AMBER, ms=6, zorder=8)
    dot_cor, = axR.plot([], [], "o", color=_CYAN,  ms=6, zorder=8)

    ZOOM = 0.065

    burn_fired = [False]
    burn_xy    = [None]

    def init():
        trail_unc.set_data_3d([], [], [])
        trail_cor.set_data_3d([], [], [])
        sc_unc._offsets3d  = ([], [], [])
        sc_cor._offsets3d  = ([], [], [])
        burn_dot._offsets3d = ([], [], [])
        vline.set_xdata([t[0]])
        dot_unc.set_data([], []); dot_cor.set_data([], [])
        return trail_unc, trail_cor, sc_unc, sc_cor, burn_dot, vline, dot_unc, dot_cor

    def update(frame):
        i = min(frame * speed, N - 1)
        t_now = float(t[i])

        trail_unc.set_data_3d(d["X_unc"][:i+1, 0],
                               d["X_unc"][:i+1, 1],
                               np.zeros(i+1))
        sc_unc._offsets3d = ([d["X_unc"][i, 0]], [d["X_unc"][i, 1]], [0])

        cor_xy = d["X_cor"][:i+1]
        mask = np.isfinite(cor_xy[:, 0])
        if mask.any():
            trail_cor.set_data_3d(cor_xy[mask, 0], cor_xy[mask, 1], np.zeros(mask.sum()))
            sc_cor._offsets3d = ([cor_xy[mask, 0][-1]], [cor_xy[mask, 1][-1]], [0])
        else:
            trail_cor.set_data_3d([], [], [])
            sc_cor._offsets3d = ([], [], [])

        if t_now >= d["tc"] and not burn_fired[0]:
            burn_fired[0] = True
            pre = d["X_cor"][:d["tc_idx"]+1]
            valid_pre = pre[np.isfinite(pre[:, 0])]
            if len(valid_pre):
                burn_xy[0] = valid_pre[-1, :2]
        if burn_fired[0] and burn_xy[0] is not None:
            burn_dot._offsets3d = ([burn_xy[0][0]], [burn_xy[0][1]], [0])

        cx, cy = d["X_unc"][i, 0], d["X_unc"][i, 1]
        ax3.set_xlim3d(cx - ZOOM, cx + ZOOM)
        ax3.set_ylim3d(cy - ZOOM, cy + ZOOM)
        ax3.set_zlim3d(-0.01, 0.01)

        vline.set_xdata([t_now])
        dot_unc.set_data([t_now], [d["miss_unc"][i]])
        if np.isfinite(d["miss_cor"][i]):
            dot_cor.set_data([t_now], [d["miss_cor"][i]])

        phase = "PRE-BURN" if t_now < d["tc"] else f"POST-BURN  |Δv|={np.linalg.norm(d['dv']):.2e} ND"
        fig.suptitle(
            f"Phase 2 — Midcourse Correction  |  t = {t_now:.3f} ND  |  {phase}",
            color=_TEXT, fontsize=12, y=0.98
        )

        return trail_unc, trail_cor, sc_unc, sc_cor, burn_dot, vline, dot_unc, dot_cor

    n_frames = int(np.ceil(N / speed)) + 30
    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        blit=False, interval=1000/fps)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _try_save(ani, OUT_DIR / "anim_02_targeting_v2.mp4", fps)
    plt.close(fig)



def animate_phase3(fps=30, sim_speed=0.77):
    print("Phase 3: running EKF…")
    mu, model, L, x0_nom = _setup_cr3bp()
    d = _compute_phase3(mu, model, x0_nom)
    t, N = d["t"], len(d["t"])
    dt    = float(t[1] - t[0])
    speed = max(1, round(sim_speed / (fps * dt)))
    nis_lo = chi2.ppf(0.025, df=2)
    nis_hi = chi2.ppf(0.975, df=2)

    fig = plt.figure(figsize=(16, 7), facecolor=_BG)
    ax3 = fig.add_axes([0.02, 0.05, 0.54, 0.88], projection="3d")
    axR = fig.add_axes([0.60, 0.12, 0.37, 0.75])

    _style_3d(ax3)
    _dark_right_ax(axR)
    _add_stars(ax3)

    p2 = d["p2"]
    ax3.scatter([p2[0]], [p2[1]], [0], s=70, color="#CCCCCC", zorder=6,
                depthshade=False, label="Moon")

    ax3.view_init(elev=22, azim=-55)

    legend_txt = (
        "● Moon (target)\n"
        "─ True trajectory\n"
        "── EKF estimate\n"
        "⋯ LOS bearing ray"
    )
    ax3.text2D(0.01, 0.01, legend_txt, transform=ax3.transAxes,
               color=_TEXT, fontsize=8, va="bottom",
               bbox=dict(facecolor=_PANEL, edgecolor=_BORDER, alpha=0.85, pad=4))

    axR.fill_between(t, 0, d["sig3"], color=_VIOLET, alpha=0.12)
    axR.plot(t, d["pos_err"], color=_CYAN, lw=0.8, alpha=0.25)
    axR.set_yscale("log")
    axR.set_xlabel("t [ND]"); axR.set_ylabel("‖pos err‖  [ND]")
    axR.set_title("Position Error + 3σ Bound")
    axR.plot([], [], color=_VIOLET, alpha=0.5, lw=6, label="3σ (x)")
    axR.plot([], [], color=_CYAN,            label="‖pos err‖")
    axR.legend(fontsize=8, loc="lower right")

    trail_true, = ax3.plot([], [], [], color=_CYAN,  lw=1.6, alpha=0.90)
    trail_hat,  = ax3.plot([], [], [], color=_AMBER, lw=1.6, alpha=0.90, ls="--")
    sc_true     = ax3.scatter([], [], [], s=70,  color=_RED,   zorder=9,  depthshade=False)
    sc_hat      = ax3.scatter([], [], [], s=55,  color="#CC88FF", zorder=10, depthshade=False)
    los_line,   = ax3.plot([], [], [], color=_VIOLET, lw=0.8, alpha=0.55, ls=":")

    vline  = axR.axvline(t[0], color=_WHITE, lw=1.2, alpha=0.7)
    dot_e, = axR.plot([], [], "o", color=_CYAN, ms=6, zorder=8)

    ZOOM = 0.070

    def init():
        for a in [trail_true, trail_hat, los_line]:
            a.set_data_3d([], [], [])
        sc_true._offsets3d  = ([], [], [])
        sc_hat._offsets3d   = ([], [], [])
        vline.set_xdata([t[0]])
        dot_e.set_data([], [])
        return trail_true, trail_hat, los_line, sc_true, sc_hat, vline, dot_e

    def update(frame):
        i = min(frame * speed, N - 1)
        t_now = float(t[i])

        trail_true.set_data_3d(d["X_true"][:i+1, 0],
                                d["X_true"][:i+1, 1],
                                np.zeros(i+1))
        trail_hat.set_data_3d( d["X_hat"][:i+1, 0],
                                d["X_hat"][:i+1, 1],
                                np.zeros(i+1))

        sc_true._offsets3d = ([d["X_true"][i, 0]], [d["X_true"][i, 1]], [0])
        sc_hat._offsets3d  = ([d["X_hat"][i, 0]],  [d["X_hat"][i, 1]],  [0])

        rx, ry = d["X_true"][i, 0], d["X_true"][i, 1]
        mx, my = float(p2[0]), float(p2[1])
        los_line.set_data_3d([rx, mx], [ry, my], [0, 0])

        cx, cy = rx, ry
        ax3.set_xlim3d(cx - ZOOM, cx + ZOOM)
        ax3.set_ylim3d(cy - ZOOM, cy + ZOOM)
        ax3.set_zlim3d(-0.01, 0.01)

        vline.set_xdata([t_now])
        dot_e.set_data([t_now], [d["pos_err"][i]])

        fig.suptitle(
            f"Phase 3 — Bearing-Only EKF  |  t = {t_now:.3f} ND"
            f"    Δt = {t[1]-t[0]:.2f}    ‖r̂−r‖ = {d['pos_err'][i]:.2e} ND",
            color=_TEXT, fontsize=11.5, y=0.98
        )

        return trail_true, trail_hat, los_line, sc_true, sc_hat, vline, dot_e

    n_frames = int(np.ceil(N / speed)) + 30
    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        blit=False, interval=1000/fps)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _try_save(ani, OUT_DIR / "anim_03_bearings_v2.mp4", fps)
    plt.close(fig)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["2", "3"], default=None)
    parser.add_argument("--fps",   type=int, default=30)
    parser.add_argument("--sim-speed", type=float, default=0.77,
                        help="Simulated ND time per real second (same for both phases)")
    args = parser.parse_args()

    if args.phase in (None, "2"):
        animate_phase2(fps=args.fps, sim_speed=args.sim_speed)
    if args.phase in (None, "3"):
        animate_phase3(fps=args.fps, sim_speed=args.sim_speed)
