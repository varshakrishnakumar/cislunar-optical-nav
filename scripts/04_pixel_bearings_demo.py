from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from _common import ensure_src_on_path

ensure_src_on_path()

import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import chi2
from visualization.style import plt

from dynamics.cr3bp import CR3BP
from dynamics.integrators import propagate
from nav.ekf import ekf_propagate_cr3bp_stm
from nav.measurements.bearing import bearing_update_tangent
from nav.measurements.pixel_bearing import pixel_detection_to_bearing
from cv.camera import Intrinsics
from cv.pointing import camera_dcm_from_boresight
from cv.sim_measurements import simulate_pixel_measurement


_BG     = "#080B14"
_PANEL  = "#0E1220"
_BORDER = "#1C2340"
_TEXT   = "#DCE0EC"
_DIM    = "#5A6080"
_CYAN   = "#22D3EE"
_AMBER  = "#F59E0B"
_GREEN  = "#10B981"
_RED    = "#F43F5E"
_VIOLET = "#8B5CF6"
_ORANGE = "#FB923C"
_MOON_C = "#C8CDD8"
_EARTH_C= "#3B82F6"


def _apply_dark_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor":  _BG,
        "axes.facecolor":    _PANEL,
        "axes.edgecolor":    _BORDER,
        "axes.labelcolor":   _TEXT,
        "axes.titlecolor":   _TEXT,
        "text.color":        _TEXT,
        "xtick.color":       _TEXT,
        "ytick.color":       _TEXT,
        "grid.color":        _BORDER,
        "grid.alpha":        1.0,
        "grid.linestyle":    "--",
        "lines.linewidth":   2.0,
        "legend.facecolor":  _PANEL,
        "legend.edgecolor":  _BORDER,
        "legend.labelcolor": _TEXT,
        "savefig.facecolor": _BG,
        "savefig.edgecolor": _BG,
        "font.size":         11,
    })


def _run_ffmpeg(frames_dir: Path, out_mp4: Path, fps: int) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(int(fps)),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "17",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _render_split_frame(
    *,
    out_path: Path,
    t: float,
    r_true_xy: np.ndarray,
    r_est_xy: np.ndarray,
    r_moon: np.ndarray,
    r_earth: np.ndarray,
    u_px_now: float,
    v_px_now: float,
    valid_now: bool,
    pos_err_hist: np.ndarray,
    nis_hist: np.ndarray,
    t_hist: np.ndarray,
    cam_width: int,
    cam_height: int,
    intr: Intrinsics,
    update_used: bool,
) -> None:
    dpi = 100
    fig_w, fig_h = 1600 / dpi, 720 / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=_BG)

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        left=0.05, right=0.97, top=0.94, bottom=0.09,
        wspace=0.30, hspace=0.35,
    )

    ax_cam = fig.add_subplot(gs[:, 0])
    ax_cam.set_facecolor("#050709")
    ax_cam.set_xlim(0, cam_width)
    ax_cam.set_ylim(cam_height, 0)

    ax_cam.plot([0, cam_width, cam_width, 0, 0],
                [0, 0, cam_height, cam_height, 0],
                color=_BORDER, lw=1.0)
    cx, cy = float(intr.cx), float(intr.cy)
    ax_cam.plot([cx - 14, cx + 14], [cy, cy], color=_DIM, lw=0.8)
    ax_cam.plot([cx, cx], [cy - 14, cy + 14], color=_DIM, lw=0.8)

    if valid_now and np.isfinite(u_px_now) and np.isfinite(v_px_now):
        dot_color = _GREEN if update_used else _AMBER
        ax_cam.scatter([u_px_now], [v_px_now], s=60, color=dot_color,
                       zorder=5, marker="o")
        ax_cam.plot([u_px_now - 7, u_px_now + 7], [v_px_now, v_px_now],
                    color=dot_color, lw=1.2, zorder=4)
        ax_cam.plot([u_px_now, u_px_now], [v_px_now - 7, v_px_now + 7],
                    color=dot_color, lw=1.2, zorder=4)

    status = "ACCEPTED" if update_used else ("DETECTED" if valid_now else "NO DETECT")
    status_color = _GREEN if update_used else (_AMBER if valid_now else _RED)
    ax_cam.text(12, 18, f"t = {t:.3f} dimensionless\n{status}",
                color=status_color, fontsize=10, family="monospace",
                va="top", ha="left",
                bbox=dict(facecolor="#050709", edgecolor=_BORDER, pad=5, alpha=0.85))
    ax_cam.text(12, cam_height - 12, "Camera Image Plane",
                color=_DIM, fontsize=9, family="monospace", va="bottom")
    ax_cam.set_xlabel("u [px]", color=_TEXT)
    ax_cam.set_ylabel("v [px]", color=_TEXT)

    ax_traj = fig.add_subplot(gs[0, 1])
    ax_traj.set_facecolor(_PANEL)
    ax_traj.scatter([r_earth[0]], [r_earth[1]], s=80, c=_EARTH_C, zorder=5, label="Earth")
    ax_traj.scatter([r_moon[0]],  [r_moon[1]],  s=55, c=_MOON_C,  zorder=5, label="Moon")


    ax_traj.plot(r_true_xy[:, 0], r_true_xy[:, 1],
                 color=_CYAN, lw=1.6, alpha=0.9, label="truth")
    ax_traj.scatter([r_true_xy[-1, 0]], [r_true_xy[-1, 1]],
                    s=30, c=_CYAN, zorder=6)

    if np.all(np.isfinite(r_est_xy[-1])):
        ax_traj.plot(r_est_xy[:, 0], r_est_xy[:, 1],
                     color=_AMBER, lw=1.6, ls=(0, (5, 3)), alpha=0.9, label="EKF est")
        ax_traj.scatter([r_est_xy[-1, 0]], [r_est_xy[-1, 1]],
                        s=30, c=_AMBER, marker="D", zorder=6)

    ax_traj.set_title("CR3BP XY Orbit (near L1)", color=_TEXT, fontsize=10)
    ax_traj.set_xlabel("x [dimensionless CR3BP length]", color=_TEXT, fontsize=9)
    ax_traj.set_ylabel("y [dimensionless CR3BP length]", color=_TEXT, fontsize=9)
    ax_traj.legend(loc="lower left", fontsize=8)
    ax_traj.grid(True)

    cx_r = float(r_true_xy[-1, 0])
    cy_r = float(r_true_xy[-1, 1])
    half = 0.04
    ax_traj.set_xlim(cx_r - half * 2.5, cx_r + half * 2.5)
    ax_traj.set_ylim(cy_r - half, cy_r + half)

    for sp in ax_traj.spines.values():
        sp.set_edgecolor(_BORDER)

    ax_err = fig.add_subplot(gs[1, 1])
    ax_err.set_facecolor(_PANEL)

    ax_err.semilogy(t_hist, pos_err_hist + 1e-12, color=_CYAN, lw=1.5,
                    label="‖pos err‖")

    ax_nis = ax_err.twinx()
    ax_nis.set_facecolor("none")
    nis_valid = np.isfinite(nis_hist)
    if nis_valid.any():
        ax_nis.fill_between(t_hist,
                            chi2.ppf(0.025, df=2), chi2.ppf(0.975, df=2),
                            color=_GREEN, alpha=0.10)
        ax_nis.scatter(t_hist[nis_valid], nis_hist[nis_valid], s=5,
                       color=_GREEN, alpha=0.7, zorder=3)
    ax_nis.set_ylabel("NIS", color=_GREEN, fontsize=9)
    ax_nis.tick_params(axis="y", colors=_GREEN)
    ax_nis.set_ylim(0, 20)

    ax_err.set_title("Position Error & NIS", color=_TEXT, fontsize=10)
    ax_err.set_xlabel("t [dimensionless CR3BP time]", color=_TEXT, fontsize=9)
    ax_err.set_ylabel("‖r̂ − r‖  [dimensionless CR3BP length]", color=_CYAN, fontsize=9)
    ax_err.tick_params(axis="y", colors=_CYAN)
    ax_err.grid(True)
    for sp in ax_err.spines.values():
        sp.set_edgecolor(_BORDER)

    fig.suptitle(
        "Cislunar Optical Navigation — Pixel-Based Bearing EKF  |  Earth-Moon CR3BP",
        color=_TEXT, fontsize=11, y=0.98
    )
    fig.savefig(out_path, dpi=dpi, facecolor=_BG)
    plt.close(fig)


def main() -> None:
    _apply_dark_theme()

    plots_dir  = Path("results/plots");  plots_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = Path("results/videos"); videos_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = videos_dir / "04_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    mu = 0.0121505856
    model = CR3BP(mu=mu)

    L1x    = model.lagrange_points()["L1"][0]
    x0_nom = np.array([L1x - 1e-3, 0.0, 0.0, 0.0, 0.05, 0.0], dtype=float)

    r_body  = np.asarray(model.primary2, dtype=float)
    r_earth = np.asarray(model.primary1, dtype=float)

    t0, tf   = 0.0, 6.0
    dt_meas  = 0.02
    sigma_px = 1.5
    q_acc    = 1e-14
    seed     = 7

    t_meas = np.arange(t0, tf + 1e-12, dt_meas)
    N      = len(t_meas)

    intr = Intrinsics(fx=400., fy=400., cx=320., cy=240., width=640, height=480)

    x_true0      = x0_nom.copy()
    x_true0[:3] += np.array([1e-4, -1e-4, 0.0])

    res_truth = propagate(model.eom, (t0, tf), x_true0, t_eval=t_meas,
                          rtol=1e-11, atol=1e-13)
    if not res_truth.success:
        raise RuntimeError(f"Truth propagation failed: {res_truth.message}")
    X_true = res_truth.x

    rng = np.random.default_rng(seed)

    u_px_arr     = np.full(N, np.nan)
    v_px_arr     = np.full(N, np.nan)
    valid_arr    = np.zeros(N, dtype=bool)
    R_cam_arr    = np.zeros((N, 3, 3), dtype=float)

    for k in range(N):
        r_sc = X_true[k, :3]
        boresight = r_body - r_sc
        R_cam = camera_dcm_from_boresight(boresight, camera_forward_axis="+z")
        R_cam_arr[k] = R_cam

        meas = simulate_pixel_measurement(
            r_sc=r_sc, r_body=r_body,
            intrinsics=intr, R_cam_from_frame=R_cam,
            sigma_px=sigma_px, rng=rng, t=float(t_meas[k]),
            dropout_p=0.0, out_of_frame="drop", behind="drop",
        )
        if meas.valid and np.isfinite(meas.u_px):
            u_px_arr[k] = meas.u_px
            v_px_arr[k] = meas.v_px
            valid_arr[k] = True

    xhat         = x0_nom.copy()
    xhat[:3]    += np.array([1e-4, 1e-4, 0.0])
    P            = np.diag([1e-6] * 6).astype(float)

    X_hat        = np.zeros((N, 6), dtype=float)
    nis_arr      = np.full(N, np.nan, dtype=float)
    pos_err_arr  = np.zeros(N, dtype=float)
    upd_used_arr = np.zeros(N, dtype=bool)
    P_diag_arr   = np.zeros((N, 6), dtype=float)

    X_hat[0]      = xhat
    P_diag_arr[0] = np.diag(P)
    pos_err_arr[0]= np.linalg.norm(xhat[:3] - X_true[0, :3])

    t_prev = t_meas[0]
    for k in range(1, N):
        tk = float(t_meas[k])
        xhat, P, _ = ekf_propagate_cr3bp_stm(
            mu=mu, x=xhat, P=P, t0=t_prev, t1=tk, q_acc=q_acc
        )

        if valid_arr[k]:
            R_frame_from_cam = R_cam_arr[k].T
            u_meas, sigma_theta = pixel_detection_to_bearing(
                u_px_arr[k], v_px_arr[k], sigma_px, intr, R_frame_from_cam
            )
            if np.all(np.isfinite(u_meas)):
                upd = bearing_update_tangent(xhat, P, u_meas, r_body, float(sigma_theta))
                nis_arr[k] = upd.nis
                if upd.accepted:
                    xhat, P = upd.x_upd, upd.P_upd
                    upd_used_arr[k] = True

        X_hat[k]       = xhat
        P_diag_arr[k]  = np.diag(P)
        pos_err_arr[k] = np.linalg.norm(xhat[:3] - X_true[k, :3])
        t_prev = tk

    vel_err_arr = np.linalg.norm(X_hat[:, 3:6] - X_true[:, 3:6], axis=1)

    print(f"Valid measurements : {int(valid_arr.sum())} / {N}")
    print(f"Updates used       : {int(upd_used_arr.sum())}")
    print(f"Final pos error    : {pos_err_arr[-1]:.3e} dimensionless CR3BP length")
    print(f"Final vel error    : {vel_err_arr[-1]:.3e} dimensionless CR3BP velocity")
    print(f"Mean NIS (k≥1)     : {np.nanmean(nis_arr[1:]):.3f}")

    fig, axs = plt.subplots(2, 2, figsize=(13, 8),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.30})
    fig.patch.set_facecolor(_BG)

    ax = axs[0, 0]
    ax.set_facecolor(_PANEL)
    ax.scatter([r_earth[0]], [r_earth[1]], s=90, c=_EARTH_C, zorder=5, label="Earth")
    ax.scatter([r_body[0]],  [r_body[1]],  s=65, c=_MOON_C,  zorder=5, label="Moon")
    ax.plot(X_true[:, 0], X_true[:, 1], color=_CYAN,  lw=1.8, label="Truth")
    ax.plot(X_hat[:, 0],  X_hat[:, 1],  color=_AMBER, lw=1.8, ls=(0, (6, 3)),
            label="EKF estimate", alpha=0.9)
    upd_idx = np.where(upd_used_arr)[0]
    ax.scatter(X_true[upd_idx, 0], X_true[upd_idx, 1],
               s=12, c=_GREEN, zorder=6, label="Updates", alpha=0.7)
    ax.set_title("CR3BP XY Trajectory", color=_TEXT)
    ax.set_xlabel("x  [dimensionless CR3BP length]", color=_TEXT)
    ax.set_ylabel("y  [dimensionless CR3BP length]", color=_TEXT)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True)
    for sp in ax.spines.values(): sp.set_edgecolor(_BORDER)

    ax = axs[0, 1]
    ax.set_facecolor(_PANEL)
    sig_pos = 3.0 * np.sqrt(np.abs(P_diag_arr[:, 0]))
    ax.fill_between(t_meas, 0, sig_pos, color=_VIOLET, alpha=0.15, label="3σ (x)")
    ax.semilogy(t_meas, pos_err_arr + 1e-12, color=_CYAN,  lw=1.8, label="‖pos err‖")
    ax.semilogy(t_meas, vel_err_arr + 1e-12, color=_AMBER, lw=1.5, ls="--",
                label="‖vel err‖", alpha=0.85)
    ax.set_title("State Estimation Errors", color=_TEXT)
    ax.set_xlabel("t  [dimensionless CR3BP time]", color=_TEXT)
    ax.set_ylabel("Error norm  [dimensionless CR3BP units]", color=_TEXT)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True)
    for sp in ax.spines.values(): sp.set_edgecolor(_BORDER)

    ax = axs[1, 0]
    ax.set_facecolor(_PANEL)
    nis_lo = chi2.ppf(0.025, df=2)
    nis_hi = chi2.ppf(0.975, df=2)
    ax.fill_between(t_meas, nis_lo, nis_hi, color=_GREEN, alpha=0.12,
                    label=f"95% χ²(2): [{nis_lo:.2f}, {nis_hi:.2f}]")
    ax.axhline(2.0, color=_GREEN, lw=0.9, ls="--", alpha=0.5)
    nis_ok = np.isfinite(nis_arr)
    in_band  = nis_ok & (nis_arr >= nis_lo) & (nis_arr <= nis_hi)
    out_band = nis_ok & ~in_band
    ax.scatter(t_meas[in_band],  nis_arr[in_band],  s=10, c=_GREEN, zorder=4)
    ax.scatter(t_meas[out_band], nis_arr[out_band], s=10, c=_RED,   zorder=4)
    ax.set_title(f"NIS  (mean={np.nanmean(nis_arr[1:]):.2f})", color=_TEXT)
    ax.set_xlabel("t  [dimensionless CR3BP time]", color=_TEXT)
    ax.set_ylabel("NIS", color=_TEXT)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 18)
    ax.grid(True)
    for sp in ax.spines.values(): sp.set_edgecolor(_BORDER)

    ax = axs[1, 1]
    ax.set_facecolor("#050709")
    ax.set_xlim(0, intr.width)
    ax.set_ylim(intr.height, 0)
    ax.plot([0, intr.width, intr.width, 0, 0],
            [0, 0, intr.height, intr.height, 0], color=_BORDER, lw=0.8)
    ax.axvline(intr.cx, color=_DIM, lw=0.7, ls="--")
    ax.axhline(intr.cy, color=_DIM, lw=0.7, ls="--")
    v_mask = valid_arr
    scatter = ax.scatter(u_px_arr[v_mask], v_px_arr[v_mask], c=t_meas[v_mask],
                         cmap="plasma", s=8, vmin=t0, vmax=tf, zorder=3)
    cb = fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("t [dimensionless CR3BP time]", color=_TEXT)
    cb.ax.yaxis.set_tick_params(color=_TEXT)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=_TEXT)
    ax.set_title("Moon Detection Track on Image Plane", color=_TEXT)
    ax.set_xlabel("u  [px]", color=_TEXT)
    ax.set_ylabel("v  [px]", color=_TEXT)
    for sp in ax.spines.values(): sp.set_edgecolor(_BORDER)

    fig.suptitle(
        "Pixel-Based Optical Navigation — EKF Near L1  |  Earth-Moon CR3BP",
        color=_TEXT, fontsize=13, y=1.01
    )
    fig.savefig(plots_dir / "04_pixel_bearings_report.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {plots_dir / '04_pixel_bearings_report.png'}")

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found — skipping video generation.")
        return

    print(f"Rendering {N} video frames ...")
    stride = 2
    frame_id = 0
    for k in range(0, N, stride):
        _render_split_frame(
            out_path    = frames_dir / f"frame_{frame_id:05d}.png",
            t           = float(t_meas[k]),
            r_true_xy   = X_true[: k + 1, :2],
            r_est_xy    = X_hat[: k + 1, :2],
            r_moon      = r_body[:2],
            r_earth     = r_earth[:2],
            u_px_now    = float(u_px_arr[k]),
            v_px_now    = float(v_px_arr[k]),
            valid_now   = bool(valid_arr[k]),
            pos_err_hist= pos_err_arr[: k + 1],
            nis_hist    = nis_arr[: k + 1],
            t_hist      = t_meas[: k + 1],
            cam_width   = intr.width,
            cam_height  = intr.height,
            intr        = intr,
            update_used = bool(upd_used_arr[k]),
        )
        frame_id += 1

    out_mp4 = videos_dir / "04_pixel_bearings.mp4"
    video_fps = max(1, int(round((1.0 / (dt_meas * stride)) * 0.5)))
    _run_ffmpeg(frames_dir, out_mp4, fps=video_fps)
    print(f"Wrote: {out_mp4}")


if __name__ == "__main__":
    main()
