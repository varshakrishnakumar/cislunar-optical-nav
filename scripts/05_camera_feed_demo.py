from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from _common import ensure_src_on_path

ensure_src_on_path()

import numpy as np
from matplotlib.patches import Rectangle
from visualization.style import plt
from cv.pointing import camera_dcm_from_boresight
from dynamics.cr3bp import CR3BP
from dynamics.integrators import propagate
from nav.ekf import ekf_propagate_cr3bp_stm
from nav.measurements.bearing import bearing_update_tangent
from nav.measurements.pixel_bearing import pixel_detection_to_bearing
from cv.camera import Intrinsics
from cv.sim_measurements import simulate_bbox_measurement


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


def _run_ffmpeg(frames_dir: Path, out_mp4: Path, fps: int) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(int(fps)),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _style_right_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_TEXT)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(_BORDER)
    ax.grid(True, color=_BORDER, alpha=1.0, ls="--")


def _render_camera_frame(
    *,
    out_path: Path,
    width: int,
    height: int,
    t: float,
    valid: bool,
    u_px: float | None,
    v_px: float | None,
    bbox_xyxy: tuple[float, float, float, float] | None,
    update_used: bool,
    overlay_text: bool,
    dot_radius_px: float,
) -> None:
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="#050709")
    ax = fig.add_axes([0, 0, 1, 1])
    blank = np.zeros((height, width), dtype=np.float32)
    ax.imshow(blank, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
    ax.set_axis_off()

    ax.plot([1, width - 1, width - 1, 1, 1],
            [1, 1, height - 1, height - 1, 1],
            color=_BORDER, lw=0.8, alpha=0.6)

    if valid and (u_px is not None) and (v_px is not None):
        dot_color = _GREEN if update_used else _AMBER
        ax.add_patch(plt.Circle((u_px, v_px), radius=dot_radius_px,
                                fill=True, linewidth=0, color=dot_color))
        if bbox_xyxy is not None:
            xmin, ymin, xmax, ymax = bbox_xyxy
            ax.add_patch(Rectangle(
                (xmin, ymin),
                max(1.0, xmax - xmin), max(1.0, ymax - ymin),
                fill=False, linewidth=1.4, edgecolor=dot_color, alpha=0.85,
            ))

    if overlay_text:
        if valid and (u_px is not None):
            uv_str = f"u,v = ({u_px:7.1f}, {v_px:7.1f})"
            status = "UPDATE" if update_used else "DETECT"
            status_color = _GREEN if update_used else _AMBER
        else:
            uv_str = "u,v = (   —  ,    —  )"
            status = "NO SIGNAL"
            status_color = _RED

        txt = f"t = {t:7.3f} dimensionless\n{uv_str}\n{status}"
        ax.text(14, 22, txt, color=status_color, fontsize=12,
                family="monospace", va="top", ha="left",
                bbox=dict(facecolor=(0, 0, 0, 0.55),
                          edgecolor=(_BORDER + "80"), pad=6))

    fig.savefig(out_path, dpi=dpi, facecolor="#050709")
    plt.close(fig)


def _render_split_frame(
    *,
    out_path: Path,
    width_left: int,
    height_left: int,
    t: float,
    valid: bool,
    u_px: float | None,
    v_px: float | None,
    bbox_xyxy: tuple[float, float, float, float] | None,
    update_used: bool,
    overlay_text: bool,
    dot_radius_px: float,
    r_true_xy: np.ndarray,
    r_est_xy: np.ndarray,
) -> None:
    dpi = 100
    right_w   = int(0.95 * height_left)
    total_w   = width_left + right_w
    total_h   = height_left

    fig = plt.figure(figsize=(total_w / dpi, total_h / dpi),
                     dpi=dpi, facecolor=_BG)

    axL = fig.add_axes([0.0, 0.0, width_left / total_w, 1.0])
    blank = np.zeros((height_left, width_left), dtype=np.float32)
    axL.imshow(blank, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
    axL.set_axis_off()
    axL.plot([1, width_left - 1, width_left - 1, 1, 1],
             [1, 1, height_left - 1, height_left - 1, 1],
             color=_BORDER, lw=0.8, alpha=0.6)

    if valid and (u_px is not None) and (v_px is not None):
        dot_color = _GREEN if update_used else _AMBER
        axL.add_patch(plt.Circle((u_px, v_px), radius=dot_radius_px,
                                 fill=True, linewidth=0, color=dot_color))
        if bbox_xyxy is not None:
            xmin, ymin, xmax, ymax = bbox_xyxy
            axL.add_patch(Rectangle(
                (xmin, ymin),
                max(1.0, xmax - xmin), max(1.0, ymax - ymin),
                fill=False, linewidth=1.4, edgecolor=dot_color, alpha=0.85,
            ))

    if overlay_text:
        if valid and (u_px is not None):
            uv_str = f"u,v = ({u_px:7.1f}, {v_px:7.1f})"
            status = "UPDATE" if update_used else "DETECT"
            sc = _GREEN if update_used else _AMBER
        else:
            uv_str = "u,v = (   —  ,    —  )"
            status = "NO SIGNAL"
            sc = _RED
        axL.text(14, 22, f"t = {t:7.3f} dimensionless\n{uv_str}\n{status}",
                 color=sc, fontsize=12, family="monospace",
                 va="top", ha="left",
                 bbox=dict(facecolor=(0, 0, 0, 0.55),
                           edgecolor=(_BORDER + "80"), pad=6))

    axR = fig.add_axes([width_left / total_w, 0.05,
                        right_w / total_w * 0.92, 0.88])
    _style_right_ax(axR)


    if r_true_xy.shape[0] >= 1:
        axR.plot(r_true_xy[:, 0], r_true_xy[:, 1],
                 color=_CYAN, lw=2.0, label="truth")
        axR.scatter([r_true_xy[-1, 0]], [r_true_xy[-1, 1]],
                    s=40, c=_CYAN, zorder=5)
    if r_est_xy.shape[0] >= 1 and np.all(np.isfinite(r_est_xy[-1])):
        axR.plot(r_est_xy[:, 0], r_est_xy[:, 1],
                 color=_AMBER, lw=2.0, ls=(0, (5, 3)), label="EKF est", alpha=0.9)
        axR.scatter([r_est_xy[-1, 0]], [r_est_xy[-1, 1]],
                    s=40, c=_AMBER, marker="D", zorder=5)

    axR.set_title("XY Orbit (near L1)", fontsize=10)
    axR.set_xlabel("x [dimensionless CR3BP length]", fontsize=9)
    axR.set_ylabel("y [dimensionless CR3BP length]", fontsize=9)
    axR.legend(loc="lower left", fontsize=9)

    pts = np.vstack([r_true_xy, r_est_xy])
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] >= 2:
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        ctr  = 0.5 * (mn + mx)
        span = float(np.max(mx - mn))
        span = max(span, 1e-5)
        pad  = 0.25 * span
        axR.set_xlim(ctr[0] - 0.5 * span - pad, ctr[0] + 0.5 * span + pad)
        axR.set_ylim(ctr[1] - 0.5 * span - pad, ctr[1] + 0.5 * span + pad)

    fig.savefig(out_path, dpi=dpi, facecolor=_BG)
    plt.close(fig)


def main() -> None:
    videos_dir = Path("results/videos"); videos_dir.mkdir(parents=True, exist_ok=True)
    frames_dir      = videos_dir / "05_cam_frames"
    frames_split_dir = videos_dir / "05_cam_frames_split"
    for d in (frames_dir, frames_split_dir):
        d.mkdir(parents=True, exist_ok=True)

    mu    = 0.0121505856
    model = CR3BP(mu=mu)

    L1x    = model.lagrange_points()["L1"][0]
    x0_nom = np.array([L1x - 1.2e-3, 2.5e-4, 0.0, 0.0, 0.045, 0.0], dtype=float)
    r_body = np.asarray(model.primary2, dtype=float).reshape(3)

    fps      = 30
    SLOW_MO  = 5.0
    dt_frame = (1.0 / fps) / SLOW_MO
    t0, tf   = 0.0, 14.0
    t_frame  = np.arange(t0, tf + 1e-12, dt_frame)
    N        = len(t_frame)

    width, height = 1280, 720
    intr = Intrinsics(fx=480., fy=480., cx=width / 2, cy=height / 2,
                      width=width, height=height)
    body_radius = 0.0045

    sigma_px               = 1.8
    dropout_prob           = 0.05
    centroid_bias_fraction = 0.06
    OVERLAY_TEXT           = True
    DOT_RADIUS_PX          = 3.5
    RUN_EKF                = True
    q_acc                  = 3e-10
    P0 = np.diag([2e-5] * 6).astype(float)

    rng = np.random.default_rng(7)

    res_truth = propagate(model.eom, (t0, tf), x0_nom, t_eval=t_frame)
    if not res_truth.success:
        raise RuntimeError(f"Truth propagation failed: {res_truth.message}")
    X_true = res_truth.x
    r_true = X_true[:, :3]

    u_px      = np.full(N, np.nan)
    v_px      = np.full(N, np.nan)
    valid     = np.zeros(N, dtype=bool)
    bbox_xyxy = [None] * N
    R_cam_arr = np.zeros((N, 3, 3), dtype=float)

    for k in range(N):
        R_cam_arr[k] = camera_dcm_from_boresight(
            r_body - r_true[k], camera_forward_axis="+z"
        )
        meas = simulate_bbox_measurement(
            r_sc=r_true[k], r_body=r_body,
            body_radius=body_radius, intrinsics=intr,
            R_cam_from_frame=R_cam_arr[k],
            sigma_px=sigma_px, rng=rng, t=float(t_frame[k]),
            dropout_p=dropout_prob, out_of_frame="drop", behind="drop",
            centroid_bias_fraction=centroid_bias_fraction,
        )
        if meas.valid:
            u_px[k]      = float(meas.u_px)
            v_px[k]      = float(meas.v_px)
            valid[k]     = True
            bbox_xyxy[k] = meas.bbox_xyxy

    X_hat_hist    = np.full((N, 6), np.nan, dtype=float)
    upd_used_hist = np.zeros(N, dtype=bool)

    if RUN_EKF:
        x_hat  = x0_nom.copy()
        x_hat[:3] += np.array([3e-4, -2e-4, 0.0])
        x_hat[3:6] += np.array([0.0, 2e-4, 0.0])
        P      = P0.copy()
        t_prev = float(t_frame[0])

        for k in range(N):
            t_k = float(t_frame[k])
            if k > 0:
                x_hat, P, _ = ekf_propagate_cr3bp_stm(
                    mu=mu, x=x_hat, P=P, t0=t_prev, t1=t_k, q_acc=q_acc
                )
            t_prev = t_k

            if valid[k]:
                u_meas, sigma_theta = pixel_detection_to_bearing(
                    u_px[k], v_px[k], sigma_px, intr, R_cam_arr[k].T
                )
                if np.all(np.isfinite(u_meas)):
                    upd = bearing_update_tangent(x_hat, P, u_meas, r_body,
                                                 float(sigma_theta))
                    if upd.accepted:
                        x_hat, P = upd.x_upd, upd.P_upd
                        upd_used_hist[k] = True

            X_hat_hist[k] = x_hat

    print(f"Rendering {N} frames ...")
    for k in range(N):
        t_k      = float(t_frame[k])
        is_valid = bool(valid[k])
        up       = float(u_px[k]) if is_valid else None
        vp       = float(v_px[k]) if is_valid else None
        bb       = bbox_xyxy[k]   if is_valid else None
        upd_k    = bool(upd_used_hist[k])

        _render_camera_frame(
            out_path=frames_dir / f"frame_{k:05d}.png",
            width=width, height=height, t=t_k,
            valid=is_valid, u_px=up, v_px=vp, bbox_xyxy=bb,
            update_used=upd_k, overlay_text=OVERLAY_TEXT,
            dot_radius_px=DOT_RADIUS_PX,
        )

        true_xy = r_true[: k + 1, :2]
        est_xy  = (X_hat_hist[: k + 1, :2]
                   if RUN_EKF and np.all(np.isfinite(X_hat_hist[k, :2]))
                   else np.full_like(true_xy, np.nan))

        _render_split_frame(
            out_path=frames_split_dir / f"frame_{k:05d}.png",
            width_left=width, height_left=height, t=t_k,
            valid=is_valid, u_px=up, v_px=vp, bbox_xyxy=bb,
            update_used=upd_k, overlay_text=OVERLAY_TEXT,
            dot_radius_px=DOT_RADIUS_PX,
            r_true_xy=true_xy, r_est_xy=est_xy,
        )

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")

    out_mp4       = videos_dir / "05_camera_feed.mp4"
    out_mp4_split = videos_dir / "05_camera_feed_split.mp4"
    _run_ffmpeg(frames_dir,       out_mp4,       fps=fps)
    _run_ffmpeg(frames_split_dir, out_mp4_split, fps=fps)

    print("Wrote videos:")
    print(f"  {out_mp4}")
    print(f"  {out_mp4_split}")


if __name__ == "__main__":
    main()
