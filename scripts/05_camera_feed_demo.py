from __future__ import annotations

from pathlib import Path
import subprocess
import shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from dynamics.cr3bp import CR3BP
from dynamics.integrators import propagate

from nav.ekf import ekf_propagate_cr3bp_stm
from nav.measurements.bearing import bearing_update_tangent
from nav.measurements.pixel_bearing import pixel_detection_to_bearing

from cv.camera import Intrinsics
from cv.sim_measurements import simulate_bbox_measurement


def _run_ffmpeg(frames_dir: Path, out_mp4: Path, fps: int) -> None:
    """
    Create MP4 from frame_%05d.png using ffmpeg.
    Uses yuv420p and pads to even width/height for compatibility.
    """
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
    subprocess.run(cmd, check=True)


def _style_dark_axes(ax):
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.25))
    ax.grid(True, alpha=0.12)


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
    overlay_text: bool,
    dot_radius_px: float,
) -> None:
    """
    Synthetic "camera feed" frame:
      - black background image plane
      - dot at (u_px, v_px) if valid
      - bbox rectangle if bbox_xyxy provided
      - overlay text (t, u/v, valid)
    Coordinate convention: imshow(origin="upper") => y increases downward.
    """
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])

    blank = np.zeros((height, width), dtype=np.float32)
    ax.imshow(blank, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
    ax.set_axis_off()

    if valid and (u_px is not None) and (v_px is not None):
        ax.add_patch(
            plt.Circle((u_px, v_px), radius=dot_radius_px, fill=True, linewidth=0.0, color="white")
        )

        if bbox_xyxy is not None:
            xmin, ymin, xmax, ymax = bbox_xyxy
            ax.add_patch(
                Rectangle(
                    (xmin, ymin),
                    max(1.0, xmax - xmin),
                    max(1.0, ymax - ymin),
                    fill=False,
                    linewidth=1.6,
                    edgecolor=(1, 1, 1, 0.85),
                )
            )

    if overlay_text:
        if valid and (u_px is not None) and (v_px is not None):
            uv_str = f"u,v = ({u_px:7.2f}, {v_px:7.2f})"
        else:
            uv_str = "u,v = (   nan,    nan)"
        txt = f"t = {t:7.3f}\n{uv_str}\nvalid = {bool(valid)}"
        ax.text(
            14,
            22,
            txt,
            color="white",
            fontsize=12,
            family="monospace",
            va="top",
            ha="left",
            bbox=dict(facecolor=(0, 0, 0, 0.35), edgecolor=(1, 1, 1, 0.15), pad=6),
        )

    fig.savefig(out_path, dpi=dpi)
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
    overlay_text: bool,
    dot_radius_px: float,
    r_true_xy: np.ndarray,   # (k+1,2)
    r_est_xy: np.ndarray,    # (k+1,2)
) -> None:
    """
    Split-screen:
      left: camera frame
      right: XY trajectory (true + estimated) with current point highlighted
    """
    dpi = 100
    right_w = int(0.95 * height_left)
    total_w = width_left + right_w
    total_h = height_left

    fig = plt.figure(figsize=(total_w / dpi, total_h / dpi), dpi=dpi, facecolor="black")

    # Left axis
    axL = fig.add_axes([0.0, 0.0, width_left / total_w, 1.0])
    blank = np.zeros((height_left, width_left), dtype=np.float32)
    axL.imshow(blank, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
    axL.set_axis_off()

    if valid and (u_px is not None) and (v_px is not None):
        axL.add_patch(
            plt.Circle((u_px, v_px), radius=dot_radius_px, fill=True, linewidth=0.0, color="white")
        )
        if bbox_xyxy is not None:
            xmin, ymin, xmax, ymax = bbox_xyxy
            axL.add_patch(
                Rectangle(
                    (xmin, ymin),
                    max(1.0, xmax - xmin),
                    max(1.0, ymax - ymin),
                    fill=False,
                    linewidth=1.6,
                    edgecolor=(1, 1, 1, 0.85),
                )
            )

    if overlay_text:
        if valid and (u_px is not None) and (v_px is not None):
            uv_str = f"u,v = ({u_px:7.2f}, {v_px:7.2f})"
        else:
            uv_str = "u,v = (   nan,    nan)"
        txt = f"t = {t:7.3f}\n{uv_str}\nvalid = {bool(valid)}"
        axL.text(
            14,
            22,
            txt,
            color="white",
            fontsize=12,
            family="monospace",
            va="top",
            ha="left",
            bbox=dict(facecolor=(0, 0, 0, 0.35), edgecolor=(1, 1, 1, 0.15), pad=6),
        )

    # Right axis
    axR = fig.add_axes([width_left / total_w, 0.0, right_w / total_w, 1.0])
    _style_dark_axes(axR)

    if r_true_xy.shape[0] >= 1:
        axR.plot(r_true_xy[:, 0], r_true_xy[:, 1], linewidth=2.0, label="true")
        axR.scatter([r_true_xy[-1, 0]], [r_true_xy[-1, 1]], s=35, marker="o", label="true now")

    if r_est_xy.shape[0] >= 1 and np.all(np.isfinite(r_est_xy[-1])):
        axR.plot(r_est_xy[:, 0], r_est_xy[:, 1], linestyle=(0, (6, 4)), linewidth=2.0, label="EKF est")
        axR.scatter([r_est_xy[-1, 0]], [r_est_xy[-1, 1]], s=35, marker="o", label="est now")

    axR.set_title("XY trajectory (truth + estimate)")
    axR.set_xlabel("x")
    axR.set_ylabel("y")
    axR.legend(loc="upper left", fontsize=9)

    pts = np.vstack([r_true_xy, r_est_xy])
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] >= 2:
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        ctr = 0.5 * (mins + maxs)
        span = float(np.max(maxs - mins))
        span = max(span, 1e-6)
        pad = 0.30 * span
        axR.set_xlim(ctr[0] - 0.5 * span - pad, ctr[0] + 0.5 * span + pad)
        axR.set_ylim(ctr[1] - 0.5 * span - pad, ctr[1] + 0.5 * span + pad)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    videos_dir = Path("results/videos")
    videos_dir.mkdir(parents=True, exist_ok=True)

    out_mp4 = videos_dir / "05_camera_feed.mp4"
    out_mp4_split = videos_dir / "05_camera_feed_split.mp4"

    frames_dir = videos_dir / "05_camera_frames"
    frames_split_dir = videos_dir / "05_camera_frames_split"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames_split_dir.mkdir(parents=True, exist_ok=True)

    mu = 0.0121505856
    model = CR3BP(mu=mu)


    fps = 30
    SLOW_MO = 5.0  # 5x slower physical motion
    dt_frame = (1.0 / fps) / SLOW_MO

    t0, tf = 0.0, 14.0
    t_frame = np.arange(t0, tf + 1e-12, dt_frame)
    N = len(t_frame)

    L = model.lagrange_points()
    L1 = L["L1"]
    x_true0 = np.array(
        [L1[0] - 1.2e-3, 2.5e-4, 0.0, 0.0, 0.045, 0.0],
        dtype=float
    )

    r_body = np.asarray(model.primary2, dtype=float).reshape(3)

    body_radius = 0.0045

    width, height = 1280, 720
    intr = Intrinsics(
        fx=480.0, fy=480.0,  
        cx=width / 2, cy=height / 2,
        width=width, height=height
    )


    R_cam_from_frame = np.array([
        [0.0, 1.0, 0.0],  # x_cam <- y_global
        [0.0, 0.0, 1.0],  # y_cam <- z_global
        [1.0, 0.0, 0.0],  # z_cam <- x_global (depth)
    ], dtype=float)
    R_frame_from_cam = R_cam_from_frame.T


    sigma_px = 1.8
    dropout_prob = 0.05
    centroid_bias_fraction = 0.06  # deliberate mismatch

    # Visual options (all on)
    OVERLAY_TEXT = True
    DOT_RADIUS_PX = 3.5

    # EKF options:
    # We keep mismatch, but we prevent catastrophic end divergence by:
    # - a bit more process noise
    # - a bit larger initial covariance
    RUN_EKF = True
    q_acc = 3e-10
    P0 = np.diag([2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5]).astype(float)

    rng = np.random.default_rng(7)

    res_truth = propagate(model.eom, (t0, tf), x_true0, t_eval=t_frame)
    if not res_truth.success:
        raise RuntimeError(f"Truth propagation failed: {res_truth.message}")

    X_true = res_truth.x
    r_true = X_true[:, :3]

    u_px = np.full(N, np.nan)
    v_px = np.full(N, np.nan)
    valid = np.zeros(N, dtype=bool)
    bbox_xyxy = [None] * N

    for k in range(N):
        meas = simulate_bbox_measurement(
            r_sc=r_true[k],
            r_body=r_body,
            body_radius=body_radius,
            intrinsics=intr,
            R_cam_from_frame=R_cam_from_frame,
            sigma_px=sigma_px,
            rng=rng,
            t=float(t_frame[k]),
            dropout_p=dropout_prob,
            out_of_frame="drop",
            behind="drop",
            centroid_bias_fraction=centroid_bias_fraction,
        )

        if meas.valid:
            u_px[k] = float(meas.u_px)
            v_px[k] = float(meas.v_px)
            valid[k] = True
            bbox_xyxy[k] = meas.bbox_xyxy  # (xmin, ymin, xmax, ymax)

    X_hat_hist = np.full((N, 6), np.nan, dtype=float)
    if RUN_EKF:
        x_hat = x_true0.copy()
        x_hat[:3] += np.array([3e-4, -2e-4, 0.0])
        x_hat[3:6] += np.array([0.0, 2e-4, 0.0])
        P = P0.copy()

        t_prev = float(t_frame[0])
        for k in range(N):
            t_k = float(t_frame[k])
            if k > 0:
                x_hat, P, _Phi = ekf_propagate_cr3bp_stm(
                    mu=mu, x=x_hat, P=P, t0=t_prev, t1=t_k, q_acc=q_acc
                )
            t_prev = t_k

            # Update only when we have a detection
            if valid[k]:
                u_meas_global, sigma_theta = pixel_detection_to_bearing(
                    u_px[k], v_px[k], sigma_px, intr, R_frame_from_cam
                )
                x_hat, P, _y, _nis = bearing_update_tangent(
                    x_hat, P, u_meas_global, r_body, sigma_theta
                )

            X_hat_hist[k] = x_hat

    for k in range(N):
        t_k = float(t_frame[k])
        is_valid = bool(valid[k])
        up = None if not is_valid else float(u_px[k])
        vp = None if not is_valid else float(v_px[k])
        bb = None if not is_valid else bbox_xyxy[k]

        _render_camera_frame(
            out_path=frames_dir / f"frame_{k:05d}.png",
            width=width,
            height=height,
            t=t_k,
            valid=is_valid,
            u_px=up,
            v_px=vp,
            bbox_xyxy=bb,
            overlay_text=OVERLAY_TEXT,
            dot_radius_px=DOT_RADIUS_PX,
        )

        true_xy = r_true[: k + 1, :2]
        if RUN_EKF and np.all(np.isfinite(X_hat_hist[k, :2])):
            est_xy = X_hat_hist[: k + 1, :2]
        else:
            est_xy = np.full_like(true_xy, np.nan)

        _render_split_frame(
            out_path=frames_split_dir / f"frame_{k:05d}.png",
            width_left=width,
            height_left=height,
            t=t_k,
            valid=is_valid,
            u_px=up,
            v_px=vp,
            bbox_xyxy=bb,
            overlay_text=OVERLAY_TEXT,
            dot_radius_px=DOT_RADIUS_PX,
            r_true_xy=true_xy,
            r_est_xy=est_xy,
        )

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; required to write mp4 automatically.")

    _run_ffmpeg(frames_dir, out_mp4, fps=fps)
    _run_ffmpeg(frames_split_dir, out_mp4_split, fps=fps)

    print("Wrote videos:")
    print(" -", out_mp4)
    print(" -", out_mp4_split)
    print("Frames kept in:")
    print(" -", frames_dir)
    print(" -", frames_split_dir)


if __name__ == "__main__":
    main()