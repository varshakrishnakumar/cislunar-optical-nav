from __future__ import annotations

from pathlib import Path
import subprocess
import shutil
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from dynamics.cr3bp import CR3BP
from dynamics.integrators import propagate

from nav.ekf import ekf_propagate_cr3bp_stm
from nav.measurements.bearing import bearing_update_tangent
from nav.measurements.pixel_bearing import pixel_detection_to_bearing

from cv.camera import Intrinsics
from cv.sim_measurements import simulate_pixel_measurement

def _set_tracking_bounds(ax, r_true, v_true, X_hat_hist, k,
                         window=200, lead=0.10, pad=1.35, err_gain=8.0,
                         view_min=1.0e-3, view_max=0.08, smooth_alpha=0.985,
                         q=0.90, state=None):

    i0 = max(0, k - window)
    pts = np.vstack([r_true[i0:k+1], X_hat_hist[i0:k+1, :3]])

    v = v_true[k]
    speed = float(np.linalg.norm(v))
    lead_vec = lead * v if speed > 1e-12 else 0.0
    center_target = r_true[k] + lead_vec

    d = np.linalg.norm(pts - center_target[None, :], axis=1)
    rad = float(np.quantile(d, q))

    err = float(np.linalg.norm(X_hat_hist[k, :3] - r_true[k]))
    half_target = pad * rad + err_gain * err
    half_target = float(np.clip(half_target, view_min, view_max))

    if state is None:
        state = {"center": center_target.copy(), "half": half_target}
    else:
        state["center"] = smooth_alpha * state["center"] + (1 - smooth_alpha) * center_target
        state["half"]   = smooth_alpha * state["half"]   + (1 - smooth_alpha) * half_target

    c = state["center"]
    d_true = float(np.max(np.abs(r_true[k] - c)))
    d_est  = float(np.max(np.abs(X_hat_hist[k, :3] - c)))
    need_half = max(d_true, d_est) + 1.5 * view_min
    state["half"] = float(np.clip(max(state["half"], need_half), view_min, view_max))

    h = state["half"]
    ax.set_xlim(c[0]-h, c[0]+h)
    ax.set_ylim(c[1]-h, c[1]+h)
    ax.set_zlim(c[2]-h, c[2]+h)
    return state

def _style_3d_space(ax):
    ax.set_facecolor("black")
    ax.grid(False)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((0, 0, 0, 1))
            axis.pane.set_edgecolor((0, 0, 0, 0))
        except Exception:
            pass
        try:
            axis._axinfo["grid"]["color"] = (0, 0, 0, 0)
        except Exception:
            pass

    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.zaxis.label.set_color("white")
    ax.title.set_color("white")

    ax.set_axis_off()
    
def _blend_angle_deg(a, b, w):
    d = (b - a + 180.0) % 360.0 - 180.0
    return (a + w * d) % 360.0

def _style_2d_space(fig):
    """Dark theme for 2D plots."""
    fig.patch.set_facecolor("black")
    for ax in fig.axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color((1, 1, 1, 0.35))
        ax.grid(True, alpha=0.15)


def _legend_dark(ax):
    leg = ax.get_legend()
    if leg is None:
        leg = ax.legend(loc="upper left", fontsize=9)

    frame = leg.get_frame()
    frame.set_facecolor((0, 0, 0, 0.25))   
    frame.set_edgecolor((1, 1, 1, 0.18))  
    frame.set_linewidth(0.8)
    leg.borderpad = 0.35
    leg.labelspacing = 0.35
    leg.handlelength = 2.0
    leg.handletextpad = 0.6

    for t in leg.get_texts():
        t.set_color((1, 1, 1, 0.92))


def _make_starfield(rng: np.random.Generator, center: np.ndarray, radius: float, n: int = 2500):
    """
    Stars distributed on a sphere shell around the scene.
    Returns positions (n,3), sizes (n,), colors_rgba (n,4).
    """
    dirs = rng.normal(size=(n, 3))
    dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)

    rr = radius * (0.85 + 0.30 * rng.random(n))
    stars = center + dirs * rr[:, None]

    bright = rng.random(n) ** 3

    alpha = 0.03 + 0.60 * bright
    sizes = 0.8 + 7.5 * bright

    types = rng.choice(3, size=n, p=[0.22, 0.60, 0.18])

    cool    = np.array([0.82, 0.90, 1.00])
    neutral = np.array([1.00, 1.00, 1.00])
    warm    = np.array([1.00, 0.93, 0.82])

    base_rgb = np.zeros((n, 3), dtype=float)
    base_rgb[types == 0] = cool
    base_rgb[types == 1] = neutral
    base_rgb[types == 2] = warm

    whiten = (bright ** 0.35)[:, None]  # 0..1
    rgb = (1.0 - 0.35 * whiten) * base_rgb + (0.35 * whiten) * np.array([1.0, 1.0, 1.0])

    colors = np.zeros((n, 4), dtype=float)
    colors[:, :3] = np.clip(rgb, 0.0, 1.0)
    colors[:, 3] = np.clip(alpha, 0.0, 1.0)

    return stars, sizes, colors


def _run_ffmpeg(frames_dir: Path, out_mp4: Path, fps: int = 30) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    plots_dir = Path("results/plots"); plots_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = Path("results/videos"); videos_dir.mkdir(parents=True, exist_ok=True)

    mu = 0.0121505856
    model = CR3BP(mu=mu)

    t0, tf = 0.0, 6.0
    dt_meas = 0.02
    t_meas = np.arange(t0, tf + 1e-12, dt_meas)

    L = model.lagrange_points()
    L1 = L["L1"]

    x_true0 = np.array([L1[0] - 1e-3, 0.0, 0.0, 0.0, 0.05, 0.0], dtype=float)

    r_body = np.asarray(model.primary2, dtype=float).reshape(3)

    width, height = 1280, 720
    intr = Intrinsics(fx=800.0, fy=800.0, cx=width / 2, cy=height / 2, width=width, height=height)

    R_cam_from_frame = np.eye(3)
    R_frame_from_cam = np.eye(3)

    sigma_px = 1.5
    dropout_prob = 0.0
    q_acc = 1e-12

    x_hat = x_true0.copy()
    x_hat[:3] += np.array([2e-4, -1e-4, 0.0])
    x_hat[3:6] += np.array([0.0, 2e-4, 0.0])
    P = np.diag([1e-6] * 6).astype(float)

    rng = np.random.default_rng(7)

    res_truth = propagate(model.eom, (t0, tf), x_true0, t_eval=t_meas)
    if not res_truth.success:
        raise RuntimeError(f"Truth propagation failed: {res_truth.message}")

    X_true = res_truth.x
    r_true = X_true[:, :3]
    v_true = X_true[:, 3:6]
    N = len(t_meas)

    u_px = np.full(N, np.nan)
    v_px = np.full(N, np.nan)
    valid = np.zeros(N, dtype=bool)
    R_cam_from_frame_hist = np.zeros((N, 3, 3), dtype=float)

    for k in range(N):
        los = r_body - r_true[k]
        los /= np.linalg.norm(los)

        z_cam = los
        x_tmp = np.array([1,0,0])
        if abs(np.dot(z_cam, x_tmp)) > 0.9:
            x_tmp = np.array([0,1,0])

        x_cam = np.cross(x_tmp, z_cam)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)

        R_cam_from_frame = np.vstack([x_cam, y_cam, z_cam])
        R_cam_from_frame_hist[k] = R_cam_from_frame
        meas = simulate_pixel_measurement(
            r_sc=r_true[k],
            r_body=r_body,
            intrinsics=intr,
            R_cam_from_frame=R_cam_from_frame,
            sigma_px=sigma_px,
            rng=rng,
            t=float(t_meas[k]),
            dropout_p=dropout_prob,
            out_of_frame="drop",
            behind="drop",
        )
        if meas.valid:
            u_px[k] = meas.u_px
            v_px[k] = meas.v_px
            valid[k] = True

    X_hat_hist = np.zeros((N, 6), dtype=float)
    pos_err = np.full(N, np.nan)
    vel_err = np.full(N, np.nan)
    nis_hist = np.full(N, np.nan)

    fps = 30
    stride = 3  # save every Nth point
    traj_frames = videos_dir / "04_pixel_bearings_frames_traj"; traj_frames.mkdir(parents=True, exist_ok=True)
    err_frames  = videos_dir / "04_pixel_bearings_frames_err";  err_frames.mkdir(parents=True, exist_ok=True)

    mins = r_true.min(axis=0)
    maxs = r_true.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    span = max(span, 1e-3)

    zoom = 0.70
    box = zoom * span

    # Starfield
    stars_rng = np.random.default_rng(123)
    stars, star_sizes, star_colors = _make_starfield(stars_rng, center=center, radius=6.0 * span, n=3500)

    p1 = np.asarray(model.primary1, dtype=float).reshape(3)
    p2 = np.asarray(model.primary2, dtype=float).reshape(3)

    t_prev = float(t_meas[0])
    frame_id = 0
    
    cam_state = None
    yaw_state = None
    roll_state = 0.0
    

    for k in range(N):
        t_k = float(t_meas[k])

        if k > 0:
            x_hat, P, _Phi = ekf_propagate_cr3bp_stm(mu=mu, x=x_hat, P=P, t0=t_prev, t1=t_k, q_acc=q_acc)
        t_prev = t_k

        # if valid[k]:
        #     u_meas_global, sigma_theta = pixel_detection_to_bearing(u_px[k], v_px[k], sigma_px, intr, R_frame_from_cam)
        #     x_hat, P, y, nis = bearing_update_tangent(x_hat, P, u_meas_global, r_body, sigma_theta)
        #     nis_hist[k] = float(nis) if np.isfinite(nis) else np.nan
        if valid[k]:
            # Use the SAME attitude that produced the pixel measurement
            R_frame_from_cam = R_cam_from_frame_hist[k].T

            u_meas_global, sigma_theta = pixel_detection_to_bearing(
                u_px[k], v_px[k], sigma_px, intr, R_frame_from_cam
            )
            x_hat, P, y, nis = bearing_update_tangent(x_hat, P, u_meas_global, r_body, sigma_theta)
            nis_hist[k] = float(nis) if np.isfinite(nis) else np.nan

        X_hat_hist[k] = x_hat
        pos_err[k] = float(np.linalg.norm(x_hat[:3] - r_true[k]))
        vel_err[k] = float(np.linalg.norm(x_hat[3:6] - v_true[k]))

        if (k % stride != 0) and (k != N - 1):
            continue
        fig = plt.figure(figsize=(11, 7), facecolor="black")
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        C_TRUE       = "#BFEFFF"
        C_EST        = "#FFB454"
        C_NOW_TRUE   = "#FF6E6E"
        C_NOW_EST    = "#D7A6FF"
        C_EARTH_CORE = "#2F7DFF"
        C_EARTH_GLOW = "#7FB6FF"
        C_MOON_CORE  = "#E6E7EA"
        C_MOON_GLOW  = "#C9CED6"
        C_CONN       = "#FFFFFF"
        ax.set_box_aspect((1, 1, 1))  
        ax.scatter(stars[:, 0], stars[:, 1], stars[:, 2],
           s=star_sizes, c=star_colors, linewidths=0)

        ax.scatter([p1[0]], [p1[1]], [p1[2]], s=520, c=C_EARTH_GLOW, alpha=0.06, linewidths=0)
        ax.scatter([p1[0]], [p1[1]], [p1[2]], s=90,  c=C_EARTH_CORE, alpha=0.95, linewidths=0, label="Earth")

        ax.scatter([p2[0]], [p2[1]], [p2[2]], s=420, c=C_MOON_GLOW, alpha=0.05, linewidths=0)
        ax.scatter([p2[0]], [p2[1]], [p2[2]], s=70,  c=C_MOON_CORE, alpha=0.95, linewidths=0, label="Moon")


        trail = 300
        i_trail = max(0, k - trail)
        
        ax.plot(r_true[i_trail:k+1, 0], r_true[i_trail:k+1, 1], r_true[i_trail:k+1, 2],
                linewidth=6.0, alpha=0.10, color=C_TRUE)
        ax.plot(r_true[i_trail:k+1, 0], r_true[i_trail:k+1, 1], r_true[i_trail:k+1, 2],
                linewidth=2.6, alpha=0.95, color=C_TRUE, label="True")

        ax.plot(X_hat_hist[i_trail:k+1, 0], X_hat_hist[i_trail:k+1, 1], X_hat_hist[i_trail:k+1, 2],
                linewidth=6.0, alpha=0.08, color=C_EST)
        ax.plot(X_hat_hist[i_trail:k+1, 0], X_hat_hist[i_trail:k+1, 1], X_hat_hist[i_trail:k+1, 2],
                linestyle=(0, (6, 4)), linewidth=2.6, alpha=0.95, color=C_EST, label="EKF est")
        
        err = float(np.linalg.norm(x_hat[:3] - r_true[k]))
        conn_alpha = min(0.95, 0.25 + 40.0*err)
        conn_lw = float(np.clip(1.5 + 30.0*err, 1.5, 6.0))
        ax.plot([r_true[k,0], x_hat[0]],
            [r_true[k,1], x_hat[1]],
            [r_true[k,2], x_hat[2]],
            linewidth=conn_lw, alpha=min(0.85, conn_alpha), color=C_CONN)
        
        ax.scatter([r_true[k,0]], [r_true[k,1]], [r_true[k,2]],
           s=60, marker="o", c=C_NOW_TRUE,
           edgecolors=(1,1,1,0.85), linewidths=1.1,
           depthshade=False, label="True now")

        ax.scatter([x_hat[0]], [x_hat[1]], [x_hat[2]],
                s=60, marker="o", c=C_NOW_EST,
                edgecolors=(1,1,1,0.85), linewidths=1.1,
                depthshade=False, label="Est now")
        
        info = Line2D([], [], linestyle="none", marker=None, color="none",
                    label=f"t={t_k:.2f}  Δt={dt_meas:.2f}  ||r̂−r||={err:.2e}")

        handles, labels = ax.get_legend_handles_labels()
        handles.append(info)
        labels.append(info.get_label())
        ax.legend(handles, labels, loc="upper left", fontsize=10)
        _legend_dark(ax)


        ax.set_title(f"Cislunar-style 3D View (t={t_k:.2f})")
        
        cam_state = _set_tracking_bounds(
            ax, r_true, v_true, X_hat_hist, k,
            window=trail,
            lead=0.09,
            pad=1.45,         
            err_gain=9.0,    
            view_max=0.08,
            view_min=2.0e-3,
            smooth_alpha=0.97,
            q=0.90,
            state=cam_state
        )
        
        t_video = t_k 

        az_orbit = 35.0 + 4.0 * t_video
        el = 18.0 + 2.0 * np.sin(0.7 * t_video)

        vx, vy = float(v_true[k, 0]), float(v_true[k, 1])
        vnorm = float(np.linalg.norm(v_true[k]))
        if vnorm < 1e-6:
            heading_cam = az_orbit  # fallback: keep orbit
        else:
            heading_deg = np.degrees(np.arctan2(vy, vx))
            behind_deg = 182.0  
            heading_cam = (heading_deg + behind_deg) % 360.0

        follow = 0.45  
        follow = float(np.clip(follow, 0.0, 1.0))

        az_target = _blend_angle_deg(az_orbit, heading_cam, follow)

        yaw_alpha = 0.98  # closer to 1 => smoother / slower following
        if yaw_state is None:
            yaw_state = az_target
        else:
            # move yaw_state a small step toward az_target
            yaw_state = _blend_angle_deg(yaw_state, az_target, 1.0 - yaw_alpha)

        az = yaw_state  
        if 0 < k < N - 1:
            a = (v_true[k + 1] - v_true[k - 1]) / (2.0 * dt_meas)
        elif k == 0:
            a = (v_true[1] - v_true[0]) / dt_meas
        else:
            a = (v_true[-1] - v_true[-2]) / dt_meas

        v = v_true[k]
        vnorm = float(np.linalg.norm(v))
        eps = 1e-12

        turn_rate = float(np.linalg.norm(np.cross(v, a))) / (vnorm * vnorm + eps)

        bank_gain = 35.0      # tune: 20–60 (larger => more roll)
        bank_max_deg = 10.0   # cap roll magnitude
        roll_target = float(np.clip(bank_gain * turn_rate, -bank_max_deg, bank_max_deg))

        # Smooth roll
        roll_alpha = 0.96
        roll_state = float(roll_alpha * roll_state + (1.0 - roll_alpha) * roll_target)

        try:
            ax.view_init(elev=el, azim=az, roll=roll_state)
        except TypeError:
            ax.view_init(elev=el, azim=az)

        _style_3d_space(ax)


        fig.tight_layout(pad=0.0)
        fig.savefig(traj_frames / f"frame_{frame_id:05d}.png", dpi=170)
        plt.close(fig)

        fig2 = plt.figure(figsize=(11, 7), facecolor="black")
        ax1 = fig2.add_subplot(3, 1, 1)
        ax2 = fig2.add_subplot(3, 1, 2)
        ax3 = fig2.add_subplot(3, 1, 3)

        ax1.plot(t_meas[:k + 1], pos_err[:k + 1], linewidth=2.0, label="pos_err")
        ax1.set_ylabel("||r̂-r||")
        ax1.legend(loc="upper left")

        ax2.plot(t_meas[:k + 1], vel_err[:k + 1], linewidth=2.0, label="vel_err")
        ax2.set_ylabel("||v̂-v||")
        ax2.legend(loc="upper left")

        ax3.plot(t_meas[:k + 1], nis_hist[:k + 1], linewidth=2.0, label="NIS")
        ax3.set_xlabel("t")
        ax3.set_ylabel("NIS")
        ax3.legend(loc="upper left")

        fig2.suptitle(f"Filter Diagnostics (t={t_k:.2f})", color="white")

        _style_2d_space(fig2)
        fig2.tight_layout()
        fig2.savefig(err_frames / f"frame_{frame_id:05d}.png", dpi=170)
        plt.close(fig2)

        frame_id += 1

    # --- Final static error summary plot ---
    fig = plt.figure(figsize=(11, 7))

    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    ax1.plot(t_meas, pos_err, linewidth=2.0)
    ax1.set_ylabel("||r̂ - r||")
    ax1.grid(True)

    ax2.plot(t_meas, vel_err, linewidth=2.0)
    ax2.set_ylabel("||v̂ - v||")
    ax2.grid(True)

    ax3.plot(t_meas, nis_hist, linewidth=2.0)
    ax3.set_xlabel("t")
    ax3.set_ylabel("NIS")
    ax3.grid(True)

    fig.suptitle("04 Pixel Bearings — Filter Errors")
    fig.tight_layout()

    fig.savefig(plots_dir / "04_pixel_bearings_errors.png", dpi=200)

    upper95 = stats.chi2.ppf(0.95, 2)
    ax3.axhline(upper95, linestyle='--', linewidth=1.2, label="95%")
    ax3.legend()
    plt.close(fig)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; required to write mp4 automatically.")

    traj_mp4 = videos_dir / "04_pixel_bearings_traj.mp4"
    err_mp4 = videos_dir / "04_pixel_bearings_error.mp4"
    fps_out = fps / stride
    _run_ffmpeg(traj_frames, traj_mp4, fps=int(round(fps_out)))
    _run_ffmpeg(err_frames,  err_mp4,  fps=int(round(fps_out)))


    print("Wrote videos:")
    print(" -", traj_mp4)
    print(" -", err_mp4)
    print("Frames kept in:")
    print(" -", traj_frames)
    print(" -", err_frames)


if __name__ == "__main__":
    main()