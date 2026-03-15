from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Repo imports (your project layout)
from dynamics.cr3bp import CR3BP, propagate
from nav.ekf import ekf_propagate_cr3bp_stm
from nav.measurements.bearing import bearing_update_tangent
from nav.measurements.pixel_bearing import pixel_detection_to_bearing
from cv.sim_measurements import simulate_pixel_measurement
from cv.camera import CameraIntrinsics

# From 02 (your targeting solver)
from guidance.targeting import solve_single_impulse_position_target


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def _nearest_index(t_grid: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(t_grid - t)))


def _plot_traj(
    xs_nom: np.ndarray,
    xs_true_unc: np.ndarray,
    xs_true_perf: np.ndarray,
    xs_true_ekf: np.ndarray,
    outpath: Path,
) -> None:
    plt.figure()
    plt.plot(xs_nom[:, 0], xs_nom[:, 1], label="nominal")
    plt.plot(xs_true_unc[:, 0], xs_true_unc[:, 1], label="truth uncorrected")
    plt.plot(xs_true_perf[:, 0], xs_true_perf[:, 1], label="truth corrected (perfect)")
    plt.plot(xs_true_ekf[:, 0], xs_true_ekf[:, 1], label="truth corrected (EKF)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Midcourse correction trajectories")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _plot_pos_err(t: np.ndarray, pos_err: np.ndarray, outpath: Path) -> None:
    plt.figure()
    plt.plot(t, pos_err)
    plt.xlabel("t")
    plt.ylabel("||r_hat - r_true||")
    plt.title("EKF position error vs time (to tc)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _plot_nis(t: np.ndarray, nis: np.ndarray, outpath: Path) -> None:
    plt.figure()
    plt.plot(t, nis)
    plt.xlabel("t")
    plt.ylabel("NIS")
    plt.title("NIS vs time (to tc)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _plot_dv_bar(dv_perf: np.ndarray, dv_ekf: np.ndarray, outpath: Path) -> None:
    dv_perf_mag = _norm(dv_perf)
    dv_ekf_mag = _norm(dv_ekf)
    dv_delta = _norm(dv_ekf - dv_perf)

    plt.figure()
    plt.bar([0, 1, 2], [dv_perf_mag, dv_ekf_mag, dv_delta])
    plt.xticks([0, 1, 2], ["|dv| perfect", "|dv| ekf", "||Δdv||"])
    plt.ylabel("Δv magnitude")
    plt.title("Δv comparison")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _make_camera() -> Tuple[CameraIntrinsics, np.ndarray]:
    """
    Returns:
      intr: camera intrinsics
      R_cam_from_frame: camera rotation wrt 'frame' (global)
    """
    intr = CameraIntrinsics(
        fx=400.0,
        fy=400.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )
    R_cam_from_frame = np.eye(3)
    return intr, R_cam_from_frame


def run_case(
    mu: float,
    t0: float,
    tf: float,
    tc: float,
    dt_meas: float,
    sigma_px: float,
    dropout_prob: float,
    seed: int,
    dx0: np.ndarray,
    est_err: np.ndarray,
    *,
    fixed_camera_pointing: bool = False,
) -> Dict[str, Any]:
    """
    Single-run pipeline for 06A.

    Returns a dict with (at least):
      dv_perfect_mag, dv_ekf_mag, dv_delta_mag,
      miss_uncorrected, miss_perfect, miss_ekf,
      pos_err_tc, tracePpos_tc,
      valid_rate, nis_mean
    """
    rng = np.random.default_rng(int(seed))

    model = CR3BP(mu=mu)

    # nominal initial near L1 
    x0_nom = np.array([1.02, 0.0, 0.0, 0.0, -0.18, 0.0], dtype=float)

    # dense nominal propagation to tf for r_target
    t_dense = np.linspace(t0, tf, 2001)
    xs_nom = propagate(model, x0_nom, t_dense)
    r_target = xs_nom[-1, :3].copy()

    x0_true = (x0_nom + dx0).copy()
    t_meas = np.arange(t0, tf + 1e-12, dt_meas)
    xs_true = propagate(model, x0_true, t_meas)

    intr, R_cam_from_frame0 = _make_camera()

    u_meas_global: List[np.ndarray] = []
    R_cam_from_frame_hist: List[np.ndarray] = []
    valid_hist: List[bool] = []

    for k, tk in enumerate(t_meas):
        xk_true = xs_true[k]
        r_sc = xk_true[:3]

        # Choose observed body: Moon position in CR3BP is (1-mu,0,0)
        r_body = np.array([1.0 - mu, 0.0, 0.0], dtype=float)

        # "tracking attitude" vs "fixed pointing" for camera rotation history
        if fixed_camera_pointing:
            R_cam_from_frame = R_cam_from_frame0
        else:
            # simple "track" by keeping camera aligned with LOS direction:
            los = r_body - r_sc
            los = los / (np.linalg.norm(los) + 1e-12)
            # Build a crude camera frame with z-axis along LOS
            z = los
            # pick an arbitrary up that is not parallel
            up = np.array([0.0, 0.0, 1.0])
            if abs(float(np.dot(up, z))) > 0.95:
                up = np.array([0.0, 1.0, 0.0])
            x = np.cross(up, z)
            x = x / (np.linalg.norm(x) + 1e-12)
            y = np.cross(z, x)
            R_frame_from_cam = np.stack([x, y, z], axis=1)  # columns are cam axes in frame
            R_cam_from_frame = R_frame_from_cam.T

        R_cam_from_frame_hist.append(R_cam_from_frame)

        # dropout
        if rng.random() < float(dropout_prob):
            valid_hist.append(False)
            u_meas_global.append(np.full(3, np.nan))
            continue

        # simulate pixel measurement (u,v)
        u_px, v_px, _ = simulate_pixel_measurement(
            r_sc=r_sc,
            r_body=r_body,
            intr=intr,
            R_cam_from_frame=R_cam_from_frame,
            sigma_px=float(sigma_px),
            rng=rng,
        )

        # convert pixel->bearing in global frame
        # pixel_detection_to_bearing expects R_frame_from_cam
        R_frame_from_cam = R_cam_from_frame.T
        u_global, sigma_theta = pixel_detection_to_bearing(u_px, v_px, float(sigma_px), intr, R_frame_from_cam)

        valid_hist.append(True)
        u_meas_global.append(np.array(u_global, dtype=float))

    u_meas_global_arr = np.asarray(u_meas_global)
    valid_arr = np.asarray(valid_hist, dtype=bool)


    x_hat = (x0_nom + est_err).copy()  # init estimate about nominal state, not true state

    P = np.diag([1e-6] * 6).astype(float)
    q_acc = 0.0  

    x_hat_hist = []
    P_hist = []
    nis_hist = []
    pos_err_hist = []

    k_tc = _nearest_index(t_meas, tc)
    tc_eff = float(t_meas[k_tc])

    for k in range(1, k_tc + 1):
        t_prev = float(t_meas[k - 1])
        t_curr = float(t_meas[k])

        x_hat, P, _Phi = ekf_propagate_cr3bp_stm(mu, x_hat, P, t_prev, t_curr, q_acc)

        if valid_arr[k]:
            r_body = np.array([1.0 - mu, 0.0, 0.0], dtype=float)
            u_meas = u_meas_global_arr[k]
            sigma_theta = None  
            sigma_theta = 1e-3 + 0.0 * float(sigma_px)

            x_hat, P, _y, nis = bearing_update_tangent(x_hat, P, u_meas, r_body, sigma_theta)
            nis_hist.append(float(nis))
        else:
            nis_hist.append(float("nan"))

        x_hat_hist.append(x_hat.copy())
        P_hist.append(P.copy())

        x_true_k = xs_true[k]
        pos_err_hist.append(_norm(x_hat[:3] - x_true_k[:3]))

    x_hat_hist = np.asarray(x_hat_hist)
    P_hist = np.asarray(P_hist)
    nis_hist = np.asarray(nis_hist)
    pos_err_hist = np.asarray(pos_err_hist)

    x_true_tc = xs_true[k_tc]
    x_hat_tc = x_hat.copy()
    P_tc = P.copy()

    pos_err_tc = _norm(x_hat_tc[:3] - x_true_tc[:3])
    tracePpos_tc = float(np.trace(P_tc[:3, :3]))
    valid_rate = float(np.mean(valid_arr[: k_tc + 1]))
    nis_finite = nis_hist[np.isfinite(nis_hist)]
    nis_mean = float(np.mean(nis_finite)) if nis_finite.size else float("nan")

    dv_perf, info_perf = solve_single_impulse_position_target(
        mu=mu, x0=x_true_tc, t0=tc_eff, tc=tc_eff, tf=tf, r_target=r_target
    )
    dv_ekf, info_ekf = solve_single_impulse_position_target(
        mu=mu, x0=x_hat_tc, t0=tc_eff, tc=tc_eff, tf=tf, r_target=r_target
    )

    dv_perf = np.asarray(dv_perf, dtype=float)
    dv_ekf = np.asarray(dv_ekf, dtype=float)

    dv_perfect_mag = _norm(dv_perf)
    dv_ekf_mag = _norm(dv_ekf)
    dv_delta_mag = _norm(dv_ekf - dv_perf)

    # uncorrected truth
    t_dense2 = np.linspace(tc_eff, tf, 2001)
    xs_unc = propagate(model, x_true_tc, t_dense2)
    miss_unc = _norm(xs_unc[-1, :3] - r_target)

    # perfect burn truth
    x_perf0 = x_true_tc.copy()
    x_perf0[3:6] += dv_perf
    xs_perf = propagate(model, x_perf0, t_dense2)
    miss_perf = _norm(xs_perf[-1, :3] - r_target)

    # EKF burn truth
    x_ekf0 = x_true_tc.copy()
    x_ekf0[3:6] += dv_ekf
    xs_ekf = propagate(model, x_ekf0, t_dense2)
    miss_ekf = _norm(xs_ekf[-1, :3] - r_target)

    return {
        "tc": tc_eff,
        "sigma_px": float(sigma_px),
        "dropout_prob": float(dropout_prob),
        "fixed_camera_pointing": bool(fixed_camera_pointing),
        "dv_perfect_mag": dv_perfect_mag,
        "dv_ekf_mag": dv_ekf_mag,
        "dv_delta_mag": dv_delta_mag,
        "miss_uncorrected": miss_unc,
        "miss_perfect": miss_perf,
        "miss_ekf": miss_ekf,
        "pos_err_tc": pos_err_tc,
        "tracePpos_tc": tracePpos_tc,
        "valid_rate": valid_rate,
        "nis_mean": nis_mean,
        "debug": {
            "t_meas": t_meas,
            "k_tc": k_tc,
            "xs_nom": xs_nom,
            "xs_true": xs_true,
            "x_hat_hist": x_hat_hist,
            "pos_err_hist": pos_err_hist,
            "nis_hist": nis_hist,
            "xs_unc_tf": xs_unc,
            "xs_perf_tf": xs_perf,
            "xs_ekf_tf": xs_ekf,
            "dv_perf": dv_perf,
            "dv_ekf": dv_ekf,
        },
    }


def main() -> None:
    mu = 0.0121505856
    t0 = 0.0
    tf = 6.0
    tc = 2.0
    dt_meas = 0.02
    sigma_px = 1.5
    dropout_prob = 0.0
    seed = 7

    dx0 = np.array([1e-4, -1e-4, 0.0, 0.0, 0.0, 0.0], dtype=float)
    est_err = np.array([1e-4, 1e-4, 0.0, 0.0, 0.0, 0.0], dtype=float)

    out = run_case(mu, t0, tf, tc, dt_meas, sigma_px, dropout_prob, seed, dx0, est_err)

    dbg = out["debug"]
    plots_dir = Path("results/plots")
    _ensure_dir(plots_dir)

    # Trajectory plot
    _plot_traj(
        dbg["xs_nom"],
        dbg["xs_true"],
        dbg["xs_perf_tf"],
        dbg["xs_ekf_tf"],
        plots_dir / "06_midcourse_traj.png",
    )
    # Position error vs time (to tc)
    t_meas = dbg["t_meas"]
    k_tc = dbg["k_tc"]
    _plot_pos_err(t_meas[1 : k_tc + 1], dbg["pos_err_hist"], plots_dir / "06_ekf_pos_error.png")
    # NIS vs time (to tc)
    _plot_nis(t_meas[1 : k_tc + 1], dbg["nis_hist"], plots_dir / "06_ekf_nis.png")
    # dv bar chart
    _plot_dv_bar(dbg["dv_perf"], dbg["dv_ekf"], plots_dir / "06_dv_compare.png")

    print("06A complete.")
    print(f"tc used={out['tc']:.6f}")
    print(f"|dv| perfect = {out['dv_perfect_mag']:.6e}")
    print(f"|dv| ekf     = {out['dv_ekf_mag']:.6e}")
    print(f"||dv_ekf - dv_perfect|| = {out['dv_delta_mag']:.6e}")
    print("Wrote plots:")
    print(f"  {plots_dir / '06_midcourse_traj.png'}")
    print(f"  {plots_dir / '06_ekf_pos_error.png'}")
    print(f"  {plots_dir / '06_ekf_nis.png'}")
    print(f"  {plots_dir / '06_dv_compare.png'}")


if __name__ == "__main__":
    main()