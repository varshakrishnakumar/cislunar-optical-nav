
"""
scripts/06_midcourse_ekf_correction.py

06A — EKF-driven midcourse correction (pixel->bearing, Option B)

Creates:
  results/plots/06_midcourse_traj.png
  results/plots/06_ekf_pos_error.png
  results/plots/06_ekf_nis.png
  results/plots/06_dv_compare.png

Run (recommended wrapper):
  ./run.sh scripts/06_midcourse_ekf_correction.py

Or:
  python scripts/06_midcourse_ekf_correction.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from dynamics.integrators import propagate
from dynamics.variational import cr3bp_eom_with_stm
from dynamics.cr3bp import CR3BP

from nav.ekf import ekf_propagate_cr3bp_stm
from nav.measurements.bearing import bearing_update_tangent
from nav.measurements.pixel_bearing import pixel_detection_to_bearing

from cv.camera import Intrinsics
from cv.sim_measurements import simulate_pixel_measurement


# ---------------------------
# Targeting (reuse 02)
# ---------------------------

def pack_state_and_stm(x: np.ndarray, phi: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(6,)
    if phi is None:
        phi = np.eye(6, dtype=float)
    phi = np.asarray(phi, dtype=float).reshape(6, 6)
    return np.concatenate([x, phi.reshape(-1, order="F")])


def unpack_state_and_stm(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=float).reshape(-1,)
    x = z[:6].copy()
    phi = z[6:].reshape(6, 6, order="F").copy()
    return x, phi


def propagate_cr3bp_with_stm(mu: float, t0: float, tf: float, z0: np.ndarray, *, dense: bool = False):
    return propagate(
        lambda t, z: cr3bp_eom_with_stm(t, z, mu),
        (t0, tf),
        z0,
        dense_output=dense,
        rtol=1e-11,
        atol=1e-13,
        method="DOP853",
    )


def solve_single_impulse_position_target(
    *,
    mu: float,
    x0: np.ndarray,
    t0: float,
    tc: float,
    tf: float,
    r_target: np.ndarray,
    dv0: np.ndarray | None = None,
    max_iter: int = 10,
    tol: float = 1e-10,
) -> tuple[np.ndarray, dict]:
    """
    Solve for dv at tc so that r(tf) matches r_target (3 constraints, 3 unknowns).
    Uses STM: dr(tf) ≈ Phi_rv * dv.
    """
    x0 = np.asarray(x0, dtype=float).reshape(6,)
    r_target = np.asarray(r_target, dtype=float).reshape(3,)

    # Propagate to tc (STM form for convenience)
    z0 = pack_state_and_stm(x0, np.eye(6))
    res_tc = propagate_cr3bp_with_stm(mu, t0, tc, z0, dense=False)
    if not res_tc.success:
        raise RuntimeError(f"Prop to tc failed: {res_tc.message}")
    x_tc, _ = unpack_state_and_stm(res_tc.x[-1])

    return solve_single_impulse_position_target_from_tc(
        mu=mu,
        x_tc=x_tc,
        tc=tc,
        tf=tf,
        r_target=r_target,
        dv0=dv0,
        max_iter=max_iter,
        tol=tol,
    )


def solve_single_impulse_position_target_from_tc(
    *,
    mu: float,
    x_tc: np.ndarray,
    tc: float,
    tf: float,
    r_target: np.ndarray,
    dv0: np.ndarray | None = None,
    max_iter: int = 10,
    tol: float = 1e-10,
) -> tuple[np.ndarray, dict]:
    """
    Same as solve_single_impulse_position_target, but accepts the state at tc directly.

    """
    x_tc = np.asarray(x_tc, dtype=float).reshape(6,)
    r_target = np.asarray(r_target, dtype=float).reshape(3,)

    dv = np.zeros(3) if dv0 is None else np.asarray(dv0, dtype=float).reshape(3,)
    hist = []

    for k in range(1, max_iter + 1):
        x_burn = x_tc.copy()
        x_burn[3:6] += dv

        # Propagate tc->tf with STM initialized to identity
        z_tc = pack_state_and_stm(x_burn, np.eye(6))
        res_tf = propagate_cr3bp_with_stm(mu, tc, tf, z_tc, dense=False)
        if not res_tf.success:
            raise RuntimeError(f"Prop to tf failed: {res_tf.message}")

        x_tf, phi = unpack_state_and_stm(res_tf.x[-1])
        err = x_tf[:3] - r_target
        err_norm = float(np.linalg.norm(err))
        hist.append({"iter": k, "dv": dv.copy(), "err": err.copy(), "err_norm": err_norm})

        if err_norm < tol:
            return dv, {"converged": True, "iterations": k, "history": hist, "final_err": err}

        Phi_rv = phi[0:3, 3:6]
        # Newton step: dv <- dv - Phi_rv^{-1} err
        try:
            delta = np.linalg.solve(Phi_rv, err)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(Phi_rv, err, rcond=None)
        dv = dv - delta

    return dv, {"converged": False, "iterations": max_iter, "history": hist, "final_err": hist[-1]["err"]}


# ---------------------------
# Measurement simulation (Option B)
# ---------------------------

def make_tracking_camera_attitude(u_global: np.ndarray) -> np.ndarray:
    """
    Build a camera attitude such that camera +Z points at u_global (in global frame).
    Returns R_cam_from_frame (maps global vectors into camera frame).
    """
    z_cam = np.asarray(u_global, dtype=float).reshape(3,)
    z_cam = z_cam / (np.linalg.norm(z_cam) + 1e-12)

    x_tmp = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(z_cam, x_tmp))) > 0.9:
        x_tmp = np.array([0.0, 1.0, 0.0])

    x_cam = np.cross(x_tmp, z_cam)
    x_cam = x_cam / (np.linalg.norm(x_cam) + 1e-12)
    y_cam = np.cross(z_cam, x_cam)
    y_cam = y_cam / (np.linalg.norm(y_cam) + 1e-12)

    return np.vstack([x_cam, y_cam, z_cam])


# ---------------------------
# Main
# ---------------------------


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
) -> dict:
    """
    Core 06A pipeline, factored for 06B sensitivity sweeps.

    Returns a dict with:
      dv_perfect_mag, dv_ekf_mag, dv_delta_mag,
      dv_inflation, dv_inflation_pct,
      miss_uncorrected, miss_perfect, miss_ekf,
      pos_err_tc, tracePpos_tc,
      valid_rate, nis_mean,
      debug (extra arrays for plotting in 06A).
    """
    plots_debug = {}

    model = CR3BP(mu=mu)

    # --- Measurement grid
    t_meas = np.arange(t0, tf + 1e-12, dt_meas)
    N = len(t_meas)
    k_tc = int(np.argmin(np.abs(t_meas - tc)))
    tc_eff = float(t_meas[k_tc])

    # --- Step 1: define nominal target (exactly like 02)
    L = model.lagrange_points()
    L1 = L["L1"]
    x0_nom = np.array([L1[0] - 1e-3, 0.0, 0.0, 0.0, 0.05, 0.0], dtype=float)

    res_nom = propagate(model.eom, (t0, tf), x0_nom, dense_output=True, rtol=1e-11, atol=1e-13, method="DOP853")
    if not res_nom.success or res_nom.sol is None:
        raise RuntimeError(f"Nominal propagation failed: {res_nom.message}")
    x_tf_nom = res_nom.sol(tf).reshape(6,)
    r_target = x_tf_nom[:3].copy()

    # --- Step 2: create truth with injection error
    dx0 = np.asarray(dx0, dtype=float).reshape(6,)
    x0_true = x0_nom + dx0

    res_truth = propagate(model.eom, (t0, tf), x0_true, t_eval=t_meas, rtol=1e-11, atol=1e-13, method="DOP853")
    if not res_truth.success:
        raise RuntimeError(f"Truth propagation failed: {res_truth.message}")
    X_true = res_truth.x
    r_true = X_true[:, :3]

    # --- Step 3: pixel->bearing measurements (Option B)
    r_body = np.asarray(model.primary2, dtype=float).reshape(3,)  # observe the Moon

    width, height = 1280, 720
    intr = Intrinsics(fx=800.0, fy=800.0, cx=width / 2, cy=height / 2, width=width, height=height)

    rng = np.random.default_rng(int(seed))

    u_px = np.full(N, np.nan)
    v_px = np.full(N, np.nan)
    valid = np.zeros(N, dtype=bool)
    R_cam_from_frame_hist = np.zeros((N, 3, 3), dtype=float)

    # Bonus mode: fixed camera pointing (constant R_cam_from_frame)
    R_fixed = None
    if fixed_camera_pointing:
        los0 = r_body - r_true[0]
        los0 /= np.linalg.norm(los0) + 1e-12
        R_fixed = make_tracking_camera_attitude(los0)

    for k in range(N):
        los = r_body - r_true[k]
        los /= np.linalg.norm(los) + 1e-12

        if fixed_camera_pointing and R_fixed is not None:
            R_cam_from_frame = R_fixed
        else:
            R_cam_from_frame = make_tracking_camera_attitude(los)

        R_cam_from_frame_hist[k] = R_cam_from_frame

        meas = simulate_pixel_measurement(
            r_sc=r_true[k],
            r_body=r_body,
            intrinsics=intr,
            R_cam_from_frame=R_cam_from_frame,
            sigma_px=float(sigma_px),
            rng=rng,
            t=float(t_meas[k]),
            dropout_p=float(dropout_prob),
            out_of_frame="drop",
            behind="drop",
        )
        if meas.valid:
            u_px[k] = meas.u_px
            v_px[k] = meas.v_px
            valid[k] = True

    # --- Step 4: run EKF from t0 to tc (stop at tc)
    est_err = np.asarray(est_err, dtype=float).reshape(6,)
    x_hat = (x0_true + est_err).copy()  # per instructions (or x0_nom if you want)
    P = np.diag([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4]).astype(float)

    # process noise (accel random walk proxy)
    q_acc = 1e-10

    x_hat_hist = np.zeros((N, 6), dtype=float)
    P_hist = np.zeros((N, 6, 6), dtype=float)
    nis_hist = np.full(N, np.nan, dtype=float)

    # truth/est at tc
    x_true_tc = X_true[k_tc].copy()

    # EKF loop
    for k in range(N):
        if k > k_tc:
            break

        if k == 0:
            x_hat_hist[k] = x_hat
            P_hist[k] = P
        else:
            # propagate to next measurement time
            dt = float(t_meas[k] - t_meas[k - 1])
            x_hat, P, _Phi = ekf_propagate_cr3bp_stm(mu=mu, x=x_hat, P=P, t0=float(t_meas[k-1]), t1=float(t_meas[k]), q_acc=q_acc)
            x_hat_hist[k] = x_hat
            P_hist[k] = P

        if valid[k]:
            # pixel -> bearing in global frame (needs camera attitude at this time)
            u = float(u_px[k])
            v = float(v_px[k])
            R_cam_from_frame = R_cam_from_frame_hist[k]

            R_frame_from_cam = R_cam_from_frame.T
            u_meas_global, sigma_theta_k = pixel_detection_to_bearing(
                u, v, float(sigma_px), intr, R_frame_from_cam
            )

            # Update EKF with bearing tangent formulation (returns NIS)
            x_hat, P, _y, nis = bearing_update_tangent(
                x_hat,
                P,
                u_meas_global,
                r_body,
                float(sigma_theta_k),
            )
            nis_hist[k] = float(nis)

        x_hat_hist[k] = x_hat
        P_hist[k] = P

    x_hat_tc = x_hat_hist[k_tc].copy()
    P_tc = P_hist[k_tc].copy()

    # --- Step 5: burns (perfect vs EKF-based)
    # Perfect-knowledge: use x0_true and let the solver propagate to tc internally
    dv_perfect, info_perfect = solve_single_impulse_position_target(
        mu=mu, x0=x0_true, t0=t0, tc=tc_eff, tf=tf, r_target=r_target
    )

    # EKF-based: use x_hat(tc) directly (cleanest)
    dv_ekf, info_ekf = solve_single_impulse_position_target_from_tc(
        mu=mu, x_tc=x_hat_tc, tc=tc_eff, tf=tf, r_target=r_target
    )

    # --- Step 6: apply burns and compute terminal miss distances
    # Uncorrected truth is simply truth at tf:
    r_unc_tf = r_true[-1]
    miss_unc = float(np.linalg.norm(r_unc_tf - r_target))

    # Corrected-perfect: propagate truth to tc, apply dv_perfect, propagate to tf
    res_truth_to_tc = propagate(model.eom, (t0, tc_eff), x0_true, dense_output=False, rtol=1e-11, atol=1e-13, method="DOP853")
    x_true_at_tc = res_truth_to_tc.x[-1].copy()
    x_true_at_tc[3:6] += dv_perfect

    res_perf = propagate(model.eom, (tc_eff, tf), x_true_at_tc, dense_output=False, rtol=1e-11, atol=1e-13, method="DOP853")
    x_perf_tf = res_perf.x[-1].copy()
    miss_perf = float(np.linalg.norm(x_perf_tf[:3] - r_target))

    # Corrected-ekf: propagate truth to tc, apply dv_ekf, propagate to tf
    # (rebuild from res_truth_to_tc to avoid any subtle aliasing)
    x_true_at_tc2 = res_truth_to_tc.x[-1].copy()
    x_true_at_tc2[3:6] += dv_ekf
    res_ekf = propagate(model.eom, (tc_eff, tf), x_true_at_tc2, dense_output=False, rtol=1e-11, atol=1e-13, method="DOP853")
    x_ekf_tf = res_ekf.x[-1].copy()
    miss_ekf = float(np.linalg.norm(x_ekf_tf[:3] - r_target))

    # --- EKF stats at tc
    pos_err_tc = float(np.linalg.norm(x_hat_tc[:3] - x_true_tc[:3]))
    tracePpos_tc = float(np.trace(P_tc[:3, :3]))

    nis_finite = nis_hist[: k_tc + 1]
    nis_finite = nis_finite[np.isfinite(nis_finite)]
    nis_mean = float(np.mean(nis_finite)) if nis_finite.size else float("nan")
    valid_rate = float(np.mean(valid[: k_tc + 1])) if (k_tc + 1) > 0 else 0.0

    # --- Burn metrics
    dv_perfect_mag = float(np.linalg.norm(dv_perfect))
    dv_ekf_mag = float(np.linalg.norm(dv_ekf))
    dv_delta_mag = float(np.linalg.norm(dv_ekf - dv_perfect))
    dv_inflation = float(dv_ekf_mag - dv_perfect_mag)
    dv_inflation_pct = float(dv_ekf_mag / dv_perfect_mag - 1.0) if dv_perfect_mag > 0 else float("nan")

    # Debug payload for plotting (06A)
    plots_debug.update(
        dict(
            model=model,
            t_meas=t_meas,
            tc_eff=tc_eff,
            k_tc=k_tc,
            x0_nom=x0_nom,
            x0_true=x0_true,
            r_target=r_target,
            X_true=X_true,
            x_hat_hist=x_hat_hist,
            nis_hist=nis_hist,
            valid=valid,
            dv_perfect=dv_perfect,
            dv_ekf=dv_ekf,
            x_perf_tf=x_perf_tf,
            x_ekf_tf=x_ekf_tf,
        )
    )

    return dict(
        mu=float(mu),
        t0=float(t0),
        tf=float(tf),
        tc=float(tc_eff),
        dt_meas=float(dt_meas),
        sigma_px=float(sigma_px),
        dropout_prob=float(dropout_prob),
        seed=int(seed),
        fixed_camera_pointing=bool(fixed_camera_pointing),
        dv_perfect_mag=dv_perfect_mag,
        dv_ekf_mag=dv_ekf_mag,
        dv_delta_mag=dv_delta_mag,
        dv_inflation=dv_inflation,
        dv_inflation_pct=dv_inflation_pct,
        miss_uncorrected=miss_unc,
        miss_perfect=miss_perf,
        miss_ekf=miss_ekf,
        pos_err_tc=pos_err_tc,
        tracePpos_tc=tracePpos_tc,
        valid_rate=valid_rate,
        nis_mean=nis_mean,
        debug=plots_debug,
        info_perfect=info_perfect,
        info_ekf=info_ekf,
    )


def main() -> None:
    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Defaults consistent with the lab handout
    mu = 0.0121505856
    t0 = 0.0
    tf = 6.0
    tc = 2.0
    dt_meas = 0.02
    sigma_px = 1.5
    dropout_prob = 0.0
    seed = 7
    dx0 = np.array([2e-4, -1e-4, 0.0, 0.0, 2e-3, 0.0], dtype=float)
    est_err = np.array([3e-4, 2e-4, 0.0, 0.0, -1.5e-3, 0.0], dtype=float)

    out = run_case(mu, t0, tf, tc, dt_meas, sigma_px, dropout_prob, seed, dx0, est_err)

    dbg = out["debug"]
    model = dbg["model"]
    t_meas = dbg["t_meas"]
    tc_eff = dbg["tc_eff"]
    k_tc = dbg["k_tc"]
    X_true = dbg["X_true"]
    x_hat_hist = dbg["x_hat_hist"]
    nis_hist = dbg["nis_hist"]
    r_target = dbg["r_target"]
    dv_perfect = dbg["dv_perfect"]
    dv_ekf = dbg["dv_ekf"]

    # Trajectory series
    r_nom = propagate(model.eom, (t0, tf), dbg["x0_nom"], t_eval=t_meas, rtol=1e-11, atol=1e-13, method="DOP853").x[:, :3]
    r_true = X_true[:, :3]

    # corrected truth trajectories (apply dv at tc and propagate)
    # perfect
    x_true_tc = propagate(model.eom, (t0, tc_eff), dbg["x0_true"], dense_output=False, rtol=1e-11, atol=1e-13, method="DOP853").x[-1].copy()
    x_true_tc_perf = x_true_tc.copy()
    x_true_tc_perf[3:6] += dv_perfect
    r_perf = propagate(model.eom, (tc_eff, tf), x_true_tc_perf, t_eval=t_meas[k_tc:], rtol=1e-11, atol=1e-13, method="DOP853").x[:, :3]

    # ekf
    x_true_tc_ekf = x_true_tc.copy()
    x_true_tc_ekf[3:6] += dv_ekf
    r_ekf = propagate(model.eom, (tc_eff, tf), x_true_tc_ekf, t_eval=t_meas[k_tc:], rtol=1e-11, atol=1e-13, method="DOP853").x[:, :3]

    # Position error up to tc
    pos_err = np.linalg.norm(x_hat_hist[:, :3] - r_true, axis=1)

    # 1) Trajectory plot
    fig, ax = plt.subplots(figsize=(9.0, 7.5))
    ax.plot(r_nom[:, 0], r_nom[:, 1], linewidth=2.0, label="nominal")
    ax.plot(r_true[:, 0], r_true[:, 1], linewidth=2.0, label="truth (uncorrected)")
    ax.plot(r_perf[:, 0], r_perf[:, 1], linewidth=2.0, label="corrected (perfect)")
    ax.plot(r_ekf[:, 0], r_ekf[:, 1], linewidth=2.0, label="corrected (EKF)")
    ax.scatter([r_target[0]], [r_target[1]], marker="x", s=80, label="r_target")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("06A — Midcourse correction (CR3BP, pixel->bearing EKF)")
    ax.grid(True, alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / "06_midcourse_traj.png", dpi=250)
    plt.close(fig)

    # 2) Position error vs time for EKF (up to tc)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(t_meas[: k_tc + 1], pos_err[: k_tc + 1], linewidth=2.0)
    ax.axvline(tc_eff, linestyle="--", linewidth=1.2)
    ax.set_xlabel("t")
    ax.set_ylabel("||r_hat - r_true||")
    ax.set_title("06A — EKF position error (up to tc)")
    ax.grid(True, alpha=0.5)
    fig.tight_layout()
    fig.savefig(plots_dir / "06_ekf_pos_error.png", dpi=250)
    plt.close(fig)

    # 3) NIS vs time (up to tc)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(t_meas[: k_tc + 1], nis_hist[: k_tc + 1], linewidth=2.0)
    ax.axvline(tc_eff, linestyle="--", linewidth=1.2)
    ax.set_xlabel("t")
    ax.set_ylabel("NIS")
    ax.set_title("06A — NIS (up to tc)")
    ax.grid(True, alpha=0.5)
    fig.tight_layout()
    fig.savefig(plots_dir / "06_ekf_nis.png", dpi=250)
    plt.close(fig)

    # 4) Bar chart: |dv| perfect, |dv| ekf, Δ|dv|
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    labels = ["|dv| perfect", "|dv| EKF", "||dv_EKF - dv_perfect||"]
    vals = [out["dv_perfect_mag"], out["dv_ekf_mag"], out["dv_delta_mag"]]
    ax.bar(labels, vals)
    ax.set_ylabel("Δv (ND units)")
    ax.set_title("06A — Burn comparison")
    ax.grid(True, axis="y", alpha=0.5)
    fig.tight_layout()
    fig.savefig(plots_dir / "06_dv_compare.png", dpi=250)
    plt.close(fig)

    print("06A complete.")
    print(f"tc requested={tc:.6f}, tc used (nearest in t_meas)={tc_eff:.6f}, index={k_tc}")
    print("\nPerfect-knowledge burn:")
    print(f"  converged={out['info_perfect']['converged']}, iters={out['info_perfect']['iterations']}")
    print(f"  dv={dv_perfect}")
    print(f"  |dv|={out['dv_perfect_mag']:.6e}")
    print("\nEKF-based burn (uses x_hat(tc)):")
    print(f"  converged={out['info_ekf']['converged']}, iters={out['info_ekf']['iterations']}")
    print(f"  dv={dv_ekf}")
    print(f"  |dv|={out['dv_ekf_mag']:.6e}")
    print(f"\nDelta burn magnitude: ||dv_EKF - dv_perfect|| = {out['dv_delta_mag']:.6e}")

    print("\nWrote plots:")
    print(" -", plots_dir / "06_midcourse_traj.png")
    print(" -", plots_dir / "06_ekf_pos_error.png")
    print(" -", plots_dir / "06_ekf_nis.png")
    print(" -", plots_dir / "06_dv_compare.png")


if __name__ == "__main__":
    main()
