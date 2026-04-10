from __future__ import annotations

from dataclasses import asdict
from typing import NamedTuple, Optional

import numpy as np
from diagnostics.config import CaseConfig
from diagnostics.health import check_matrix_health, regularize_spd
from diagnostics.types import (
    GateRecord,
    HealthRecord,
    RunResult,
    RunSummary,
    RunTrace,
    UpdateRecord,
)

from dynamics.cr3bp import CR3BP
from dynamics.integrators import propagate
from nav.ekf import ekf_propagate_cr3bp_stm
from nav.measurements.bearing import bearing_predict_measurement, bearing_update_tangent
from nav.measurements.pixel_bearing import pixel_detection_to_bearing

from cv.camera import Intrinsics, los_cam_to_pixel, rotate_vector
from cv.sim_measurements import PixelMeasurement, simulate_pixel_measurement
from cv.pointing import camera_dcm_from_boresight, desired_los_from_estimate


Array = np.ndarray




class MeasurementSnapshot(NamedTuple):
    measurement: PixelMeasurement
    R_cam_from_frame: Array





def _norm(x: Array) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=float)))


def _unit(x: Array, eps: float = 1e-12) -> Array:
    x = np.asarray(x, dtype=float).reshape(3)
    n = float(np.linalg.norm(x))
    if not np.isfinite(n) or n <= eps:
        return np.full(3, np.nan, dtype=float)
    return x / n


def _safe_nanmean(x: Array) -> float:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(x[finite]))


def _compute_nees(err: Array, P: Array) -> float:
    err = np.asarray(err, dtype=float).reshape(-1)
    P = np.asarray(P, dtype=float)

    if not np.all(np.isfinite(P)):
        return float("nan")

    try:
        P_reg = regularize_spd(P)
        val = float(err @ np.linalg.solve(P_reg, err))
    except np.linalg.LinAlgError:
        return float("nan")

    return max(val, 0.0)


def _los_angle(u_true: Array, u_est: Array) -> float:
    u_true = _unit(u_true)
    u_est = _unit(u_est)
    if not (np.all(np.isfinite(u_true)) and np.all(np.isfinite(u_est))):
        return float("nan")
    c = float(np.clip(np.dot(u_true, u_est), -1.0, 1.0))
    return float(np.arccos(c))


def _to_health_record(h) -> HealthRecord:
    return HealthRecord(
        name=h.name,
        shape=h.shape,
        is_finite=h.is_finite,
        symmetry_error_fro=h.symmetry_error_fro,
        min_eig=h.min_eig,
        max_eig=h.max_eig,
        cond=h.cond,
        is_spd=h.is_spd,
        chol_ok=h.chol_ok,
    )


def _to_gate_record(g) -> GateRecord:
    return GateRecord(
        enabled=g.enabled,
        accepted=g.accepted,
        threshold=g.threshold,
        statistic=g.statistic,
        probability=g.probability,
        dof=g.dof,
        reason=g.reason,
    )





def make_default_camera() -> Intrinsics:
    return Intrinsics(
        fx=400.0,
        fy=400.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )


def moon_position(mu: float) -> Array:
    return np.array([1.0 - mu, 0.0, 0.0], dtype=float)


def predict_pixel_from_state(
    x: Array,
    r_body: Array,
    intr: Intrinsics,
    R_cam_from_frame: Array,
) -> tuple[float, float, Array, Array]:
    u_pred_frame = bearing_predict_measurement(x, r_body)
    u_pred_cam = rotate_vector(np.asarray(R_cam_from_frame, dtype=float), u_pred_frame)
    u_px_pred, v_px_pred = los_cam_to_pixel(u_pred_cam, intr, behind="nan")
    return (
        float(np.asarray(u_px_pred)),
        float(np.asarray(v_px_pred)),
        np.asarray(u_pred_frame, dtype=float),
        np.asarray(u_pred_cam, dtype=float),
    )


def select_camera_rotation(
    *,
    camera_mode: str,
    r_sc_true: Array,
    x_hat_for_pointing: Array,
    r_body: Array,
    R_fixed: Array,
    up_hint: Optional[Array] = None,
) -> Array:
    if camera_mode == "fixed":
        return np.asarray(R_fixed, dtype=float)

    if camera_mode == "truth_tracking":
        boresight = _unit(np.asarray(r_body, dtype=float) - np.asarray(r_sc_true, dtype=float))
        return camera_dcm_from_boresight(boresight, up_hint_I=up_hint, camera_forward_axis="+z")

    if camera_mode == "estimate_tracking":
        boresight = desired_los_from_estimate(
            xhat_sc=np.asarray(x_hat_for_pointing, dtype=float),
            target_pos_I=np.asarray(r_body, dtype=float),
        )
        return camera_dcm_from_boresight(boresight, up_hint_I=up_hint, camera_forward_axis="+z")

    raise ValueError(f"Unknown camera_mode: {camera_mode!r}")


def _maybe_delay_measurement(
    history: list[MeasurementSnapshot],
    delay_steps: int,
) -> MeasurementSnapshot | None:
    if delay_steps < 0:
        raise ValueError("delay_steps must be >= 0")
    if len(history) <= delay_steps:
        return None
    return history[-1 - delay_steps]





def run_case(cfg: CaseConfig) -> RunResult:
    rng = np.random.default_rng(int(cfg.seed))
    model = CR3BP(mu=float(cfg.mu))

    x0_nom  = np.asarray(cfg.x0_nom, dtype=float).reshape(6)
    x0_true = x0_nom + np.asarray(cfg.dx0, dtype=float).reshape(6)
    xhat0   = x0_nom + np.asarray(cfg.est_err, dtype=float).reshape(6)
    P0      = np.diag(np.asarray(cfg.noise.p0_diag, dtype=float))

    if np.any(np.diag(P0) <= 0.0):
        raise ValueError("p0_diag entries must all be > 0 to form a valid initial covariance.")

    t_meas = np.arange(float(cfg.t0), float(cfg.tf) + 1e-12, float(cfg.dt_meas), dtype=float)

    truth_res = propagate(
        model.eom,
        (float(cfg.t0), float(cfg.tf)),
        x0_true,
        t_eval=t_meas,
    )
    if not truth_res.success:
        raise RuntimeError(f"Truth propagation failed: {truth_res.message}")

    xs_true = truth_res.x
    intr    = make_default_camera()
    r_body  = moon_position(cfg.mu)

    r_body_dir = moon_position(cfg.mu) - np.asarray(cfg.x0_nom, dtype=float)[:3]
    R_fixed = camera_dcm_from_boresight(r_body_dir, camera_forward_axis="+z")

    n = t_meas.size

    x_true_hist      = np.zeros((n, 6),    dtype=float)
    xhat_minus_hist  = np.full((n, 6),     np.nan, dtype=float)
    xhat_plus_hist   = np.full((n, 6),     np.nan, dtype=float)

    P_minus_hist = np.full((n, 6, 6), np.nan, dtype=float)
    P_plus_hist  = np.full((n, 6, 6), np.nan, dtype=float)
    Phi_hist     = np.full((n, 6, 6), np.nan, dtype=float)

    err_minus_hist = np.full((n, 6), np.nan, dtype=float)
    err_plus_hist  = np.full((n, 6), np.nan, dtype=float)

    nees_minus_hist = np.full(n, np.nan, dtype=float)
    nees_plus_hist  = np.full(n, np.nan, dtype=float)

    los_true_hist  = np.full((n, 3), np.nan, dtype=float)
    los_est_hist   = np.full((n, 3), np.nan, dtype=float)
    los_angle_hist = np.full(n,      np.nan, dtype=float)
    camera_R_hist  = np.full((n, 3, 3), np.nan, dtype=float)

    updates: list[UpdateRecord] = [
        UpdateRecord(t=float(t_meas[k]), valid_measurement=False, update_used=False)
        for k in range(n)
    ]
    P_minus_health: list[Optional[HealthRecord]] = [None] * n
    P_plus_health:  list[Optional[HealthRecord]] = [None] * n

    measurement_history: list[MeasurementSnapshot] = []

    x_true_hist[0]    = xs_true[0]
    xhat_plus_hist[0] = xhat0
    P_plus_hist[0]    = P0
    err_plus_hist[0]  = xhat0 - xs_true[0]
    nees_plus_hist[0] = _compute_nees(err_plus_hist[0], P0)
    P_plus_health[0]  = _to_health_record(check_matrix_health("P_plus_0", P0))

    los_true_hist[0]  = _unit(r_body - xs_true[0, :3])
    los_est_hist[0]   = _unit(r_body - xhat0[:3])
    los_angle_hist[0] = _los_angle(los_true_hist[0], los_est_hist[0])

    xhat_curr = xhat0.copy()
    P_curr    = P0.copy()

    for k in range(1, n):
        t_prev   = float(t_meas[k - 1])
        t_curr   = float(t_meas[k])
        x_true_k = xs_true[k].copy()
        x_true_hist[k] = x_true_k

        xhat_minus, P_minus, Phi = ekf_propagate_cr3bp_stm(
            mu=float(cfg.mu),
            x=xhat_curr,
            P=P_curr,
            t0=t_prev,
            t1=t_curr,
            q_acc=float(cfg.noise.q_acc),
        )

        xhat_minus_hist[k] = xhat_minus
        P_minus_hist[k]    = P_minus
        Phi_hist[k]        = Phi
        err_minus_hist[k]  = xhat_minus - x_true_k
        nees_minus_hist[k] = _compute_nees(err_minus_hist[k], P_minus)
        P_minus_health[k]  = _to_health_record(check_matrix_health(f"P_minus_{k}", P_minus))

        R_cam_from_frame = select_camera_rotation(
            camera_mode=cfg.camera_mode,
            r_sc_true=x_true_k[:3],
            x_hat_for_pointing=xhat_minus,
            r_body=r_body,
            R_fixed=R_fixed,
        )
        camera_R_hist[k] = R_cam_from_frame

        sigma_px = float(cfg.noise.sigma_px)
        if cfg.faults.outlier_prob > 0.0 and rng.random() < float(cfg.faults.outlier_prob):
            sigma_px *= float(cfg.faults.outlier_sigma_scale)

        pm_now = simulate_pixel_measurement(
            r_sc=x_true_k[:3],
            r_body=r_body,
            intrinsics=intr,
            R_cam_from_frame=R_cam_from_frame,
            sigma_px=sigma_px,
            rng=rng,
            t=t_curr,
            dropout_p=float(cfg.faults.dropout_prob),
        )
        measurement_history.append(MeasurementSnapshot(
            measurement=pm_now,
            R_cam_from_frame=np.asarray(R_cam_from_frame, dtype=float).copy(),
        ))

        snapshot = _maybe_delay_measurement(
            measurement_history,
            int(cfg.faults.measurement_delay_steps),
        )

        pm_used: PixelMeasurement | None = None
        R_cam_used: Array | None = None
        if snapshot is not None:
            pm_used    = snapshot.measurement
            R_cam_used = snapshot.R_cam_from_frame

        xhat_plus = xhat_minus.copy()
        P_plus    = P_minus.copy()

        valid_meas = (
            pm_used is not None
            and R_cam_used is not None
            and bool(pm_used.valid)
            and np.isfinite(pm_used.u_px)
            and np.isfinite(pm_used.v_px)
        )

        upd_rec = UpdateRecord(
            t=t_curr,
            valid_measurement=bool(valid_meas),
            update_used=False,
            measurement_meta=None if pm_used is None else pm_used.meta,
        )

        if valid_meas:
            upd_rec.pixel_uv = np.array(
                [float(pm_used.u_px), float(pm_used.v_px)], dtype=float
            )

            R_cam_used        = np.asarray(R_cam_used, dtype=float)
            R_frame_from_cam  = R_cam_used.T

            u_meas, sigma_theta = pixel_detection_to_bearing(
                pm_used.u_px,
                pm_used.v_px,
                float(pm_used.sigma_px),
                intr,
                R_frame_from_cam,
            )

            u_px_pred, v_px_pred, u_pred_frame, u_pred_cam = predict_pixel_from_state(
                xhat_minus, r_body, intr, R_cam_used,
            )

            upd_rec.pixel_uv_pred = np.array([u_px_pred, v_px_pred], dtype=float)
            upd_rec.u_pred        = np.asarray(u_pred_frame, dtype=float)
            upd_rec.u_pred_cam    = np.asarray(u_pred_cam, dtype=float)


            if np.all(np.isfinite(u_meas)) and np.isfinite(sigma_theta):
                upd_rec.sigma_theta = float(sigma_theta)
                upd_rec.u_meas      = np.asarray(u_meas, dtype=float)
                upd_rec.u_meas_cam  = np.asarray(
                    rotate_vector(R_cam_used, np.asarray(u_meas, dtype=float)), dtype=float
                )

                upd = bearing_update_tangent(
                    xhat_minus,
                    P_minus,
                    u_meas,
                    r_body,
                    float(sigma_theta),
                    gating_enabled=bool(cfg.gating.enabled),
                    gate_probability=float(cfg.gating.probability),
                    gate_dof=int(cfg.gating.measurement_dim),
                )

                upd_rec.innovation  = upd.innovation
                upd_rec.nis         = float(upd.nis)
                upd_rec.H           = upd.H
                upd_rec.R           = upd.R
                upd_rec.S           = upd.S
                upd_rec.K           = upd.K
                upd_rec.gate        = _to_gate_record(upd.gate)
                upd_rec.S_health    = _to_health_record(check_matrix_health(f"S_{k}", upd.S))
                upd_rec.update_used = bool(upd.accepted)

                if upd.accepted:
                    xhat_plus = upd.x_upd
                    P_plus    = upd.P_upd

        updates[k] = upd_rec

        xhat_plus_hist[k] = xhat_plus
        P_plus_hist[k]    = P_plus
        err_plus_hist[k]  = xhat_plus - x_true_k
        nees_plus_hist[k] = _compute_nees(err_plus_hist[k], P_plus)
        P_plus_health[k]  = _to_health_record(check_matrix_health(f"P_plus_{k}", P_plus))

        los_true_hist[k]  = _unit(r_body - x_true_k[:3])
        los_est_hist[k]   = _unit(r_body - xhat_plus[:3])
        los_angle_hist[k] = _los_angle(los_true_hist[k], los_est_hist[k])

        xhat_curr = xhat_plus
        P_curr    = P_plus

    valid_flags      = np.array([u.valid_measurement for u in updates], dtype=bool)
    used_flags       = np.array([u.update_used       for u in updates], dtype=bool)
    gate_accept_flags = np.array(
        [False if u.gate is None else bool(u.gate.accepted) for u in updates],
        dtype=bool,
    )
    nis_hist = np.array([u.nis for u in updates], dtype=float)



    def _rate(flags: Array) -> float:
        return float(np.mean(flags[1:])) if n > 1 else float(flags[0])

    summary = RunSummary(
        camera_mode=cfg.camera_mode,
        num_steps=int(n),
        valid_rate=_rate(valid_flags),
        update_rate=_rate(used_flags),
        gate_accept_rate=_rate(gate_accept_flags),
        nis_mean=_safe_nanmean(nis_hist[1:]),
        nees_minus_mean=_safe_nanmean(nees_minus_hist[1:]),
        nees_plus_mean=_safe_nanmean(nees_plus_hist[1:]),
        final_pos_err=float(_norm(err_plus_hist[-1, :3])),
        final_vel_err=float(_norm(err_plus_hist[-1, 3:6])),
        final_los_angle=float(los_angle_hist[-1]),
    )

    trace = RunTrace(
        t_meas=t_meas,
        x_true_hist=x_true_hist,
        xhat_minus_hist=xhat_minus_hist,
        xhat_plus_hist=xhat_plus_hist,
        P_minus_hist=P_minus_hist,
        P_plus_hist=P_plus_hist,
        Phi_hist=Phi_hist,
        err_minus_hist=err_minus_hist,
        err_plus_hist=err_plus_hist,
        nees_minus_hist=nees_minus_hist,
        nees_plus_hist=nees_plus_hist,
        los_true_hist=los_true_hist,
        los_est_hist=los_est_hist,
        los_angle_hist=los_angle_hist,
        camera_R_hist=camera_R_hist,
        updates=updates,
        P_minus_health=P_minus_health,
        P_plus_health=P_plus_health,
    )

    config_dict = {
        "mu":          float(cfg.mu),
        "t0":          float(cfg.t0),
        "tf":          float(cfg.tf),
        "dt_meas":     float(cfg.dt_meas),
        "seed":        int(cfg.seed),
        "x0_nom":      np.asarray(cfg.x0_nom, dtype=float).tolist(),
        "dx0":         np.asarray(cfg.dx0, dtype=float).tolist(),
        "est_err":     np.asarray(cfg.est_err, dtype=float).tolist(),
        "camera_mode": cfg.camera_mode,
        "noise":       asdict(cfg.noise),
        "gating":      asdict(cfg.gating),
        "faults":      asdict(cfg.faults),
    }

    return RunResult(config=config_dict, summary=summary, trace=trace)
