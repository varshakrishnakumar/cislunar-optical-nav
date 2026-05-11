from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Literal

from _analysis_common import (
    AMBER as _AMBER,
    BG as _BG,
    BORDER as _BORDER,
    CYAN as _CYAN,
    EARTH as _EARTH_C,
    GREEN as _GREEN,
    MOON as _MOON_C,
    ORANGE as _ORANGE,
    PANEL as _PANEL,
    RED as _RED,
    TEXT as _TEXT,
    VIOLET as _VIOLET,
    apply_dark_theme as _apply_dark_theme,
)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import chi2

from dynamics.cr3bp import CR3BP
from dynamics.integrators import propagate
from nav.ekf import Qd_white_accel, ekf_propagate_cr3bp_stm, ekf_propagate_stm
from nav.measurements.bearing import (
    bearing_update_tangent, los_unit, tangent_basis,
)
from nav.measurements.pixel_bearing import pixel_detection_to_bearing
from cv.camera import Intrinsics
from cv.pointing import (
    camera_dcm_from_boresight,
    random_small_rotation_dcm,
    small_rotation_dcm,
)
from cv.sim_measurements import simulate_pixel_measurement
from guidance.targeting import solve_single_impulse_position_target


CameraMode = Literal["fixed", "truth_tracking", "estimate_tracking"]
_VALID_CAMERA_MODES = ("fixed", "truth_tracking", "estimate_tracking")

# ── SPICE / high-fidelity constants ─────────────────────────────────────────
_REPO_ROOT        = Path(__file__).resolve().parent.parent
_DEFAULT_KERNELS  = [
    _REPO_ROOT / "data" / "kernels" / "naif0012.tls",
    _REPO_ROOT / "data" / "kernels" / "de442s.bsp",
]
_EM_EPOCH         = "2026 APR 10 00:00:00 TDB"
_GM_EARTH_KM3_S2  = 398_600.435_436   # km³/s²
_GM_MOON_KM3_S2   =   4_902.800_066   # km³/s²


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def _nearest_index(t_grid: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(t_grid - t)))


def _make_camera() -> tuple[Intrinsics, np.ndarray]:
    intr    = Intrinsics(fx=400., fy=400., cx=320., cy=240., width=640, height=480)
    R_fixed = np.eye(3, dtype=float)
    return intr, R_fixed


def _resolve_camera_mode(camera_mode: Any) -> str:
    s = str(camera_mode).strip().lower()
    if s not in _VALID_CAMERA_MODES:
        raise ValueError(
            f"camera_mode must be one of {_VALID_CAMERA_MODES}, got {camera_mode!r}"
        )
    return s


# ── Filter-kind helpers (for the 06s estimator-ablation experiment) ─────────
# The default IEKF path is preserved exactly; "ekf" reuses the same code path
# with max_iterations=1; "ukf" uses the Wan/van der Merwe sigma-point UKF
# (alpha=1e-3, beta=2.0, kappa=0.0) lifted from src/cisopt/estimators/ukf.py
# so the two implementations stay numerically aligned. The truth model,
# camera, pointing law, measurement generation, gating policy
# (gating_enabled=False everywhere), process-noise tuning, and seeds are
# identical across all three filter kinds — only the predict and update
# steps differ. See `\subsection{Estimator Ablation}` in the report for the
# implementation-vs-theory caveats.
FilterKind = Literal["ekf", "iekf", "ukf"]
_VALID_FILTER_KINDS = ("ekf", "iekf", "ukf")
_UKF_ALPHA = 1.0e-3
_UKF_BETA  = 2.0
_UKF_KAPPA = 0.0


def _resolve_filter_kind(filter_kind: Any) -> str:
    s = str(filter_kind).strip().lower()
    if s not in _VALID_FILTER_KINDS:
        raise ValueError(
            f"filter_kind must be one of {_VALID_FILTER_KINDS}, got {filter_kind!r}"
        )
    return s


def _ukf_sigma_weights(n: int) -> tuple[float, np.ndarray, np.ndarray]:
    lam = _UKF_ALPHA * _UKF_ALPHA * (n + _UKF_KAPPA) - n
    c = n + lam
    Wm = np.full(2 * n + 1, 1.0 / (2.0 * c), dtype=float)
    Wc = np.full(2 * n + 1, 1.0 / (2.0 * c), dtype=float)
    Wm[0] = lam / c
    Wc[0] = lam / c + (1.0 - _UKF_ALPHA * _UKF_ALPHA + _UKF_BETA)
    return c, Wm, Wc


def _ukf_sigma_points(x: np.ndarray, P: np.ndarray, c: float) -> np.ndarray:
    n = x.size
    P_sym = 0.5 * (P + P.T)
    eigvals, eigvecs = np.linalg.eigh(P_sym)
    eigvals = np.maximum(eigvals, 0.0)
    sqrt_P = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
    L = np.linalg.cholesky(c * (sqrt_P @ sqrt_P) + 1e-18 * np.eye(n))
    sigmas = np.zeros((2 * n + 1, n), dtype=float)
    sigmas[0] = x
    for i in range(n):
        sigmas[1 + i] = x + L[:, i]
        sigmas[1 + n + i] = x - L[:, i]
    return sigmas


def _ukf_predict_cr3bp(
    *, mu: float, x: np.ndarray, P: np.ndarray,
    t0: float, t1: float, q_acc: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sigma-point CR3BP predict; returns (x_pred, P_pred). No STM."""
    n = int(x.size)
    c, Wm, Wc = _ukf_sigma_weights(n)
    sigmas = _ukf_sigma_points(x, P, c)
    model = CR3BP(mu=float(mu))
    propagated = np.zeros_like(sigmas)
    for i in range(sigmas.shape[0]):
        res = propagate(
            model.eom, (float(t0), float(t1)),
            sigmas[i], rtol=1e-11, atol=1e-13,
        )
        if not res.success:
            raise RuntimeError(f"UKF sigma propagation failed: {res.message}")
        propagated[i] = res.x[-1]
    x_pred = (Wm[:, None] * propagated).sum(axis=0)
    diff = propagated - x_pred
    P_pred = (Wc[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0)
    P_pred = 0.5 * (P_pred + P_pred.T)
    P_pred = P_pred + Qd_white_accel(float(t1) - float(t0), float(q_acc))
    return x_pred, P_pred


def _ukf_bearing_update(
    *, x: np.ndarray, P: np.ndarray,
    u_meas: np.ndarray, r_body: np.ndarray, sigma_theta: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sigma-point bearing update with tangent-plane unit-vector residual.

    Lifted verbatim from cisopt UKF for numerical alignment. Returns
    (x_upd, P_upd, nis). Always 'accepts' (no gating), matching the
    gating_enabled=False default used by bearing_update_tangent in this
    pipeline.
    """
    n = int(x.size)
    c, Wm, Wc = _ukf_sigma_weights(n)
    sigmas = _ukf_sigma_points(x, P, c)

    u_sigmas = np.zeros((sigmas.shape[0], 3), dtype=float)
    for i in range(sigmas.shape[0]):
        u_i, _ = los_unit(r_body, sigmas[i, :3])
        u_sigmas[i] = u_i

    u_pred_mean = (Wm[:, None] * u_sigmas).sum(axis=0)
    u_pred_mean = u_pred_mean / np.linalg.norm(u_pred_mean)

    e1, e2 = tangent_basis(u_pred_mean)
    E = np.vstack([e1, e2])

    z_sigmas = u_sigmas @ E.T
    z_pred = (Wm[:, None] * z_sigmas).sum(axis=0)
    z_meas = E @ np.asarray(u_meas, dtype=float).reshape(3)
    innov = z_meas - z_pred

    diff_z = z_sigmas - z_pred
    diff_x = sigmas - x
    R = (sigma_theta * sigma_theta) * np.eye(2, dtype=float)

    S = (Wc[:, None, None] * diff_z[:, :, None] * diff_z[:, None, :]).sum(axis=0) + R
    S = 0.5 * (S + S.T)
    Cxz = (Wc[:, None, None] * diff_x[:, :, None] * diff_z[:, None, :]).sum(axis=0)

    try:
        K = np.linalg.solve(S.T, Cxz.T).T
    except np.linalg.LinAlgError:
        K = Cxz @ np.linalg.pinv(S)

    x_upd = x + K @ innov
    P_upd = P - K @ S @ K.T
    P_upd = 0.5 * (P_upd + P_upd.T)

    try:
        nis = float(innov @ np.linalg.solve(S, innov))
    except np.linalg.LinAlgError:
        nis = float("nan")
    return x_upd, P_upd, nis



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
    camera_mode: CameraMode = "estimate_tracking",
    q_acc: float = 1e-14,
    sigma_att_rad: float = 0.0,
    bias_att_rad: Any = None,
    pointing_lag_steps: int = 0,
    meas_delay_steps: int = 0,
    P0_scale: float = 1.0,
    landmark_positions: Any = None,
    disable_moon_center: bool = False,
    return_debug: bool = True,
    accumulate_gramian: bool = True,
    P_cov_history: bool = False,
    filter_kind: FilterKind = "iekf",
    iekf_max_iter: int = 3,
) -> Dict[str, Any]:
    camera_mode = _resolve_camera_mode(camera_mode)
    filter_kind = _resolve_filter_kind(filter_kind)
    # filter_kind="ekf" is shorthand for "IEKF with max_iterations=1",
    # matching cisopt's EKFEstimator convention. Auto-override only when
    # the caller left iekf_max_iter at its default of 3, so an explicit
    # caller-provided value is still respected.
    if filter_kind == "ekf" and int(iekf_max_iter) == 3:
        iekf_max_iter = 1
    # UKF has no STM, so the observability Gramian cannot be accumulated
    # with the same machinery as EKF/IEKF. Silently disable rather than
    # raising — the 06s ablation driver passes accumulate_gramian=False
    # explicitly, but other callers may not.
    if filter_kind == "ukf" and accumulate_gramian:
        accumulate_gramian = False

    rng    = np.random.default_rng(int(seed))
    model  = CR3BP(mu=float(mu))

    L1x    = model.lagrange_points()["L1"][0]
    x0_nom = np.array([L1x - 1e-3, 0.0, 0.0, 0.0, 0.05, 0.0], dtype=float)
    x0_true = x0_nom + np.asarray(dx0, dtype=float).reshape(6)
    r_body  = np.array([1.0 - float(mu), 0.0, 0.0], dtype=float)

    bias_vec = (
        np.zeros(3, dtype=float)
        if bias_att_rad is None
        else np.asarray(bias_att_rad, dtype=float).reshape(3)
    )
    R_bias = small_rotation_dcm(bias_vec) if np.any(bias_vec != 0.0) else None

    if landmark_positions is None:
        lmk_arr = np.zeros((0, 3), dtype=float)
    else:
        lmk_arr = np.asarray(landmark_positions, dtype=float).reshape(-1, 3)

    pointing_lag_steps = max(0, int(pointing_lag_steps))
    meas_delay_steps   = max(0, int(meas_delay_steps))

    res_nom = propagate(model.eom, (float(t0), float(tf)), x0_nom,
                        t_eval=np.linspace(t0, tf, 2001), rtol=1e-11, atol=1e-13)
    if not res_nom.success:
        raise RuntimeError(f"Nominal propagation failed: {res_nom.message}")
    r_target = res_nom.x[-1, :3].copy()

    t_meas   = np.arange(float(t0), float(tf) + 1e-12, float(dt_meas))
    res_true = propagate(model.eom, (float(t0), float(tf)), x0_true,
                         t_eval=t_meas, rtol=1e-11, atol=1e-13)
    if not res_true.success:
        raise RuntimeError(f"Truth propagation failed: {res_true.message}")
    xs_true = res_true.x

    intr, R_fixed = _make_camera()

    k_tc   = _nearest_index(t_meas, float(tc))
    tc_eff = float(t_meas[k_tc])

    x_hat = x0_nom + np.asarray(est_err, dtype=float).reshape(6)
    # Position σ ~ 1e-3 DU, velocity σ ~ 1e-3.5 DU/TU (velocity uncertainty
    # is ~10× smaller than position in CR3BP dimensionless units).
    P     = np.diag([1e-6, 1e-6, 1e-6, 1e-7, 1e-7, 1e-7]).astype(float)
    P    *= float(P0_scale)
    q_acc = float(q_acc)

    x_hat_hist:   list[np.ndarray] = []
    # Moon-center NIS history (NaN when no Moon update at this epoch)
    nis_list:     list[float]      = []
    # Per-epoch list of all accepted-update NIS values (Moon + every landmark)
    nis_all_per_epoch: list[list[float]] = []
    nis_landmarks_per_epoch: list[list[float]] = []
    nees_list:    list[float]      = []
    innov_2d_list: list[np.ndarray] = []
    pos_err_list:  list[float]     = []
    P_diag_list:  list[np.ndarray] = []
    P_full_hist:  list[np.ndarray] = [] if P_cov_history else []
    los_inertial_hist: list[np.ndarray] = []
    range_truth_hist:  list[float]      = []
    range_estimate_hist: list[float]    = []
    x_hat_pre_buf: list[np.ndarray]     = []
    # Per-epoch accept counters (index 0 always 0; valid_rate excludes it)
    accepted_moon_arr      = np.zeros(k_tc + 1, dtype=bool)
    accepted_landmarks_arr = np.zeros(k_tc + 1, dtype=int)
    epoch_accepted_arr     = np.zeros(k_tc + 1, dtype=bool)
    # Per-source angular offset of the imaged point from the *commanded*
    # camera boresight (+z), computed from truth at k_meas. Distinguishes
    # "source stayed visible" (small offset, in FOV) from "source forced
    # near FOV edge" (large offset, near or past FOV limit). Lets the
    # 06r landmarks-under-pointing-degradation driver tell apart the
    # visibility-substitution and geometry-improvement mechanisms.
    moon_offset_rad_hist:      list[float]      = []
    landmark_offset_rad_hist:  list[np.ndarray] = []

    # Observability Gramian: W = Σ_k Φ(t_k,t_0)ᵀ Hₖᵀ Hₖ Φ(t_k,t_0)
    # Eigenvectors of W reveal which state-space directions are (poorly)
    # observable from bearing-only data. Tracked separately for moon vs
    # landmarks so the contribution of each information source can be
    # compared in the observability plot.
    Phi_cum            = np.eye(6, dtype=float) if accumulate_gramian else None
    W_obs              = np.zeros((6, 6), dtype=float) if accumulate_gramian else None
    W_obs_moon         = np.zeros((6, 6), dtype=float) if accumulate_gramian else None
    W_obs_landmarks    = np.zeros((6, 6), dtype=float) if accumulate_gramian else None
    gramian_eig_hist: list[np.ndarray] = []

    # Per-trial runtime accumulators for the 06s estimator-ablation experiment.
    # Times are wall-clock perf_counter() seconds, exclusive of measurement
    # generation, gating, observability accumulation, and bookkeeping.
    t_predict_total_s  = 0.0
    t_update_total_s   = 0.0
    n_predict_calls    = 0
    n_update_calls     = 0
    iters_used_total   = 0  # sum of iterations consumed across all updates
    t_trial_start_s    = time.perf_counter()

    for k in range(1, k_tc + 1):
        _t_pred = time.perf_counter()
        if filter_kind == "ukf":
            x_hat, P = _ukf_predict_cr3bp(
                mu=float(mu), x=x_hat, P=P,
                t0=float(t_meas[k - 1]), t1=float(t_meas[k]), q_acc=q_acc,
            )
            Phi_step = None
        else:
            x_hat, P, Phi_step = ekf_propagate_cr3bp_stm(
                mu=float(mu), x=x_hat, P=P,
                t0=float(t_meas[k - 1]), t1=float(t_meas[k]), q_acc=q_acc,
            )
        t_predict_total_s += time.perf_counter() - _t_pred
        n_predict_calls   += 1
        if accumulate_gramian and Phi_step is not None:
            Phi_cum = Phi_step @ Phi_cum

        # Optional measurement-delay simulation: pretend the truth state
        # used for measurement generation is from `meas_delay_steps` ago.
        # The filter still timestamps the update at `t_meas[k]`.
        k_meas = max(0, k - meas_delay_steps)
        r_sc_true = xs_true[k_meas, :3]

        # Pointing-lag: the camera tracks an estimate from `pointing_lag_steps`
        # steps in the past (k=current; lag=0 → use current x_hat).
        x_hat_pre_buf.append(x_hat.copy())
        if camera_mode == "estimate_tracking":
            if pointing_lag_steps > 0 and len(x_hat_pre_buf) > pointing_lag_steps:
                x_hat_for_pointing = x_hat_pre_buf[-1 - pointing_lag_steps]
            else:
                x_hat_for_pointing = x_hat
            R_cam_cmd = camera_dcm_from_boresight(
                r_body - x_hat_for_pointing[:3], camera_forward_axis="+z"
            )
        elif camera_mode == "truth_tracking":
            R_cam_cmd = camera_dcm_from_boresight(
                r_body - r_sc_true, camera_forward_axis="+z"
            )
        else:  # fixed
            R_cam_cmd = R_fixed

        # Apply systematic pointing bias (constant, deterministic) plus
        # random per-step attitude noise. Bias multiplies first, then
        # noise — so noise is a small perturbation around the biased pose.
        R_cam_actual = R_cam_cmd
        if R_bias is not None:
            R_cam_actual = R_bias @ R_cam_actual
        if sigma_att_rad > 0.0:
            R_perturb = random_small_rotation_dcm(rng, float(sigma_att_rad))
            R_cam_actual = R_perturb @ R_cam_actual

        # Angular offset of each imaged point from the commanded boresight
        # (+z in the camera frame). NaN if the source is behind the camera
        # (z<=0). This is computed from R_cam_cmd, not R_cam_actual, so
        # bias/noise show up as accept/reject churn rather than as offset.
        rho_moon_inertial = r_body - r_sc_true
        n_moon = float(np.linalg.norm(rho_moon_inertial))
        if n_moon > 1e-12:
            u_moon_cam = R_cam_cmd @ (rho_moon_inertial / n_moon)
            moon_offset_rad = (
                float(np.arccos(np.clip(u_moon_cam[2], -1.0, 1.0)))
                if u_moon_cam[2] > 0.0 else float("nan")
            )
        else:
            moon_offset_rad = float("nan")
        moon_offset_rad_hist.append(moon_offset_rad)
        if lmk_arr.shape[0] > 0:
            lmk_offsets = np.full(lmk_arr.shape[0], np.nan, dtype=float)
            for j in range(lmk_arr.shape[0]):
                rho_lmk = lmk_arr[j] - r_sc_true
                n_lmk = float(np.linalg.norm(rho_lmk))
                if n_lmk > 1e-12:
                    u_lmk_cam = R_cam_cmd @ (rho_lmk / n_lmk)
                    if u_lmk_cam[2] > 0.0:
                        lmk_offsets[j] = float(
                            np.arccos(np.clip(u_lmk_cam[2], -1.0, 1.0))
                        )
            landmark_offset_rad_hist.append(lmk_offsets)
        else:
            landmark_offset_rad_hist.append(np.zeros(0, dtype=float))

        nis_all_this_epoch:        list[float] = []
        nis_landmarks_this_epoch:  list[float] = []
        moon_accepted = False

        if not disable_moon_center:
            meas = simulate_pixel_measurement(
                r_sc=r_sc_true, r_body=r_body, intrinsics=intr,
                R_cam_from_frame=R_cam_actual, sigma_px=float(sigma_px),
                rng=rng, t=float(t_meas[k]),
                dropout_p=float(dropout_prob), out_of_frame="drop", behind="drop",
            )

            # The filter back-projects the pixel through the *commanded* DCM
            # so attitude error / bias / lag flow through as un-modeled
            # measurement error.
            if meas.valid and np.isfinite(meas.u_px):
                u_g, sig_k = pixel_detection_to_bearing(
                    meas.u_px, meas.v_px, float(sigma_px), intr, R_cam_cmd.T
                )
                if np.all(np.isfinite(u_g)):
                    _t_upd = time.perf_counter()
                    if filter_kind == "ukf":
                        x_hat, P, _nis_ukf = _ukf_bearing_update(
                            x=x_hat, P=P, u_meas=u_g,
                            r_body=r_body, sigma_theta=float(sig_k),
                        )
                        moon_accepted = True
                        nis_list.append(float(_nis_ukf))
                        nis_all_this_epoch.append(float(_nis_ukf))
                        innov_2d_list.append(np.full(2, np.nan))  # not tracked for UKF
                        iters_used_total += 1
                    else:
                        upd = bearing_update_tangent(
                            x_hat, P, u_g, r_body, float(sig_k),
                            max_iterations=int(iekf_max_iter),
                        )
                        if upd.accepted:
                            x_hat, P = upd.x_upd, upd.P_upd
                            moon_accepted = True
                            if accumulate_gramian:
                                HtH = upd.H.T @ upd.H
                                W_obs      += Phi_cum.T @ HtH @ Phi_cum
                                W_obs_moon += Phi_cum.T @ HtH @ Phi_cum
                        nis_list.append(float(upd.nis))
                        nis_all_this_epoch.append(float(upd.nis))
                        innov_2d_list.append(
                            upd.final_innovation.copy()
                            if upd.final_innovation is not None
                            else upd.innovation.copy()
                        )
                        iters_used_total += int(upd.iterations)
                    t_update_total_s += time.perf_counter() - _t_upd
                    n_update_calls   += 1
                else:
                    nis_list.append(float("nan"))
                    innov_2d_list.append(np.full(2, np.nan))
            else:
                nis_list.append(float("nan"))
                innov_2d_list.append(np.full(2, np.nan))
        else:
            nis_list.append(float("nan"))
            innov_2d_list.append(np.full(2, np.nan))

        # Optional landmark updates. Treat each landmark as an inertial
        # point near r_body; generate a separate bearing measurement and
        # update the filter scalarly. Same R_cam_actual is used so the
        # bearing inherits the same attitude error. Each accepted update
        # also contributes to the observability Gramian and is tracked
        # in the per-epoch accept counter.
        n_lmk_accepted = 0
        for lmk_pos in lmk_arr:
            lm_meas = simulate_pixel_measurement(
                r_sc=r_sc_true, r_body=lmk_pos, intrinsics=intr,
                R_cam_from_frame=R_cam_actual, sigma_px=float(sigma_px),
                rng=rng, t=float(t_meas[k]),
                dropout_p=float(dropout_prob), out_of_frame="drop", behind="drop",
            )
            if not (lm_meas.valid and np.isfinite(lm_meas.u_px)):
                continue
            u_l, sig_l = pixel_detection_to_bearing(
                lm_meas.u_px, lm_meas.v_px, float(sigma_px), intr, R_cam_cmd.T
            )
            if not np.all(np.isfinite(u_l)):
                continue
            _t_upd_l = time.perf_counter()
            if filter_kind == "ukf":
                x_hat, P, _nis_l = _ukf_bearing_update(
                    x=x_hat, P=P, u_meas=u_l,
                    r_body=lmk_pos, sigma_theta=float(sig_l),
                )
                n_lmk_accepted += 1
                nis_all_this_epoch.append(float(_nis_l))
                nis_landmarks_this_epoch.append(float(_nis_l))
                iters_used_total += 1
            else:
                upd_l = bearing_update_tangent(
                    x_hat, P, u_l, lmk_pos, float(sig_l),
                    max_iterations=int(iekf_max_iter),
                )
                if upd_l.accepted:
                    x_hat, P = upd_l.x_upd, upd_l.P_upd
                    n_lmk_accepted += 1
                    if accumulate_gramian:
                        HtH_l = upd_l.H.T @ upd_l.H
                        W_obs           += Phi_cum.T @ HtH_l @ Phi_cum
                        W_obs_landmarks += Phi_cum.T @ HtH_l @ Phi_cum
                nis_all_this_epoch.append(float(upd_l.nis))
                nis_landmarks_this_epoch.append(float(upd_l.nis))
                iters_used_total += int(upd_l.iterations)
            t_update_total_s += time.perf_counter() - _t_upd_l
            n_update_calls   += 1

        accepted_moon_arr[k]      = moon_accepted
        accepted_landmarks_arr[k] = n_lmk_accepted
        epoch_accepted_arr[k]     = bool(moon_accepted) or (n_lmk_accepted > 0)
        nis_all_per_epoch.append(nis_all_this_epoch)
        nis_landmarks_per_epoch.append(nis_landmarks_this_epoch)

        # NEES: (x̂ − x_true)ᵀ P⁻¹ (x̂ − x_true), chi²(6) distributed for a
        # consistent filter. Use truth at k (not k_meas) so the consistency
        # check is against the actual current state.
        err6 = x_hat - xs_true[k]
        try:
            nees_val = float(err6 @ np.linalg.solve(P, err6))
        except np.linalg.LinAlgError:
            nees_val = float("nan")
        nees_list.append(nees_val)

        x_hat_hist.append(x_hat.copy())
        pos_err_list.append(_norm(x_hat[:3] - xs_true[k, :3]))
        P_diag_list.append(np.diag(P).copy())
        if P_cov_history:
            P_full_hist.append(P.copy())
        # Parallax / range telemetry: LOS in inertial frame (toward Moon),
        # truth range, estimated range.
        rho_truth = r_body - xs_true[k, :3]
        rho_est   = r_body - x_hat[:3]
        n_truth = float(np.linalg.norm(rho_truth))
        n_est   = float(np.linalg.norm(rho_est))
        los_inertial_hist.append(
            (rho_truth / n_truth) if n_truth > 1e-12 else np.full(3, np.nan)
        )
        range_truth_hist.append(n_truth)
        range_estimate_hist.append(n_est)
        if accumulate_gramian:
            gramian_eig_hist.append(np.linalg.eigvalsh(W_obs).copy())

    nis_arr      = np.asarray(nis_list)
    nees_arr     = np.asarray(nees_list)

    x_true_tc    = xs_true[k_tc]
    x_hat_tc     = x_hat.copy()
    P_tc         = P.copy()
    pos_err_tc   = _norm(x_hat_tc[:3] - x_true_tc[:3])
    tracePpos_tc = float(np.trace(P_tc[:3, :3]))

    # valid_rate counts only the k=1..k_tc epochs (k=0 is the initial
    # state, never a measurement). Three flavors so reviewers can
    # distinguish moon-only vs landmark-only vs combined coverage:
    n_epochs       = k_tc  # we ran k=1..k_tc
    valid_rate     = (
        float(np.mean(epoch_accepted_arr[1:k_tc + 1]))
        if n_epochs > 0 else float("nan")
    )
    valid_rate_moon = (
        float(np.mean(accepted_moon_arr[1:k_tc + 1]))
        if n_epochs > 0 else float("nan")
    )
    n_lmk = int(lmk_arr.shape[0])
    if n_lmk > 0 and n_epochs > 0:
        valid_rate_landmarks = float(
            np.mean(accepted_landmarks_arr[1:k_tc + 1] / float(n_lmk))
        )
    else:
        valid_rate_landmarks = float("nan")

    # NIS aggregations: legacy `nis_mean` is moon-only (preserves
    # downstream MC plots); `nis_mean_all` averages across moon +
    # landmark accepted updates.
    nis_finite   = nis_arr[np.isfinite(nis_arr)]
    nis_mean     = float(np.mean(nis_finite)) if nis_finite.size else float("nan")
    flat_all = np.array(
        [v for ep in nis_all_per_epoch for v in ep if np.isfinite(v)],
        dtype=float,
    )
    nis_mean_all = float(np.mean(flat_all)) if flat_all.size else float("nan")
    flat_lmk = np.array(
        [v for ep in nis_landmarks_per_epoch for v in ep if np.isfinite(v)],
        dtype=float,
    )
    nis_mean_landmarks = (
        float(np.mean(flat_lmk)) if flat_lmk.size else float("nan")
    )
    nees_finite  = nees_arr[np.isfinite(nees_arr)]
    nees_mean    = float(np.mean(nees_finite)) if nees_finite.size else float("nan")

    result_perf = solve_single_impulse_position_target(
        propagate=propagate, mu=float(mu), x0=x_true_tc,
        t0=tc_eff, tc=tc_eff, tf=float(tf), r_target=r_target,
    )
    result_ekf = solve_single_impulse_position_target(
        propagate=propagate, mu=float(mu), x0=x_hat_tc,
        t0=tc_eff, tc=tc_eff, tf=float(tf), r_target=r_target,
    )

    dv_perf = np.asarray(result_perf.dv, dtype=float)
    dv_ekf  = np.asarray(result_ekf.dv,  dtype=float)

    t_post   = np.linspace(tc_eff, float(tf), 2001)
    res_unc  = propagate(model.eom, (tc_eff, float(tf)), x_true_tc,
                         t_eval=t_post, rtol=1e-11, atol=1e-13)
    miss_unc = _norm(res_unc.x[-1, :3] - r_target)

    x_perf0 = x_true_tc.copy(); x_perf0[3:6] += dv_perf
    res_perf = propagate(model.eom, (tc_eff, float(tf)), x_perf0,
                         t_eval=t_post, rtol=1e-11, atol=1e-13)
    miss_perf = _norm(res_perf.x[-1, :3] - r_target)

    x_ekf0 = x_true_tc.copy(); x_ekf0[3:6] += dv_ekf
    res_ekf = propagate(model.eom, (tc_eff, float(tf)), x_ekf0,
                        t_eval=t_post, rtol=1e-11, atol=1e-13)
    miss_ekf = _norm(res_ekf.x[-1, :3] - r_target)

    dv_perfect_mag = _norm(dv_perf)
    dv_ekf_mag     = _norm(dv_ekf)
    dv_delta_mag   = _norm(dv_ekf - dv_perf)
    # Signed difference of burn magnitudes. Renamed from "dv_inflation" to
    # prevent confusion with the fractional dv_inflation_pct ratio metric.
    dv_mag_bias    = dv_ekf_mag - dv_perfect_mag

    # Parallax: angular sweep of LOS-to-Moon. Net = endpoint-to-endpoint
    # angle (good for short monotonic arcs); cumulative = sum of
    # step-to-step angle deltas (correct under multi-rev / oscillatory
    # LOS geometry where the net under-counts the information-bearing
    # angular path).
    los_inertial_arr = np.asarray(los_inertial_hist)
    if los_inertial_arr.size >= 2 \
        and np.all(np.isfinite(los_inertial_arr[0])) \
        and np.all(np.isfinite(los_inertial_arr[-1])):
        cos_net = float(np.clip(
            np.dot(los_inertial_arr[0], los_inertial_arr[-1]), -1.0, 1.0
        ))
        parallax_net_rad = float(np.arccos(cos_net))
        dots = np.einsum("ij,ij->i", los_inertial_arr[:-1], los_inertial_arr[1:])
        dots = np.clip(dots, -1.0, 1.0)
        parallax_cumulative_rad = float(np.sum(np.arccos(dots)))
    else:
        parallax_net_rad        = float("nan")
        parallax_cumulative_rad = float("nan")
    # Backwards-compat: parallax_total_rad keeps its old name (= net) so
    # already-written 06m plots/CSVs don't break.
    parallax_total_rad = parallax_net_rad
    range_err_tc = float(abs(range_estimate_hist[-1] - range_truth_hist[-1])) \
        if range_truth_hist else float("nan")

    out: Dict[str, Any] = {
        "tc":           tc_eff,
        "sigma_px":     float(sigma_px),
        "dropout_prob": float(dropout_prob),
        "camera_mode":  camera_mode,
        "q_acc":        q_acc,
        "sigma_att_rad": float(sigma_att_rad),
        "P0_scale":      float(P0_scale),
        "n_landmarks":   int(lmk_arr.shape[0]),
        "pointing_lag_steps": int(pointing_lag_steps),
        "meas_delay_steps":   int(meas_delay_steps),
        "dv_perfect_mag":   dv_perfect_mag,
        "dv_ekf_mag":       dv_ekf_mag,
        "dv_delta_mag":     dv_delta_mag,
        "dv_mag_bias":      dv_mag_bias,
        "dv_inflation_pct": (
            float("nan") if dv_perfect_mag == 0.0
            else dv_ekf_mag / dv_perfect_mag - 1.0
        ),
        "miss_uncorrected": miss_unc,
        "miss_perfect":     miss_perf,
        "miss_ekf":         miss_ekf,
        "pos_err_tc":       pos_err_tc,
        "tracePpos_tc":     tracePpos_tc,
        # valid_rate (legacy name) is now epoch-level: any accepted
        # update at this epoch counts. valid_rate_moon and
        # valid_rate_landmarks split it for the landmark comparison.
        "valid_rate":       valid_rate,
        "valid_rate_moon":  valid_rate_moon,
        "valid_rate_landmarks": valid_rate_landmarks,
        # NIS: legacy `nis_mean` is moon-only (matches old plots);
        # `nis_mean_all` mixes moon + landmarks; `nis_mean_landmarks`
        # is landmark-only.
        "nis_mean":            nis_mean,
        "nis_mean_all":        nis_mean_all,
        "nis_mean_landmarks":  nis_mean_landmarks,
        "nees_mean":        nees_mean,
        "parallax_net_rad":        parallax_net_rad,
        "parallax_cumulative_rad": parallax_cumulative_rad,
        "parallax_total_rad":      parallax_total_rad,  # legacy alias
        "range_err_tc":       range_err_tc,
        # Per-source median angular offset from the commanded boresight
        # (radians). Used by the 06r landmarks-under-pointing-degradation
        # driver to distinguish "source stayed visible" from "source
        # forced near FOV edge / out of frame". NaN where source is
        # behind camera; landmarks_med pools over (epochs × landmarks)
        # for compactness.
        "moon_offset_rad_med": float(
            np.nanmedian(moon_offset_rad_hist)
            if moon_offset_rad_hist
            and any(np.isfinite(o) for o in moon_offset_rad_hist)
            else float("nan")
        ),
        "landmark_offset_rad_med": float(
            np.nanmedian(np.concatenate(landmark_offset_rad_hist))
            if (
                landmark_offset_rad_hist
                and any(arr.size for arr in landmark_offset_rad_hist)
                and np.any(np.isfinite(np.concatenate(landmark_offset_rad_hist)))
            )
            else float("nan")
        ),
        # Filter-ablation bookkeeping: which filter ran, total wall time,
        # per-call wall times in microseconds, and per-update mean iteration
        # count. For UKF, iters_used_mean is always 1 (single sigma-point
        # update); for EKF it is always 1 (max_iterations=1); for IEKF it is
        # the trial mean of bearing_update_tangent's reported iteration
        # counter.
        "filter_kind":         filter_kind,
        "iekf_max_iter":       int(iekf_max_iter),
        "t_trial_total_s":     float(time.perf_counter() - t_trial_start_s),
        "t_predict_total_s":   float(t_predict_total_s),
        "t_update_total_s":    float(t_update_total_s),
        "t_predict_mean_us":   float(
            1e6 * t_predict_total_s / n_predict_calls
        ) if n_predict_calls else float("nan"),
        "t_update_mean_us":    float(
            1e6 * t_update_total_s / n_update_calls
        ) if n_update_calls else float("nan"),
        "n_predict_calls":     int(n_predict_calls),
        "n_update_calls":      int(n_update_calls),
        "iters_used_mean":     float(iters_used_total / n_update_calls)
        if n_update_calls else float("nan"),
    }
    if not return_debug:
        return out

    x_hat_arr    = np.asarray(x_hat_hist)
    innov_2d_arr = np.asarray(innov_2d_list)
    pos_err_arr  = np.asarray(pos_err_list)
    P_diag_arr   = np.asarray(P_diag_list)
    gramian_eig_arr = (
        np.asarray(gramian_eig_hist) if accumulate_gramian else np.empty((0, 6))
    )
    P_full_arr = (
        np.asarray(P_full_hist) if P_cov_history else np.empty((0, 6, 6))
    )
    range_truth_arr    = np.asarray(range_truth_hist)
    range_estimate_arr = np.asarray(range_estimate_hist)

    out["debug"] = {
            "t_meas":        t_meas,
            "k_tc":          k_tc,
            "xs_nom":        res_nom.x,
            "xs_true":       xs_true,
            "x_hat_hist":    x_hat_arr,
            "pos_err_hist":  pos_err_arr,
            "P_diag_hist":   P_diag_arr,
            "P_full_hist":   P_full_arr,
            "P_tc":          P.copy(),
            "nis_hist":      nis_arr,           # moon-only (NaN where no moon update)
            "nees_hist":     nees_arr,
            "innov_2d_hist":    innov_2d_arr,
            "W_obs":            W_obs,
            "W_obs_moon":       W_obs_moon,
            "W_obs_landmarks":  W_obs_landmarks,
            "gramian_eig_hist": gramian_eig_arr,
            "los_inertial_hist": los_inertial_arr,
            "range_truth_hist":    range_truth_arr,
            "range_estimate_hist": range_estimate_arr,
            "accepted_moon_arr":      accepted_moon_arr,
            "accepted_landmarks_arr": accepted_landmarks_arr,
            "epoch_accepted_arr":     epoch_accepted_arr,
            "xs_unc_tf":    res_unc.x,
            "xs_perf_tf":   res_perf.x,
            "xs_ekf_tf":    res_ekf.x,
            "dv_perf":      dv_perf,
            "dv_ekf":       dv_ekf,
            "r_target":     r_target,
    }
    return out


def run_case_spice(
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
    kernels=None,
    epoch: str = _EM_EPOCH,
    camera_mode: CameraMode = "estimate_tracking",
    targets: tuple = ("SUN", "EARTH", "MOON"),
    q_acc_nd: float = 1e-14,
    sigma_att_rad: float = 0.0,
    bias_att_rad: Any = None,
    pointing_lag_steps: int = 0,
    meas_delay_steps: int = 0,
    P0_scale: float = 1.0,
    landmark_offsets_km: Any = None,
    disable_moon_center: bool = False,
    return_debug: bool = True,
    accumulate_gramian: bool = True,
    P_cov_history: bool = False,
) -> Dict[str, Any]:
    """High-fidelity SPICE/JPL-ephemeris variant of run_case.

    Accepts the same dimensionless CR3BP parameters as run_case and converts
    them to km / km·s / seconds using the SPICE-derived Earth-Moon scale
    factors at *epoch*.  Returns the same dict structure; miss/pos-error
    quantities are in **km** (not dimensionless CR3BP units).
    """
    # Lazy imports — callers without spiceypy can still use run_case.
    try:
        from dynamics.spice_ephemeris import make_spice_point_mass_dynamics
        from orbits.spice_bridge import (
            earth_moon_synodic_frame_from_spice,
            dimensional_synodic_to_spice_inertial_state,
        )
        from orbits.conversion import normalized_to_dimensional_state
        from orbits.types import CR3BPSystemUnits
    except ImportError as exc:
        raise ImportError(
            "run_case_spice requires 'spiceypy'. "
            "Install with: pip install cislunar-optical-nav[high-fidelity]"
        ) from exc

    camera_mode = _resolve_camera_mode(camera_mode)
    rng = np.random.default_rng(int(seed))

    bias_vec = (
        np.zeros(3, dtype=float)
        if bias_att_rad is None
        else np.asarray(bias_att_rad, dtype=float).reshape(3)
    )
    R_bias = small_rotation_dcm(bias_vec) if np.any(bias_vec != 0.0) else None

    if landmark_offsets_km is None:
        lmk_offsets_km = np.zeros((0, 3), dtype=float)
    else:
        lmk_offsets_km = np.asarray(landmark_offsets_km, dtype=float).reshape(-1, 3)

    pointing_lag_steps = max(0, int(pointing_lag_steps))
    meas_delay_steps   = max(0, int(meas_delay_steps))

    if kernels is None:
        kernels = _DEFAULT_KERNELS
    kernels = [Path(k) for k in kernels]

    # ── SPICE dynamics setup ─────────────────────────────────────────────────
    ephemeris, dynamics = make_spice_point_mass_dynamics(
        kernels=kernels,
        epoch=epoch,
        targets=list(targets),
    )
    try:
        # ── Dimensional scale factors from SPICE at epoch ────────────────────
        r_earth_0  = ephemeris.position_km("EARTH", 0.0)
        r_moon_0   = ephemeris.position_km("MOON",  0.0)
        lunit_km   = float(np.linalg.norm(r_moon_0 - r_earth_0))
        tunit_s    = float(np.sqrt(lunit_km ** 3 / (_GM_EARTH_KM3_S2 + _GM_MOON_KM3_S2)))
        vunit_km_s = lunit_km / tunit_s

        system = CR3BPSystemUnits(
            name="earth-moon-spice",
            mass_ratio=float(mu),
            radius_secondary_km=1737.4,
            lunit_km=lunit_km,
            tunit_s=tunit_s,
            libration_points={},
        )

        # ── Convert ND times → seconds ───────────────────────────────────────
        t0_s  = float(t0)      * tunit_s
        tf_s  = float(tf)      * tunit_s
        tc_s  = float(tc)      * tunit_s
        dtm_s = float(dt_meas) * tunit_s

        # ── Nominal IC: CR3BP L1 halo seed → J2000 inertial km ──────────────
        cr3bp_model   = CR3BP(mu=float(mu))
        L1x           = cr3bp_model.lagrange_points()["L1"][0]
        x0_nd         = np.array([L1x - 1e-3, 0.0, 0.0, 0.0, 0.05, 0.0], dtype=float)
        synodic_frame = earth_moon_synodic_frame_from_spice(
            ephemeris, t_s=t0_s, mass_ratio=float(mu)
        )
        state_dim = normalized_to_dimensional_state(x0_nd, system)
        x0_nom = np.asarray(
            dimensional_synodic_to_spice_inertial_state(state_dim, synodic_frame),
            dtype=float,
        )

        # Scale injection / estimation errors from ND → km / km·s
        dx0_km     = np.asarray(dx0,     dtype=float).reshape(6).copy()
        est_err_km = np.asarray(est_err, dtype=float).reshape(6).copy()
        dx0_km[:3]     *= lunit_km;   dx0_km[3:]     *= vunit_km_s
        est_err_km[:3] *= lunit_km;   est_err_km[3:] *= vunit_km_s

        x0_true = x0_nom + dx0_km

        # ── Measurement time grid ────────────────────────────────────────────
        # np.arange can drift slightly past tf_s for long arcs (the last
        # multiple of dtm_s falls just above tf_s due to FP accumulation,
        # and `+1e-6` lets it through). Clamp to t_span for solve_ivp.
        t_meas_s = np.arange(t0_s, tf_s + 1e-6, dtm_s)
        t_meas_s = t_meas_s[t_meas_s <= tf_s]
        if t_meas_s.size == 0 or t_meas_s[-1] < tf_s - dtm_s * 1e-3:
            t_meas_s = np.append(t_meas_s, tf_s)

        # ── Nominal propagation → r_target (J2000 km) ────────────────────────
        res_nom = propagate(
            dynamics.eom, (t0_s, tf_s), x0_nom,
            t_eval=np.linspace(t0_s, tf_s, 2001),
            rtol=1e-10, atol=1e-12,
        )
        if not res_nom.success:
            raise RuntimeError(f"Nominal SPICE propagation failed: {res_nom.message}")
        r_target = res_nom.x[-1, :3].copy()

        # ── Truth propagation on measurement grid ─────────────────────────────
        res_true = propagate(
            dynamics.eom, (t0_s, tf_s), x0_true,
            t_eval=t_meas_s, rtol=1e-10, atol=1e-12,
        )
        if not res_true.success:
            raise RuntimeError(f"Truth SPICE propagation failed: {res_true.message}")
        xs_true = res_true.x

        intr, R_fixed = _make_camera()
        k_tc   = _nearest_index(t_meas_s, tc_s)
        tc_eff = float(t_meas_s[k_tc])

        # ── Initial EKF state / covariance (ND σ scaled to km) ───────────────
        x_hat = x0_nom + est_err_km
        lsq   = lunit_km ** 2
        vsq   = vunit_km_s ** 2
        P     = np.diag([
            1e-6 * lsq, 1e-6 * lsq, 1e-6 * lsq,
            1e-7 * vsq, 1e-7 * vsq, 1e-7 * vsq,
        ]).astype(float)
        P    *= float(P0_scale)
        # Process-noise density in km²/s³ (proportionally converted from ND q_acc_nd)
        q_acc = float(q_acc_nd) * lsq / tunit_s ** 3

        # ── EKF loop ─────────────────────────────────────────────────────────
        x_hat_hist:    list[np.ndarray] = []
        nis_list:      list[float]      = []
        nis_all_per_epoch: list[list[float]] = []
        nis_landmarks_per_epoch: list[list[float]] = []
        nees_list:     list[float]      = []
        innov_2d_list: list[np.ndarray] = []
        pos_err_list:  list[float]      = []
        P_diag_list:   list[np.ndarray] = []
        P_full_hist:   list[np.ndarray] = []
        x_hat_pre_buf: list[np.ndarray] = []
        los_inertial_hist: list[np.ndarray] = []
        range_truth_hist:    list[float] = []
        range_estimate_hist: list[float] = []
        accepted_moon_arr      = np.zeros(k_tc + 1, dtype=bool)
        accepted_landmarks_arr = np.zeros(k_tc + 1, dtype=int)
        epoch_accepted_arr     = np.zeros(k_tc + 1, dtype=bool)
        Phi_cum         = np.eye(6, dtype=float) if accumulate_gramian else None
        W_obs           = np.zeros((6, 6), dtype=float) if accumulate_gramian else None
        W_obs_moon      = np.zeros((6, 6), dtype=float) if accumulate_gramian else None
        W_obs_landmarks = np.zeros((6, 6), dtype=float) if accumulate_gramian else None
        gramian_eig_hist: list[np.ndarray] = []

        for k in range(1, k_tc + 1):
            x_hat, P, Phi_step = ekf_propagate_stm(
                dynamics=dynamics,
                x=x_hat, P=P,
                t0=float(t_meas_s[k - 1]), t1=float(t_meas_s[k]),
                q_acc=q_acc, rtol=1e-10, atol=1e-12,
            )
            if accumulate_gramian:
                Phi_cum = Phi_step @ Phi_cum

            # Moon J2000 position at this measurement step (km).
            # When meas_delay_steps > 0, the synthetic image is built
            # with the spacecraft *and* Moon positions at the delayed
            # epoch, so the bearing is geometrically self-consistent —
            # the filter then ingests it at the current timestamp.
            t_k        = float(t_meas_s[k])
            k_meas     = max(0, k - meas_delay_steps)
            t_meas_eff = float(t_meas_s[k_meas])
            r_body          = ephemeris.position_km("MOON", t_k)
            r_body_at_meas  = (
                r_body if k_meas == k
                else ephemeris.position_km("MOON", t_meas_eff)
            )
            r_sc_true = xs_true[k_meas, :3]

            x_hat_pre_buf.append(x_hat.copy())
            if camera_mode == "estimate_tracking":
                if pointing_lag_steps > 0 and len(x_hat_pre_buf) > pointing_lag_steps:
                    x_hat_for_pointing = x_hat_pre_buf[-1 - pointing_lag_steps]
                else:
                    x_hat_for_pointing = x_hat
                # The camera commands a pointing toward the *currently
                # believed* Moon location (r_body @ t_k), even when the
                # measurement geometry is taken at the delayed epoch.
                R_cam_cmd = camera_dcm_from_boresight(
                    r_body - x_hat_for_pointing[:3], camera_forward_axis="+z"
                )
            elif camera_mode == "truth_tracking":
                R_cam_cmd = camera_dcm_from_boresight(
                    r_body_at_meas - r_sc_true, camera_forward_axis="+z"
                )
            else:
                R_cam_cmd = R_fixed

            R_cam_actual = R_cam_cmd
            if R_bias is not None:
                R_cam_actual = R_bias @ R_cam_actual
            if sigma_att_rad > 0.0:
                R_perturb = random_small_rotation_dcm(rng, float(sigma_att_rad))
                R_cam_actual = R_perturb @ R_cam_actual

            nis_all_this_epoch:       list[float] = []
            nis_landmarks_this_epoch: list[float] = []
            moon_accepted = False

            if not disable_moon_center:
                # Truth bearing geometry uses the delayed Moon position;
                # the filter still hypothesizes r_body @ t_k, so any
                # delay surfaces as un-modeled bearing residual.
                meas = simulate_pixel_measurement(
                    r_sc=r_sc_true, r_body=r_body_at_meas, intrinsics=intr,
                    R_cam_from_frame=R_cam_actual, sigma_px=float(sigma_px),
                    rng=rng, t=t_k,
                    dropout_p=float(dropout_prob), out_of_frame="drop", behind="drop",
                )

                if meas.valid and np.isfinite(meas.u_px):
                    u_g, sig_k = pixel_detection_to_bearing(
                        meas.u_px, meas.v_px, float(sigma_px), intr, R_cam_cmd.T
                    )
                    if np.all(np.isfinite(u_g)):
                        upd = bearing_update_tangent(x_hat, P, u_g, r_body, float(sig_k))
                        if upd.accepted:
                            x_hat, P = upd.x_upd, upd.P_upd
                            moon_accepted = True
                            if accumulate_gramian:
                                HtH = upd.H.T @ upd.H
                                W_obs      += Phi_cum.T @ HtH @ Phi_cum
                                W_obs_moon += Phi_cum.T @ HtH @ Phi_cum
                        nis_list.append(float(upd.nis))
                        nis_all_this_epoch.append(float(upd.nis))
                        innov_2d_list.append(
                            upd.final_innovation.copy()
                            if upd.final_innovation is not None
                            else upd.innovation.copy()
                        )
                    else:
                        nis_list.append(float("nan"))
                        innov_2d_list.append(np.full(2, np.nan))
                else:
                    nis_list.append(float("nan"))
                    innov_2d_list.append(np.full(2, np.nan))
            else:
                nis_list.append(float("nan"))
                innov_2d_list.append(np.full(2, np.nan))

            # Landmarks: Moon-fixed offsets added to the *current* Moon
            # ephemeris position for the filter, but the truth image is
            # generated against the delayed Moon position so the geometry
            # is consistent with meas_delay_steps.
            n_lmk_accepted = 0
            for off_km in lmk_offsets_km:
                lmk_pos_meas   = r_body_at_meas + off_km
                lmk_pos_filter = r_body         + off_km
                lm_meas = simulate_pixel_measurement(
                    r_sc=r_sc_true, r_body=lmk_pos_meas, intrinsics=intr,
                    R_cam_from_frame=R_cam_actual, sigma_px=float(sigma_px),
                    rng=rng, t=t_k,
                    dropout_p=float(dropout_prob), out_of_frame="drop", behind="drop",
                )
                if not (lm_meas.valid and np.isfinite(lm_meas.u_px)):
                    continue
                u_l, sig_l = pixel_detection_to_bearing(
                    lm_meas.u_px, lm_meas.v_px, float(sigma_px), intr, R_cam_cmd.T
                )
                if not np.all(np.isfinite(u_l)):
                    continue
                upd_l = bearing_update_tangent(
                    x_hat, P, u_l, lmk_pos_filter, float(sig_l)
                )
                if upd_l.accepted:
                    x_hat, P = upd_l.x_upd, upd_l.P_upd
                    n_lmk_accepted += 1
                    if accumulate_gramian:
                        HtH_l = upd_l.H.T @ upd_l.H
                        W_obs           += Phi_cum.T @ HtH_l @ Phi_cum
                        W_obs_landmarks += Phi_cum.T @ HtH_l @ Phi_cum
                nis_all_this_epoch.append(float(upd_l.nis))
                nis_landmarks_this_epoch.append(float(upd_l.nis))

            accepted_moon_arr[k]      = moon_accepted
            accepted_landmarks_arr[k] = n_lmk_accepted
            epoch_accepted_arr[k]     = bool(moon_accepted) or (n_lmk_accepted > 0)
            nis_all_per_epoch.append(nis_all_this_epoch)
            nis_landmarks_per_epoch.append(nis_landmarks_this_epoch)

            err6 = x_hat - xs_true[k]
            try:
                nees_val = float(err6 @ np.linalg.solve(P, err6))
            except np.linalg.LinAlgError:
                nees_val = float("nan")
            nees_list.append(nees_val)

            x_hat_hist.append(x_hat.copy())
            pos_err_list.append(_norm(x_hat[:3] - xs_true[k, :3]))
            P_diag_list.append(np.diag(P).copy())
            if P_cov_history:
                P_full_hist.append(P.copy())
            rho_truth = r_body - xs_true[k, :3]
            rho_est   = r_body - x_hat[:3]
            n_truth = float(np.linalg.norm(rho_truth))
            n_est   = float(np.linalg.norm(rho_est))
            los_inertial_hist.append(
                (rho_truth / n_truth) if n_truth > 1e-12 else np.full(3, np.nan)
            )
            range_truth_hist.append(n_truth)
            range_estimate_hist.append(n_est)
            if accumulate_gramian:
                gramian_eig_hist.append(np.linalg.eigvalsh(W_obs).copy())

        nis_arr         = np.asarray(nis_list)
        nees_arr        = np.asarray(nees_list)

        x_true_tc    = xs_true[k_tc]
        x_hat_tc     = x_hat.copy()
        P_tc         = P.copy()
        pos_err_tc   = _norm(x_hat_tc[:3] - x_true_tc[:3])
        tracePpos_tc = float(np.trace(P_tc[:3, :3]))

        n_epochs = k_tc
        valid_rate = (
            float(np.mean(epoch_accepted_arr[1:k_tc + 1]))
            if n_epochs > 0 else float("nan")
        )
        valid_rate_moon = (
            float(np.mean(accepted_moon_arr[1:k_tc + 1]))
            if n_epochs > 0 else float("nan")
        )
        n_lmk = int(lmk_offsets_km.shape[0])
        if n_lmk > 0 and n_epochs > 0:
            valid_rate_landmarks = float(
                np.mean(accepted_landmarks_arr[1:k_tc + 1] / float(n_lmk))
            )
        else:
            valid_rate_landmarks = float("nan")

        nis_finite   = nis_arr[np.isfinite(nis_arr)]
        nis_mean     = float(np.mean(nis_finite)) if nis_finite.size else float("nan")
        flat_all = np.array(
            [v for ep in nis_all_per_epoch for v in ep if np.isfinite(v)],
            dtype=float,
        )
        nis_mean_all = float(np.mean(flat_all)) if flat_all.size else float("nan")
        flat_lmk = np.array(
            [v for ep in nis_landmarks_per_epoch for v in ep if np.isfinite(v)],
            dtype=float,
        )
        nis_mean_landmarks = (
            float(np.mean(flat_lmk)) if flat_lmk.size else float("nan")
        )
        nees_finite  = nees_arr[np.isfinite(nees_arr)]
        nees_mean    = float(np.mean(nees_finite)) if nees_finite.size else float("nan")

        # ── Targeting ──────────────────────────────────────────────────────────
        result_perf = solve_single_impulse_position_target(
            propagate=propagate, dynamics=dynamics, x0=x_true_tc,
            t0=tc_eff, tc=tc_eff, tf=tf_s, r_target=r_target,
        )
        result_ekf = solve_single_impulse_position_target(
            propagate=propagate, dynamics=dynamics, x0=x_hat_tc,
            t0=tc_eff, tc=tc_eff, tf=tf_s, r_target=r_target,
        )

        dv_perf = np.asarray(result_perf.dv, dtype=float)
        dv_ekf  = np.asarray(result_ekf.dv,  dtype=float)

        t_post   = np.linspace(tc_eff, tf_s, 2001)
        res_unc  = propagate(dynamics.eom, (tc_eff, tf_s), x_true_tc,
                             t_eval=t_post, rtol=1e-10, atol=1e-12)
        miss_unc = _norm(res_unc.x[-1, :3] - r_target)

        x_perf0 = x_true_tc.copy(); x_perf0[3:6] += dv_perf
        res_perf = propagate(dynamics.eom, (tc_eff, tf_s), x_perf0,
                             t_eval=t_post, rtol=1e-10, atol=1e-12)
        miss_perf = _norm(res_perf.x[-1, :3] - r_target)

        x_ekf0 = x_true_tc.copy(); x_ekf0[3:6] += dv_ekf
        res_ekf = propagate(dynamics.eom, (tc_eff, tf_s), x_ekf0,
                            t_eval=t_post, rtol=1e-10, atol=1e-12)
        miss_ekf = _norm(res_ekf.x[-1, :3] - r_target)

        dv_perfect_mag = _norm(dv_perf)
        dv_ekf_mag     = _norm(dv_ekf)
        dv_delta_mag   = _norm(dv_ekf - dv_perf)
        dv_mag_bias    = dv_ekf_mag - dv_perfect_mag

        los_inertial_arr = np.asarray(los_inertial_hist)
        if los_inertial_arr.size >= 2 \
            and np.all(np.isfinite(los_inertial_arr[0])) \
            and np.all(np.isfinite(los_inertial_arr[-1])):
            cos_net = float(np.clip(
                np.dot(los_inertial_arr[0], los_inertial_arr[-1]), -1.0, 1.0
            ))
            parallax_net_rad = float(np.arccos(cos_net))
            dots = np.einsum("ij,ij->i",
                             los_inertial_arr[:-1], los_inertial_arr[1:])
            dots = np.clip(dots, -1.0, 1.0)
            parallax_cumulative_rad = float(np.sum(np.arccos(dots)))
        else:
            parallax_net_rad        = float("nan")
            parallax_cumulative_rad = float("nan")
        parallax_total_rad = parallax_net_rad
        range_err_tc = float(abs(range_estimate_hist[-1] - range_truth_hist[-1])) \
            if range_truth_hist else float("nan")

        out: Dict[str, Any] = {
            "tc":             tc_eff,
            "sigma_px":       float(sigma_px),
            "dropout_prob":   float(dropout_prob),
            "camera_mode":    camera_mode,
            "q_acc_nd":       float(q_acc_nd),
            "sigma_att_rad":  float(sigma_att_rad),
            "P0_scale":       float(P0_scale),
            "n_landmarks":    int(lmk_offsets_km.shape[0]),
            "pointing_lag_steps": int(pointing_lag_steps),
            "meas_delay_steps":   int(meas_delay_steps),
            "units":          "km/km_s",
            "lunit_km":       lunit_km,
            "tunit_s":        tunit_s,
            "dv_perfect_mag": dv_perfect_mag,    # km/s
            "dv_ekf_mag":     dv_ekf_mag,         # km/s
            "dv_delta_mag":   dv_delta_mag,       # km/s
            "dv_mag_bias":    dv_mag_bias,        # km/s (signed)
            "dv_inflation_pct": (
                float("nan") if dv_perfect_mag == 0.0
                else dv_ekf_mag / dv_perfect_mag - 1.0
            ),
            "miss_uncorrected": miss_unc,    # km
            "miss_perfect":     miss_perf,   # km
            "miss_ekf":         miss_ekf,    # km
            "pos_err_tc":       pos_err_tc,  # km
            "tracePpos_tc":     tracePpos_tc,
            "valid_rate":           valid_rate,
            "valid_rate_moon":      valid_rate_moon,
            "valid_rate_landmarks": valid_rate_landmarks,
            "nis_mean":             nis_mean,
            "nis_mean_all":         nis_mean_all,
            "nis_mean_landmarks":   nis_mean_landmarks,
            "nees_mean":        nees_mean,
            "parallax_net_rad":        parallax_net_rad,
            "parallax_cumulative_rad": parallax_cumulative_rad,
            "parallax_total_rad":      parallax_total_rad,  # legacy alias
            "range_err_tc":       range_err_tc,
        }
        if not return_debug:
            return out

        x_hat_arr       = np.asarray(x_hat_hist)
        innov_2d_arr    = np.asarray(innov_2d_list)
        pos_err_arr     = np.asarray(pos_err_list)
        P_diag_arr      = np.asarray(P_diag_list)
        gramian_eig_arr = (
            np.asarray(gramian_eig_hist) if accumulate_gramian else np.empty((0, 6))
        )
        P_full_arr = (
            np.asarray(P_full_hist) if P_cov_history else np.empty((0, 6, 6))
        )
        range_truth_arr    = np.asarray(range_truth_hist)
        range_estimate_arr = np.asarray(range_estimate_hist)

        out["debug"] = {
                "t_meas":           t_meas_s,
                "k_tc":             k_tc,
                "xs_nom":           res_nom.x,
                "xs_true":          xs_true,
                "x_hat_hist":       x_hat_arr,
                "pos_err_hist":     pos_err_arr,
                "P_diag_hist":      P_diag_arr,
                "P_full_hist":      P_full_arr,
                "P_tc":             P.copy(),
                "nis_hist":         nis_arr,
                "nees_hist":        nees_arr,
                "innov_2d_hist":    innov_2d_arr,
                "W_obs":            W_obs,
                "W_obs_moon":       W_obs_moon,
                "W_obs_landmarks":  W_obs_landmarks,
                "gramian_eig_hist": gramian_eig_arr,
                "los_inertial_hist":   los_inertial_arr,
                "range_truth_hist":    range_truth_arr,
                "range_estimate_hist": range_estimate_arr,
                "accepted_moon_arr":      accepted_moon_arr,
                "accepted_landmarks_arr": accepted_landmarks_arr,
                "epoch_accepted_arr":     epoch_accepted_arr,
                "xs_unc_tf":        res_unc.x,
                "xs_perf_tf":       res_perf.x,
                "xs_ekf_tf":        res_ekf.x,
                "dv_perf":          dv_perf,
                "dv_ekf":           dv_ekf,
                "r_target":         r_target,
                "lunit_km":         lunit_km,
                "tunit_s":          tunit_s,
        }
        return out
    finally:
        ephemeris.close()


def main() -> None:
    _apply_dark_theme()
    plots_dir = Path("results/mc/single_trial")
    _ensure_dir(plots_dir)

    mu, t0, tf, tc = 0.0121505856, 0.0, 6.0, 2.0
    dt_meas, sigma_px, dropout_prob, seed = 0.02, 1.5, 0.0, 7
    dx0     = np.array([1e-4, -1e-4, 0.0, 0.0, 0.0, 0.0], dtype=float)
    est_err = np.array([1e-4,  1e-4, 0.0, 0.0, 0.0, 0.0], dtype=float)

    print("Running 06 midcourse EKF correction ...")
    out = run_case(mu, t0, tf, tc, dt_meas, sigma_px, dropout_prob, seed,
                   dx0, est_err, camera_mode="estimate_tracking", q_acc=1e-9)
    dbg = out["debug"]

    t_meas  = dbg["t_meas"]
    k_tc    = dbg["k_tc"]
    t_ekf   = t_meas[1: k_tc + 1]
    xs_nom  = dbg["xs_nom"]
    xs_true = dbg["xs_true"]
    x_hat   = dbg["x_hat_hist"]
    pos_err = dbg["pos_err_hist"]
    P_diag  = dbg["P_diag_hist"]
    nis     = dbg["nis_hist"]
    nees         = dbg["nees_hist"]
    innov2d      = dbg["innov_2d_hist"]
    W_obs        = dbg["W_obs"]
    gramian_eig  = dbg["gramian_eig_hist"]   # (k_tc, 6), ascending eigenvalues
    xs_unc  = dbg["xs_unc_tf"]
    xs_perf = dbg["xs_perf_tf"]
    xs_ekf  = dbg["xs_ekf_tf"]
    r_tgt   = dbg["r_target"]
    sig_pos = 3.0 * np.sqrt(np.abs(P_diag[:, 0]))

    print(f"  camera_mode   = {out['camera_mode']}")
    print(f"  |dv| perfect  = {out['dv_perfect_mag']:.4e} dimensionless CR3BP velocity")
    print(f"  |dv| EKF      = {out['dv_ekf_mag']:.4e} dimensionless CR3BP velocity")
    print(f"  miss_ekf      = {out['miss_ekf']:.4e} dimensionless CR3BP length")
    print(f"  NIS mean      = {out['nis_mean']:.3f}  (expected ≈ 2 for consistent 2-D bearing)")
    print(f"  NEES mean     = {out['nees_mean']:.3f}  (expected ≈ 6 for consistent 6-D state)")
    print(f"  valid_rate    = {out['valid_rate']:.3f}")

    def _ax_style(ax: plt.Axes) -> None:
        ax.set_facecolor(_PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(_BORDER)
        ax.grid(True)

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            left=0.07, right=0.97, top=0.93, bottom=0.09,
                            wspace=0.28, hspace=0.38)

    ax = fig.add_subplot(gs[0, 0])
    _ax_style(ax)
    model_p = CR3BP(mu=mu)
    p1, p2 = model_p.primary1, model_p.primary2
    ax.scatter([p1[0]], [p1[1]], s=90, c=_EARTH_C, zorder=5, label="Earth")
    ax.scatter([p2[0]], [p2[1]], s=60, c=_MOON_C,  zorder=5, label="Moon")
    ax.plot(xs_nom[:, 0],  xs_nom[:, 1],  color=_CYAN,   lw=1.4, alpha=0.7, label="nominal")
    ax.plot(xs_true[:, 0], xs_true[:, 1], color=_AMBER,  lw=1.4, ls="--",   label="truth")
    ax.plot(xs_unc[:, 0],  xs_unc[:, 1],  color=_AMBER,  lw=1.0, ls=":",    alpha=0.55)
    ax.plot(xs_perf[:, 0], xs_perf[:, 1], color=_GREEN,  lw=1.6, label="perfect Δv")
    ax.plot(xs_ekf[:, 0],  xs_ekf[:, 1],  color=_VIOLET, lw=1.6, ls=(0,(5,3)), label="EKF Δv")
    ax.scatter([r_tgt[0]],            [r_tgt[1]],            s=90, marker="*", c=_AMBER, zorder=6, label="target")
    ax.scatter([xs_true[k_tc, 0]],    [xs_true[k_tc, 1]],    s=70, c=_RED,    zorder=6, label="tc")
    ax.set_title(f"XY Trajectory  [{out['camera_mode']}]", color=_TEXT)
    ax.set_xlabel("x  [dimensionless CR3BP length]", color=_TEXT)
    ax.set_ylabel("y  [dimensionless CR3BP length]", color=_TEXT)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    _ax_style(ax)
    ax.fill_between(t_ekf, 0, sig_pos, color=_VIOLET, alpha=0.15, label="3σ (x)")
    ax.semilogy(t_ekf, pos_err + 1e-12, color=_CYAN,  lw=1.8, label="‖r̂ − r‖")
    ax.semilogy(t_ekf,
                np.linalg.norm(x_hat[:, 3:6] - xs_true[1: k_tc + 1, 3:6], axis=1) + 1e-12,
                color=_AMBER, lw=1.5, ls="--", label="‖v̂ − v‖", alpha=0.85)
    ax.axvline(tc, color=_RED, lw=0.9, ls="--", alpha=0.6)
    ax.set_title("EKF State Errors (to tc)", color=_TEXT)
    ax.set_xlabel("t  [dimensionless CR3BP time]", color=_TEXT)
    ax.set_ylabel("Error  [dimensionless CR3BP units]", color=_TEXT)
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, 0])
    _ax_style(ax)
    nis_lo = chi2.ppf(0.025, df=2)
    nis_hi = chi2.ppf(0.975, df=2)
    ax.fill_between(t_ekf, nis_lo, nis_hi, color=_GREEN, alpha=0.10,
                    label=f"95% χ²(2): [{nis_lo:.2f}, {nis_hi:.2f}]")
    ax.axhline(2.0, color=_GREEN, lw=0.8, ls="--", alpha=0.5)
    nis_ok = np.isfinite(nis)
    in_b   = nis_ok & (nis >= nis_lo) & (nis <= nis_hi)
    out_b  = nis_ok & ~in_b
    ax.scatter(t_ekf[in_b],  nis[in_b],  s=12, c=_GREEN, zorder=4)
    ax.scatter(t_ekf[out_b], nis[out_b], s=12, c=_RED,   zorder=4)
    ax.set_ylim(0, 16)
    ax.set_title(
        f"NIS  (mean = {out['nis_mean']:.2f},  valid = {out['valid_rate']:.2f})",
        color=_TEXT,
    )
    ax.set_xlabel("t  [dimensionless CR3BP time]", color=_TEXT)
    ax.set_ylabel("NIS", color=_TEXT)
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    _ax_style(ax)
    miss_labels = ["uncorrected", "perfect Δv", "EKF Δv"]
    miss_vals   = [out["miss_uncorrected"], out["miss_perfect"], out["miss_ekf"]]
    miss_cols   = [_AMBER, _GREEN, _VIOLET]
    bars = ax.bar(miss_labels, miss_vals, color=miss_cols, edgecolor=_BORDER, lw=0.8)
    for bar, val in zip(bars, miss_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + max(miss_vals) * 0.02,
                f"{val:.2e}", ha="center", va="bottom", color=_TEXT, fontsize=9)
    ax.set_title("Terminal Miss Distance  ‖r(tf) − r_target‖", color=_TEXT)
    ax.set_ylabel("Miss distance  [dimensionless CR3BP length]", color=_TEXT)
    ax.grid(True, axis="y")

    fig.suptitle(
        f"EKF Midcourse Correction — Bearing-Only Nav near L1  [{out['camera_mode']}]",
        color=_TEXT, fontsize=13, y=0.98,
    )
    fig.patch.set_facecolor(_BG)
    fig.savefig(plots_dir / "06_midcourse_report.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(_BORDER)
    labels = ["|dv| perfect", "|dv| EKF", "‖Δdv‖"]
    vals   = [out["dv_perfect_mag"], out["dv_ekf_mag"], out["dv_delta_mag"]]
    cols   = [_GREEN, _VIOLET, _RED]
    bars   = ax.bar(labels, vals, color=cols, edgecolor=_BORDER, lw=0.8, width=0.55)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(vals) * 0.02,
                f"{val:.3e}", ha="center", va="bottom", color=_TEXT, fontsize=10)
    ax.set_title("Burn Magnitude Comparison", color=_TEXT, fontsize=12)
    ax.set_ylabel("Δv  [dimensionless CR3BP velocity]", color=_TEXT)
    ax.grid(True, axis="y")
    infl = out["dv_inflation_pct"]
    if np.isfinite(infl):
        ax.text(0.98, 0.97, f"Inflation: {infl*100:+.2f}%",
                transform=ax.transAxes, ha="right", va="top",
                color=_ORANGE, fontsize=11,
                bbox=dict(facecolor=_PANEL, edgecolor=_BORDER, pad=5))
    fig.savefig(plots_dir / "06_dv_compare.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3: Filter consistency diagnostics ─────────────────────────────
    fig3 = plt.figure(figsize=(14, 9))
    gs3  = gridspec.GridSpec(2, 2, figure=fig3,
                             left=0.07, right=0.97, top=0.93, bottom=0.09,
                             wspace=0.30, hspace=0.42)

    # --- [0,0] NEES vs time (chi²(6) 95% CI) --------------------------------
    ax_nees = fig3.add_subplot(gs3[0, 0])
    _ax_style(ax_nees)
    nees_lo = chi2.ppf(0.025, df=6)
    nees_hi = chi2.ppf(0.975, df=6)
    ax_nees.fill_between(t_ekf, nees_lo, nees_hi, color=_GREEN, alpha=0.10,
                         label=f"95% χ²(6): [{nees_lo:.1f}, {nees_hi:.1f}]")
    ax_nees.axhline(6.0, color=_GREEN, lw=0.8, ls="--", alpha=0.5)
    nees_ok  = np.isfinite(nees)
    nees_inb = nees_ok & (nees >= nees_lo) & (nees <= nees_hi)
    nees_out = nees_ok & ~nees_inb
    ax_nees.scatter(t_ekf[nees_inb], nees[nees_inb], s=12, c=_GREEN, zorder=4)
    ax_nees.scatter(t_ekf[nees_out], nees[nees_out], s=12, c=_RED,   zorder=4)
    ax_nees.set_ylim(0, max(30.0, float(np.nanpercentile(nees, 99)) * 1.2))
    ax_nees.set_title(
        f"NEES  (mean={out['nees_mean']:.2f}, expected≈6)",
        color=_TEXT,
    )
    ax_nees.set_xlabel("t  [dimensionless CR3BP time]", color=_TEXT)
    ax_nees.set_ylabel("NEES", color=_TEXT)
    ax_nees.legend(fontsize=8)

    # --- [0,1] Covariance trace vs time --------------------------------------
    ax_cov = fig3.add_subplot(gs3[0, 1])
    _ax_style(ax_cov)
    tr_pos = P_diag[:, 0] + P_diag[:, 1] + P_diag[:, 2]
    tr_vel = P_diag[:, 3] + P_diag[:, 4] + P_diag[:, 5]
    tr_all = tr_pos + tr_vel
    ax_cov.semilogy(t_ekf, tr_pos + 1e-20, color=_CYAN,   lw=1.8, label="tr(P_pos)")
    ax_cov.semilogy(t_ekf, tr_vel + 1e-20, color=_AMBER,  lw=1.5, ls="--", label="tr(P_vel)")
    ax_cov.semilogy(t_ekf, tr_all + 1e-20, color=_VIOLET, lw=1.2, ls=":",  label="tr(P_full)", alpha=0.7)
    ax_cov.axvline(tc, color=_RED, lw=0.9, ls="--", alpha=0.6)
    ax_cov.set_title("Covariance Trace vs Time", color=_TEXT)
    ax_cov.set_xlabel("t  [dimensionless CR3BP time]", color=_TEXT)
    ax_cov.set_ylabel("tr(P)  [dimensionless CR3BP²]", color=_TEXT)
    ax_cov.legend(fontsize=9)

    # --- [1,0] Innovation ACF (whiteness test) --------------------------------
    ax_acf = fig3.add_subplot(gs3[1, 0])
    _ax_style(ax_acf)
    max_lag = min(30, len(innov2d) // 4)
    for comp_idx, (comp_color, comp_label) in enumerate(
        [(_CYAN, "innov[0]"), (_AMBER, "innov[1]")]
    ):
        vals_c = innov2d[:, comp_idx]
        valid_c = vals_c[np.isfinite(vals_c)]
        if valid_c.size < 4:
            continue
        vm = valid_c - valid_c.mean()
        var = float(np.dot(vm, vm))
        if var == 0.0:
            continue
        acf_lags = np.arange(0, max_lag + 1)
        acf_vals = np.array([
            float(np.dot(vm[:len(vm) - lag], vm[lag:])) / var
            if lag < len(vm) else 0.0
            for lag in acf_lags
        ])
        markerline, stemlines, _ = ax_acf.stem(
            acf_lags, acf_vals,
            linefmt=comp_color, markerfmt=f"o",
            basefmt=" ",
            label=comp_label,
        )
        markerline.set_color(comp_color)
        markerline.set_markersize(4)
        plt.setp(stemlines, color=comp_color, linewidth=1.0, alpha=0.7)
    # 95% confidence bounds for white noise: ±1.96/√N
    n_valid = int(np.sum(np.isfinite(innov2d[:, 0])))
    if n_valid > 0:
        ci_bound = 1.96 / np.sqrt(max(n_valid, 1))
        ax_acf.axhline( ci_bound, color=_GREEN, lw=1.0, ls="--", alpha=0.7, label=f"95% CI (±{ci_bound:.2f})")
        ax_acf.axhline(-ci_bound, color=_GREEN, lw=1.0, ls="--", alpha=0.7)
    ax_acf.axhline(0, color=_TEXT, lw=0.5, alpha=0.3)
    ax_acf.set_title("Innovation ACF  (whiteness test)", color=_TEXT)
    ax_acf.set_xlabel("Lag  [measurement steps]", color=_TEXT)
    ax_acf.set_ylabel("Autocorrelation", color=_TEXT)
    ax_acf.set_ylim(-1.1, 1.1)
    ax_acf.legend(fontsize=9)

    # --- [1,1] Measurement validity indicator --------------------------------
    ax_val = fig3.add_subplot(gs3[1, 1])
    _ax_style(ax_val)
    valid_mask = np.isfinite(nis)
    ax_val.bar(t_ekf[valid_mask],  np.ones(valid_mask.sum()),
               width=(t_ekf[1] - t_ekf[0]) * 0.9,
               color=_GREEN,  alpha=0.8, label="accepted")
    ax_val.bar(t_ekf[~valid_mask], np.ones((~valid_mask).sum()),
               width=(t_ekf[1] - t_ekf[0]) * 0.9,
               color=_RED,    alpha=0.6, label="dropped/invalid")
    ax_val.axvline(tc, color=_AMBER, lw=0.9, ls="--", alpha=0.6)
    ax_val.set_ylim(0, 1.5)
    ax_val.set_yticks([])
    ax_val.set_title(
        f"Measurement Validity  (accept rate = {out['valid_rate']:.2f})",
        color=_TEXT,
    )
    ax_val.set_xlabel("t  [dimensionless CR3BP time]", color=_TEXT)
    ax_val.legend(fontsize=9)

    fig3.suptitle(
        f"Filter Consistency Diagnostics  [{out['camera_mode']}]",
        color=_TEXT, fontsize=13, y=0.98,
    )
    fig3.patch.set_facecolor(_BG)
    fig3.savefig(plots_dir / "06_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig3)

    # ── Figure 4: Observability analysis ─────────────────────────────────────
    eigvals_final, eigvecs_final = np.linalg.eigh(W_obs)  # ascending order
    cond_num = float(eigvals_final[-1]) / max(float(eigvals_final[0]), 1e-30)

    fig4 = plt.figure(figsize=(14, 9))
    gs4  = gridspec.GridSpec(2, 2, figure=fig4,
                             left=0.07, right=0.97, top=0.93, bottom=0.09,
                             wspace=0.32, hspace=0.45)

    # --- [0,0] Final eigenvalue spectrum: reveals separation between
    #           observable (large λ) and unobservable (small λ) directions. ----
    ax_eig = fig4.add_subplot(gs4[0, 0])
    _ax_style(ax_eig)
    eig_labels = [f"λ{i+1}" for i in range(6)]
    eig_max    = float(max(eigvals_final.max(), 1e-30))
    eig_colors = [
        _RED   if v < eig_max * 1e-4 else
        _AMBER if v < eig_max * 1e-2 else
        _GREEN
        for v in eigvals_final
    ]
    ax_eig.bar(
        eig_labels,
        np.maximum(eigvals_final, 1e-30),
        color=eig_colors,
        edgecolor=_BORDER,
        lw=0.8,
    )
    ax_eig.set_yscale("log")
    ax_eig.set_title(
        f"Gramian Eigenvalue Spectrum  (cond = {cond_num:.2e})",
        color=_TEXT,
    )
    ax_eig.set_xlabel("Eigenvalue index  (λ₁ = least observable)", color=_TEXT)
    ax_eig.set_ylabel("λ  [dimensionless]", color=_TEXT)
    ax_eig.text(
        0.97, 0.03,
        f"λ_min = {eigvals_final[0]:.2e}\nλ_max = {eigvals_final[-1]:.2e}",
        transform=ax_eig.transAxes, ha="right", va="bottom",
        color=_TEXT, fontsize=8.5,
        bbox=dict(facecolor=_BG, edgecolor=_BORDER, pad=4),
    )

    # --- [0,1] Eigenvalue growth over the observation arc --------------------
    ax_evo = fig4.add_subplot(gs4[0, 1])
    _ax_style(ax_evo)
    evo_colors = [_RED, _AMBER, _ORANGE, _VIOLET, _CYAN, _GREEN]
    evo_styles = ["-", "--", ":", "-.", (0,(3,1)), (0,(5,2))]
    for i in range(6):
        eig_i = gramian_eig[:, i]
        pos_mask = eig_i > 0
        if pos_mask.sum() > 1:
            ax_evo.semilogy(
                t_ekf[pos_mask], eig_i[pos_mask],
                color=evo_colors[i], lw=1.8 if i in (0, 5) else 1.1,
                ls=evo_styles[i],
                alpha=1.0 if i in (0, 5) else 0.65,
                label=f"λ{i+1}",
            )
    ax_evo.axvline(tc, color=_RED, lw=0.9, ls="--", alpha=0.6)
    ax_evo.set_title("Gramian Eigenvalue Growth  (log scale)", color=_TEXT)
    ax_evo.set_xlabel("t  [dimensionless CR3BP time]", color=_TEXT)
    ax_evo.set_ylabel("λ  [dimensionless]", color=_TEXT)
    ax_evo.legend(fontsize=8, ncol=2)

    # --- [1,0] XY trajectory + unobservable direction arrow ------------------
    # The eigenvector for λ_min is the state-space direction bearing-only
    # measurements can barely distinguish.  Its position projection (first 2
    # components) shows the spatial direction of poor observability.
    ax_dir = fig4.add_subplot(gs4[1, 0])
    _ax_style(ax_dir)
    model_dir = CR3BP(mu=mu)
    ax_dir.plot(xs_true[:k_tc+1, 0], xs_true[:k_tc+1, 1],
                color=_AMBER, lw=1.6, ls="--", alpha=0.85, label="truth arc")
    # Moon inside the zoom window; Earth at x=0 is far off-frame and omitted
    # to keep the trajectory readable.
    ax_dir.scatter([model_dir.primary2[0]], [model_dir.primary2[1]],
                   s=55, c=_MOON_C, zorder=5, label="Moon")

    unobs_pos_xy = eigvecs_final[:2, 0]
    pos_norm = float(np.linalg.norm(unobs_pos_xy))
    if pos_norm > 1e-12:
        unobs_xy  = unobs_pos_xy / pos_norm
        traj_span = max(float(np.ptp(xs_true[:k_tc+1, 0])),
                        float(np.ptp(xs_true[:k_tc+1, 1])), 1e-10)
        arrow_len = 0.10 * traj_span
        mid_k = k_tc // 2
        ax0, ay0 = float(xs_true[mid_k, 0]), float(xs_true[mid_k, 1])
        ax_dir.annotate(
            "",
            xy=(ax0 + unobs_xy[0] * arrow_len, ay0 + unobs_xy[1] * arrow_len),
            xytext=(ax0 - unobs_xy[0] * arrow_len, ay0 - unobs_xy[1] * arrow_len),
            arrowprops=dict(arrowstyle="<->", color=_RED, lw=2.2),
        )
        # Label pinned to the top-right corner (axes fraction) so it never
        # collides with the trajectory.
        ax_dir.text(
            0.97, 0.97,
            f"least-obs. direction\n({unobs_xy[0]:+.2f}, {unobs_xy[1]:+.2f})",
            transform=ax_dir.transAxes,
            color=_RED, fontsize=8, ha="right", va="top",
            bbox=dict(facecolor=_BG, edgecolor=_BORDER, alpha=0.85, pad=4),
        )

    # "equal" aspect with datalim keeps the subplot rectangle at its grid
    # size — "box" was compressing the whole subplot when the x- and y-ranges
    # differed by orders of magnitude, making the title unreadable.
    ax_dir.set_aspect("equal", adjustable="datalim")
    ax_dir.set_title(
        "Unobservable Direction  (λ_min eigvec, position projection)",
        color=_TEXT,
    )
    ax_dir.set_xlabel("x  [dimensionless CR3BP length]", color=_TEXT)
    ax_dir.set_ylabel("y  [dimensionless CR3BP length]", color=_TEXT)
    ax_dir.legend(fontsize=8, loc="lower left")

    # --- [1,1] Rolling acceptance rate per arc --------------------------------
    # Smoothed over a sliding window so you can see how gate acceptance varies
    # along the arc — e.g. degrading near apoapsis or dropout zones.
    ax_roll = fig4.add_subplot(gs4[1, 1])
    _ax_style(ax_roll)
    valid_step = np.isfinite(nis).astype(float)
    win        = max(5, len(valid_step) // 10)
    kernel     = np.ones(win) / win
    rolling_rate = np.convolve(valid_step, kernel, mode="same")
    ax_roll.fill_between(t_ekf, 0, valid_step,
                         color=_GREEN, alpha=0.12, step="mid", label="per-step accept")
    ax_roll.plot(t_ekf, rolling_rate, color=_CYAN, lw=2.0,
                 label=f"rolling mean  (win = {win} steps)")
    ax_roll.axhline(float(np.mean(valid_step)), color=_AMBER, lw=1.2, ls="--",
                    label=f"arc mean = {float(np.mean(valid_step)):.2f}")
    ax_roll.axvline(tc, color=_RED, lw=0.9, ls="--", alpha=0.6)
    ax_roll.set_ylim(-0.05, 1.12)
    ax_roll.set_title(
        f"Rolling Acceptance Rate per Arc  (win = {win} steps ≈ 10%)",
        color=_TEXT,
    )
    ax_roll.set_xlabel("t  [dimensionless CR3BP time]", color=_TEXT)
    ax_roll.set_ylabel("acceptance fraction", color=_TEXT)
    ax_roll.legend(fontsize=9)

    fig4.suptitle(
        f"Observability Analysis  [{out['camera_mode']}]",
        color=_TEXT, fontsize=13, y=0.98,
    )
    fig4.patch.set_facecolor(_BG)
    fig4.savefig(plots_dir / "06_observability.png", dpi=200, bbox_inches="tight")
    plt.close(fig4)

    print("Wrote:")
    print(f"  {plots_dir / '06_midcourse_report.png'}")
    print(f"  {plots_dir / '06_dv_compare.png'}")
    print(f"  {plots_dir / '06_diagnostics.png'}")
    print(f"  {plots_dir / '06_observability.png'}")

    # ── SPICE / high-fidelity comparison ─────────────────────────────────────
    kernels_present = all(Path(k).exists() for k in _DEFAULT_KERNELS)
    if not kernels_present:
        missing = [str(k) for k in _DEFAULT_KERNELS if not Path(k).exists()]
        print(f"\n[SPICE] skipped — kernel files not found:\n  " + "\n  ".join(missing))
        return

    print("\nRunning 06 SPICE/JPL high-fidelity midcourse EKF correction ...")
    out_s = run_case_spice(
        mu, t0, tf, tc, dt_meas, sigma_px, dropout_prob, seed,
        dx0, est_err, camera_mode="estimate_tracking",
    )
    lu = out_s["lunit_km"]
    tu = out_s["tunit_s"]
    vu = lu / tu   # km/s per ND velocity unit

    print(f"  lunit = {lu:.1f} km   tunit = {tu/86400:.4f} days   vunit = {vu*1000:.3f} m/s")
    print()
    print(f"  {'Metric':<26}  {'CR3BP (ND)':<18}  {'SPICE (phys)'}")
    print(f"  {'-'*26}  {'-'*18}  {'-'*28}")
    print(f"  {'|dv| perfect':<26}  {out['dv_perfect_mag']:.4e} ND       "
          f"{out_s['dv_perfect_mag']*1000:.4f} m/s")
    print(f"  {'|dv| EKF':<26}  {out['dv_ekf_mag']:.4e} ND       "
          f"{out_s['dv_ekf_mag']*1000:.4f} m/s")
    print(f"  {'miss uncorrected':<26}  {out['miss_uncorrected']:.4e} ND       "
          f"{out_s['miss_uncorrected']:.3f} km")
    print(f"  {'miss perfect Δv':<26}  {out['miss_perfect']:.4e} ND       "
          f"{out_s['miss_perfect']:.3f} km")
    print(f"  {'miss EKF Δv':<26}  {out['miss_ekf']:.4e} ND       "
          f"{out_s['miss_ekf']:.3f} km")
    print(f"  {'pos err at tc':<26}  {out['pos_err_tc']:.4e} ND       "
          f"{out_s['pos_err_tc']:.3f} km")
    print(f"  {'NIS mean (expect≈2)':<26}  {out['nis_mean']:.3f}              "
          f"{out_s['nis_mean']:.3f}")
    print(f"  {'NEES mean (expect≈6)':<26}  {out['nees_mean']:.3f}              "
          f"{out_s['nees_mean']:.3f}")
    print(f"  {'valid rate':<26}  {out['valid_rate']:.3f}              "
          f"{out_s['valid_rate']:.3f}")


if __name__ == "__main__":
    main()
