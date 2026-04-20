from __future__ import annotations

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
from nav.ekf import ekf_propagate_cr3bp_stm, ekf_propagate_stm
from nav.measurements.bearing import bearing_update_tangent
from nav.measurements.pixel_bearing import pixel_detection_to_bearing
from cv.camera import Intrinsics
from cv.pointing import camera_dcm_from_boresight
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
    return_debug: bool = True,
    accumulate_gramian: bool = True,
) -> Dict[str, Any]:
    camera_mode = _resolve_camera_mode(camera_mode)

    rng    = np.random.default_rng(int(seed))
    model  = CR3BP(mu=float(mu))

    L1x    = model.lagrange_points()["L1"][0]
    x0_nom = np.array([L1x - 1e-3, 0.0, 0.0, 0.0, 0.05, 0.0], dtype=float)
    x0_true = x0_nom + np.asarray(dx0, dtype=float).reshape(6)
    r_body  = np.array([1.0 - float(mu), 0.0, 0.0], dtype=float)

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
    q_acc = float(q_acc)

    x_hat_hist:   list[np.ndarray] = []
    nis_list:     list[float]      = []
    nees_list:    list[float]      = []
    innov_2d_list: list[np.ndarray] = []
    pos_err_list:  list[float]     = []
    P_diag_list:  list[np.ndarray] = []
    valid_arr     = np.zeros(k_tc + 1, dtype=bool)

    # Observability Gramian: W = Σ_k Φ(t_k,t_0)ᵀ Hₖᵀ Hₖ Φ(t_k,t_0)
    # Eigenvectors of W reveal which state-space directions are (poorly)
    # observable from bearing-only data. Gated for Monte Carlo efficiency.
    Phi_cum          = np.eye(6, dtype=float) if accumulate_gramian else None
    W_obs            = np.zeros((6, 6), dtype=float) if accumulate_gramian else None
    gramian_eig_hist: list[np.ndarray] = []

    for k in range(1, k_tc + 1):
        x_hat, P, Phi_step = ekf_propagate_cr3bp_stm(
            mu=float(mu), x=x_hat, P=P,
            t0=float(t_meas[k - 1]), t1=float(t_meas[k]), q_acc=q_acc,
        )
        if accumulate_gramian:
            Phi_cum = Phi_step @ Phi_cum

        r_sc_true = xs_true[k, :3]
        if camera_mode == "fixed":
            R_cam = R_fixed
        elif camera_mode == "truth_tracking":
            R_cam = camera_dcm_from_boresight(r_body - r_sc_true,
                                              camera_forward_axis="+z")
        else:
            R_cam = camera_dcm_from_boresight(r_body - x_hat[:3],
                                              camera_forward_axis="+z")

        meas = simulate_pixel_measurement(
            r_sc=r_sc_true, r_body=r_body, intrinsics=intr,
            R_cam_from_frame=R_cam, sigma_px=float(sigma_px),
            rng=rng, t=float(t_meas[k]),
            dropout_p=float(dropout_prob), out_of_frame="drop", behind="drop",
        )

        if meas.valid and np.isfinite(meas.u_px):
            u_g, sig_k = pixel_detection_to_bearing(
                meas.u_px, meas.v_px, float(sigma_px), intr, R_cam.T
            )
            if np.all(np.isfinite(u_g)):
                upd = bearing_update_tangent(
                    x_hat, P, u_g, r_body, float(sig_k)
                )
                if upd.accepted:
                    x_hat, P = upd.x_upd, upd.P_upd
                    if accumulate_gramian:
                        W_obs += Phi_cum.T @ upd.H.T @ upd.H @ Phi_cum
                nis_list.append(float(upd.nis))
                innov_2d_list.append(
                    upd.final_innovation.copy()
                    if upd.final_innovation is not None
                    else upd.innovation.copy()
                )
                valid_arr[k] = True
            else:
                nis_list.append(float("nan"))
                innov_2d_list.append(np.full(2, np.nan))
        else:
            nis_list.append(float("nan"))
            innov_2d_list.append(np.full(2, np.nan))

        # NEES: (x̂ − x_true)ᵀ P⁻¹ (x̂ − x_true), chi²(6) distributed for a
        # consistent filter.
        err6 = x_hat - xs_true[k]
        try:
            nees_val = float(err6 @ np.linalg.solve(P, err6))
        except np.linalg.LinAlgError:
            nees_val = float("nan")
        nees_list.append(nees_val)

        x_hat_hist.append(x_hat.copy())
        pos_err_list.append(_norm(x_hat[:3] - xs_true[k, :3]))
        P_diag_list.append(np.diag(P).copy())
        if accumulate_gramian:
            gramian_eig_hist.append(np.linalg.eigvalsh(W_obs).copy())

    nis_arr      = np.asarray(nis_list)
    nees_arr     = np.asarray(nees_list)

    x_true_tc    = xs_true[k_tc]
    x_hat_tc     = x_hat.copy()
    P_tc         = P.copy()
    pos_err_tc   = _norm(x_hat_tc[:3] - x_true_tc[:3])
    tracePpos_tc = float(np.trace(P_tc[:3, :3]))
    valid_rate   = float(np.mean(valid_arr[: k_tc + 1]))
    nis_finite   = nis_arr[np.isfinite(nis_arr)]
    nis_mean     = float(np.mean(nis_finite)) if nis_finite.size else float("nan")
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

    out: Dict[str, Any] = {
        "tc":           tc_eff,
        "sigma_px":     float(sigma_px),
        "dropout_prob": float(dropout_prob),
        "camera_mode":  camera_mode,
        "q_acc":        q_acc,
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
        "valid_rate":       valid_rate,
        "nis_mean":         nis_mean,
        "nees_mean":        nees_mean,
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

    out["debug"] = {
            "t_meas":        t_meas,
            "k_tc":          k_tc,
            "xs_nom":        res_nom.x,
            "xs_true":       xs_true,
            "x_hat_hist":    x_hat_arr,
            "pos_err_hist":  pos_err_arr,
            "P_diag_hist":   P_diag_arr,
            "nis_hist":      nis_arr,
            "nees_hist":     nees_arr,
            "innov_2d_hist":    innov_2d_arr,
            "W_obs":            W_obs,
            "gramian_eig_hist": gramian_eig_arr,
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
    return_debug: bool = True,
    accumulate_gramian: bool = True,
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
        t_meas_s = np.arange(t0_s, tf_s + 1e-6, dtm_s)
        if t_meas_s[-1] < tf_s - dtm_s * 1e-3:
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
        # Process-noise density in km²/s³ (proportionally converted from ND q_acc_nd)
        q_acc = float(q_acc_nd) * lsq / tunit_s ** 3

        # ── EKF loop ─────────────────────────────────────────────────────────
        x_hat_hist:    list[np.ndarray] = []
        nis_list:      list[float]      = []
        nees_list:     list[float]      = []
        innov_2d_list: list[np.ndarray] = []
        pos_err_list:  list[float]      = []
        P_diag_list:   list[np.ndarray] = []
        valid_arr      = np.zeros(k_tc + 1, dtype=bool)
        Phi_cum        = np.eye(6, dtype=float) if accumulate_gramian else None
        W_obs          = np.zeros((6, 6), dtype=float) if accumulate_gramian else None
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

            # Moon J2000 position at this measurement step (km)
            t_k    = float(t_meas_s[k])
            r_body = ephemeris.position_km("MOON", t_k)
            r_sc_true = xs_true[k, :3]

            if camera_mode == "fixed":
                R_cam = R_fixed
            elif camera_mode == "truth_tracking":
                R_cam = camera_dcm_from_boresight(r_body - r_sc_true,
                                                  camera_forward_axis="+z")
            else:
                R_cam = camera_dcm_from_boresight(r_body - x_hat[:3],
                                                  camera_forward_axis="+z")

            meas = simulate_pixel_measurement(
                r_sc=r_sc_true, r_body=r_body, intrinsics=intr,
                R_cam_from_frame=R_cam, sigma_px=float(sigma_px),
                rng=rng, t=t_k,
                dropout_p=float(dropout_prob), out_of_frame="drop", behind="drop",
            )

            if meas.valid and np.isfinite(meas.u_px):
                u_g, sig_k = pixel_detection_to_bearing(
                    meas.u_px, meas.v_px, float(sigma_px), intr, R_cam.T
                )
                if np.all(np.isfinite(u_g)):
                    upd = bearing_update_tangent(x_hat, P, u_g, r_body, float(sig_k))
                    if upd.accepted:
                        x_hat, P = upd.x_upd, upd.P_upd
                        if accumulate_gramian:
                            W_obs += Phi_cum.T @ upd.H.T @ upd.H @ Phi_cum
                    nis_list.append(float(upd.nis))
                    innov_2d_list.append(
                        upd.final_innovation.copy()
                        if upd.final_innovation is not None
                        else upd.innovation.copy()
                    )
                    valid_arr[k] = True
                else:
                    nis_list.append(float("nan"))
                    innov_2d_list.append(np.full(2, np.nan))
            else:
                nis_list.append(float("nan"))
                innov_2d_list.append(np.full(2, np.nan))

            err6 = x_hat - xs_true[k]
            try:
                nees_val = float(err6 @ np.linalg.solve(P, err6))
            except np.linalg.LinAlgError:
                nees_val = float("nan")
            nees_list.append(nees_val)

            x_hat_hist.append(x_hat.copy())
            pos_err_list.append(_norm(x_hat[:3] - xs_true[k, :3]))
            P_diag_list.append(np.diag(P).copy())
            if accumulate_gramian:
                gramian_eig_hist.append(np.linalg.eigvalsh(W_obs).copy())

        nis_arr         = np.asarray(nis_list)
        nees_arr        = np.asarray(nees_list)

        x_true_tc    = xs_true[k_tc]
        x_hat_tc     = x_hat.copy()
        P_tc         = P.copy()
        pos_err_tc   = _norm(x_hat_tc[:3] - x_true_tc[:3])
        tracePpos_tc = float(np.trace(P_tc[:3, :3]))
        valid_rate   = float(np.mean(valid_arr[: k_tc + 1]))
        nis_finite   = nis_arr[np.isfinite(nis_arr)]
        nis_mean     = float(np.mean(nis_finite)) if nis_finite.size else float("nan")
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

        out: Dict[str, Any] = {
            "tc":             tc_eff,
            "sigma_px":       float(sigma_px),
            "dropout_prob":   float(dropout_prob),
            "camera_mode":    camera_mode,
            "q_acc_nd":       float(q_acc_nd),
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
            "valid_rate":       valid_rate,
            "nis_mean":         nis_mean,
            "nees_mean":        nees_mean,
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

        out["debug"] = {
                "t_meas":           t_meas_s,
                "k_tc":             k_tc,
                "xs_nom":           res_nom.x,
                "xs_true":          xs_true,
                "x_hat_hist":       x_hat_arr,
                "pos_err_hist":     pos_err_arr,
                "P_diag_hist":      P_diag_arr,
                "nis_hist":         nis_arr,
                "nees_hist":        nees_arr,
                "innov_2d_hist":    innov_2d_arr,
                "W_obs":            W_obs,
                "gramian_eig_hist": gramian_eig_arr,
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
    plots_dir = Path("results/plots")
    _ensure_dir(plots_dir)

    mu, t0, tf, tc = 0.0121505856, 0.0, 6.0, 2.0
    dt_meas, sigma_px, dropout_prob, seed = 0.02, 1.5, 0.0, 7
    dx0     = np.array([1e-4, -1e-4, 0.0, 0.0, 0.0, 0.0], dtype=float)
    est_err = np.array([1e-4,  1e-4, 0.0, 0.0, 0.0, 0.0], dtype=float)

    print("Running 06 midcourse EKF correction ...")
    out = run_case(mu, t0, tf, tc, dt_meas, sigma_px, dropout_prob, seed,
                   dx0, est_err, camera_mode="estimate_tracking")
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
