from __future__ import annotations

import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from _common import ensure_src_on_path

ensure_src_on_path()

import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import chi2
from visualization.style import plt

from dynamics.cr3bp import CR3BP
from dynamics.integrators import propagate
from nav.ekf import ekf_propagate_cr3bp_stm
from nav.measurements.bearing import bearing_update_tangent
from nav.measurements.pixel_bearing import pixel_detection_to_bearing
from cv.sim_measurements import PixelMeasurement, simulate_pixel_measurement
from cv.camera import Intrinsics
from cv.pointing import (
    camera_dcm_from_boresight,
    estimate_based_camera_attitude,
    off_boresight_angle,
)

Array = np.ndarray


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
        "lines.linewidth":   1.8,
        "legend.facecolor":  _PANEL,
        "legend.edgecolor":  _BORDER,
        "legend.labelcolor": _TEXT,
        "savefig.facecolor": _BG,
        "savefig.edgecolor": _BG,
        "font.size":         11,
    })



MU = 0.0121505856

CAM = dict(
    width=1024, height=1024,
    fx=800.0, fy=800.0, cx=512.0, cy=512.0,
    sigma_px=1.0, dropout_p=0.0,
    out_of_frame="drop", behind="drop",
)

SIM = dict(
    dt=0.02, t0=0.0, tf=6.0, seed=7,
    q_acc=1e-14,
)

_X0_NOM = np.array([0.8359, 0.0, 0.0, 0.0, 0.05, 0.0], dtype=float)

INIT = dict(
    x0_nom=_X0_NOM.copy(),
    dx0=np.array([1e-4, -1e-4, 0.0, 0.0, 0.0, 0.0], dtype=float),
    est_err=np.array([1e-4, 1e-4, 0.0, 0.0, 0.0, 0.0], dtype=float),
    P0_diag=np.array([1e-6] * 6, dtype=float),
)

_r_moon_nom   = np.array([1.0 - MU, 0.0, 0.0], dtype=float)
_boresight_nom = _r_moon_nom - _X0_NOM[:3]
_R_CAM_FIXED  = camera_dcm_from_boresight(_boresight_nom, camera_forward_axis="+z")

POINTING = dict(
    active_pointing=True,
    up_hint_frame=np.array([0.0, 0.0, 1.0], dtype=float),
    camera_forward_axis="+z",
    fixed_R_cam_from_frame=_R_CAM_FIXED,
)

SANITY = dict(
    enabled=True,
    sigma_px=1e-6,
    dropout_p=0.0,
    dx0=np.array([1e-7, -1e-7, 0.0, 0.0, 0.0, 0.0], dtype=float),
    est_err=np.array([1e-7, 1e-7, 0.0, 0.0, 0.0, 0.0], dtype=float),
    q_acc=0.0,
)

OUTPUT = dict(
    results_dir=Path("results/active_tracking"),
    plots_dir=Path("results/active_tracking"),
    videos_dir=Path("results/videos"),
    stem_base="07_active_tracking",
)



def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _norm(x: Array) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=float)))


def _unit(x: Array, eps: float = 1e-12) -> Array:
    x = np.asarray(x, dtype=float).reshape(3)
    n = _norm(x)
    if not np.isfinite(n) or n <= eps:
        return np.full(3, np.nan, dtype=float)
    return x / n


def _angle_between(a: Array, b: Array) -> float:
    ua, ub = _unit(a), _unit(b)
    if not (np.all(np.isfinite(ua)) and np.all(np.isfinite(ub))):
        return float("nan")
    return float(np.arccos(np.clip(float(np.dot(ua, ub)), -1.0, 1.0)))


def _build_intrinsics(cam_cfg: dict[str, Any] | None = None) -> Intrinsics:
    c = CAM if cam_cfg is None else cam_cfg
    return Intrinsics(
        fx=float(c["fx"]), fy=float(c["fy"]),
        cx=float(c["cx"]), cy=float(c["cy"]),
        width=int(c["width"]), height=int(c["height"]),
    )


def _make_time_grid() -> Array:
    return np.arange(float(SIM["t0"]), float(SIM["tf"]) + 1e-12,
                     float(SIM["dt"]), dtype=float)


def _moon_position() -> Array:
    return np.array([1.0 - MU, 0.0, 0.0], dtype=float)


def _boresight_unit_in_camera(axis: str) -> Array:
    mapping = {
        "+z": [0, 0, 1], "-z": [0, 0, -1],
        "+x": [1, 0, 0], "-x": [-1, 0, 0],
        "+y": [0, 1, 0], "-y": [0, -1, 0],
    }
    if axis not in mapping:
        raise ValueError(f"Unsupported camera_forward_axis={axis!r}")
    return np.array(mapping[axis], dtype=float)


def _compute_pointing(
    xhat_sc: Array,
    r_body_true: Array,
    pointing_cfg: dict[str, Any],
) -> tuple[Array, Array]:
    if bool(pointing_cfg["active_pointing"]):
        los_cmd, R_cam = estimate_based_camera_attitude(
            xhat_sc=np.asarray(xhat_sc, dtype=float),
            target_pos_I=np.asarray(r_body_true, dtype=float),
            up_hint_I=np.asarray(pointing_cfg["up_hint_frame"], dtype=float),
            camera_forward_axis=str(pointing_cfg["camera_forward_axis"]),
        )
        return np.asarray(los_cmd, dtype=float), np.asarray(R_cam, dtype=float)

    R_cam = np.asarray(pointing_cfg["fixed_R_cam_from_frame"], dtype=float)
    los_cmd = R_cam.T @ _boresight_unit_in_camera(
        str(pointing_cfg["camera_forward_axis"])
    )
    return los_cmd, R_cam


def _invalid_pixel(t: float, sigma_px: float, reason: str) -> PixelMeasurement:
    return PixelMeasurement(
        valid=False, u_px=float("nan"), v_px=float("nan"),
        sigma_px=float(sigma_px), t=float(t), meta={"reason": reason},
    )



def _initialize_logs() -> dict[str, list]:
    return {k: [] for k in [
        "t_hist", "x_true_hist", "xhat_hist", "Pdiag_hist",
        "r_sc_true_hist", "v_sc_true_hist", "r_body_true_hist",
        "visible_hist", "update_used_hist",
        "u_px_hist", "v_px_hist", "u_px_ideal_hist", "v_px_ideal_hist",
        "los_true_hist", "los_est_hist", "los_cmd_hist",
        "angle_true_vs_est_hist", "angle_true_vs_cmd_hist",
        "off_boresight_hist", "nis_hist", "sigma_theta_hist",
        "reason_hist", "distance_from_principal_point",
    ]}


def _append_epoch_log(
    logs: dict[str, list],
    *,
    K: Intrinsics,
    t: float,
    x_true: Array,
    xhat: Array,
    P: Array,
    r_sc_true: Array,
    v_sc_true: Array,
    r_body_true: Array,
    visible: bool,
    update_used: bool,
    pix: PixelMeasurement,
    los_true: Array,
    los_est: Array,
    los_cmd: Array,
    sigma_theta: float,
    nis: float,
) -> None:
    angle_true_vs_est = _angle_between(los_true, los_est)
    angle_true_vs_cmd = _angle_between(los_true, los_cmd)
    meta = {}
    if hasattr(pix, "meta") and isinstance(pix.meta, dict):
        meta = pix.meta
    offb = float(off_boresight_angle(los_true, los_cmd)) if visible else float("nan")

    u_px_val = float(pix.u_px) if visible and np.isfinite(pix.u_px) else float("nan")
    v_px_val = float(pix.v_px) if visible and np.isfinite(pix.v_px) else float("nan")
    if visible and np.isfinite(u_px_val) and np.isfinite(v_px_val):
        r_img = float(np.hypot(u_px_val - float(K.cx), v_px_val - float(K.cy)))
    else:
        r_img = float("nan")

    logs["t_hist"].append(float(t))
    logs["x_true_hist"].append(np.asarray(x_true, dtype=float).copy())
    logs["xhat_hist"].append(np.asarray(xhat, dtype=float).copy())
    logs["Pdiag_hist"].append(np.diag(np.asarray(P, dtype=float)).copy())
    logs["r_sc_true_hist"].append(np.asarray(r_sc_true, dtype=float).copy())
    logs["v_sc_true_hist"].append(np.asarray(v_sc_true, dtype=float).copy())
    logs["r_body_true_hist"].append(np.asarray(r_body_true, dtype=float).copy())
    logs["visible_hist"].append(bool(visible))
    logs["update_used_hist"].append(bool(update_used))
    logs["u_px_hist"].append(u_px_val)
    logs["v_px_hist"].append(v_px_val)
    logs["u_px_ideal_hist"].append(float(meta.get("u_ideal", np.nan)))
    logs["v_px_ideal_hist"].append(float(meta.get("v_ideal", np.nan)))
    logs["los_true_hist"].append(np.asarray(los_true, dtype=float).copy())
    logs["los_est_hist"].append(np.asarray(los_est, dtype=float).copy())
    logs["los_cmd_hist"].append(np.asarray(los_cmd, dtype=float).copy())
    logs["angle_true_vs_est_hist"].append(float(angle_true_vs_est))
    logs["angle_true_vs_cmd_hist"].append(float(angle_true_vs_cmd))
    logs["off_boresight_hist"].append(float(offb))
    logs["nis_hist"].append(float(nis))
    logs["sigma_theta_hist"].append(float(sigma_theta))
    logs["reason_hist"].append(meta.get("reason", "valid" if visible else "invalid"))
    logs["distance_from_principal_point"].append(r_img)


def _finalize_logs(logs: dict[str, list]) -> dict[str, Any]:
    return {
        "t_hist":              np.asarray(logs["t_hist"], dtype=float),
        "x_true_hist":         np.asarray(logs["x_true_hist"], dtype=float),
        "xhat_hist":           np.asarray(logs["xhat_hist"], dtype=float),
        "Pdiag_hist":          np.asarray(logs["Pdiag_hist"], dtype=float),
        "r_sc_true_hist":      np.asarray(logs["r_sc_true_hist"], dtype=float),
        "v_sc_true_hist":      np.asarray(logs["v_sc_true_hist"], dtype=float),
        "r_body_true_hist":    np.asarray(logs["r_body_true_hist"], dtype=float),
        "visible_hist":        np.asarray(logs["visible_hist"], dtype=bool),
        "update_used_hist":    np.asarray(logs["update_used_hist"], dtype=bool),
        "u_px_hist":           np.asarray(logs["u_px_hist"], dtype=float),
        "v_px_hist":           np.asarray(logs["v_px_hist"], dtype=float),
        "u_px_ideal_hist":     np.asarray(logs["u_px_ideal_hist"], dtype=float),
        "v_px_ideal_hist":     np.asarray(logs["v_px_ideal_hist"], dtype=float),
        "los_true_hist":       np.asarray(logs["los_true_hist"], dtype=float),
        "los_est_hist":        np.asarray(logs["los_est_hist"], dtype=float),
        "los_cmd_hist":        np.asarray(logs["los_cmd_hist"], dtype=float),
        "angle_true_vs_est_hist": np.asarray(logs["angle_true_vs_est_hist"], dtype=float),
        "angle_true_vs_cmd_hist": np.asarray(logs["angle_true_vs_cmd_hist"], dtype=float),
        "off_boresight_hist":  np.asarray(logs["off_boresight_hist"], dtype=float),
        "nis_hist":            np.asarray(logs["nis_hist"], dtype=float),
        "sigma_theta_hist":    np.asarray(logs["sigma_theta_hist"], dtype=float),
        "reason_hist":         np.asarray(logs["reason_hist"], dtype=object),
        "distance_from_principal_point": np.asarray(
            logs["distance_from_principal_point"], dtype=float),
    }



def validate_configuration() -> None:
    K = _build_intrinsics()
    if K.width is None or K.height is None:
        raise ValueError("Camera intrinsics must include width/height.")
    if INIT["x0_nom"].shape != (6,):
        raise ValueError(f"INIT['x0_nom'] must be shape (6,), got {INIT['x0_nom'].shape}")
    if float(SIM["dt"]) <= 0.0:
        raise ValueError("SIM['dt'] must be > 0")


def _scenario_name(active_pointing: bool, sanity: bool) -> str:
    base = "active" if active_pointing else "fixed"
    return f"{base}_sanity" if sanity else base


def _summarize_case(results: dict[str, Any]) -> dict[str, float]:
    x_true = np.asarray(results["x_true_hist"], dtype=float)
    xhat   = np.asarray(results["xhat_hist"],   dtype=float)
    vis    = np.asarray(results["visible_hist"],     dtype=bool)
    upd    = np.asarray(results["update_used_hist"], dtype=bool)
    pos_err = np.linalg.norm(xhat[:, :3] - x_true[:, :3], axis=1)
    vel_err = np.linalg.norm(xhat[:, 3:6] - x_true[:, 3:6], axis=1)
    return dict(
        visibility_fraction=float(np.mean(vis)),
        update_fraction=float(np.mean(upd)),
        rms_position_error=float(np.sqrt(np.mean(pos_err**2))),
        rms_velocity_error=float(np.sqrt(np.mean(vel_err**2))),
    )


def run_case(*, active_pointing: bool, sanity: bool = False) -> dict[str, Any]:
    validate_configuration()

    cam_cfg      = deepcopy(CAM)
    pointing_cfg = deepcopy(POINTING)
    init_cfg     = deepcopy(INIT)
    pointing_cfg["active_pointing"] = bool(active_pointing)

    if sanity:
        cam_cfg["sigma_px"]  = float(SANITY["sigma_px"])
        cam_cfg["dropout_p"] = float(SANITY["dropout_p"])
        init_cfg["dx0"]      = np.asarray(SANITY["dx0"],     dtype=float).copy()
        init_cfg["est_err"]  = np.asarray(SANITY["est_err"], dtype=float).copy()
        q_acc = float(SANITY["q_acc"])
        seed  = int(SIM["seed"]) + 101
    else:
        q_acc = float(SIM["q_acc"])
        seed  = int(SIM["seed"])

    rng   = np.random.default_rng(seed)
    model = CR3BP(mu=MU)
    K     = _build_intrinsics(cam_cfg)
    t_meas = _make_time_grid()

    x0_nom   = np.asarray(init_cfg["x0_nom"],  dtype=float)
    x0_true  = x0_nom + np.asarray(init_cfg["dx0"],    dtype=float)
    xhat     = x0_nom + np.asarray(init_cfg["est_err"],dtype=float)
    P        = np.diag(np.asarray(init_cfg["P0_diag"], dtype=float))

    res_truth = propagate(
        model.eom, (float(t_meas[0]), float(t_meas[-1])), x0_true,
        t_eval=t_meas, rtol=1e-11, atol=1e-13, method="DOP853",
    )
    if not res_truth.success:
        raise RuntimeError(f"Truth propagation failed: {res_truth.message}")
    xs_true        = np.asarray(res_truth.x, dtype=float)
    r_body_const   = _moon_position()
    logs           = _initialize_logs()

    x_true0   = np.asarray(xs_true[0], dtype=float)
    los_true0 = _unit(r_body_const - x_true0[:3])
    los_est0  = _unit(r_body_const - xhat[:3])
    los_cmd0, _ = _compute_pointing(xhat, r_body_const, pointing_cfg)

    _append_epoch_log(
        logs, K=K, t=float(t_meas[0]),
        x_true=x_true0, xhat=xhat, P=P,
        r_sc_true=x_true0[:3], v_sc_true=x_true0[3:6],
        r_body_true=r_body_const,
        visible=False, update_used=False,
        pix=_invalid_pixel(float(t_meas[0]), float(cam_cfg["sigma_px"]),
                           "initial_epoch_no_measurement"),
        los_true=los_true0, los_est=los_est0, los_cmd=los_cmd0,
        sigma_theta=float("nan"), nis=float("nan"),
    )

    for k in range(1, len(t_meas)):
        t_prev = float(t_meas[k - 1])
        t_curr = float(t_meas[k])

        x_true    = np.asarray(xs_true[k], dtype=float)
        r_sc_true = x_true[:3].copy()
        v_sc_true = x_true[3:6].copy()
        r_body    = r_body_const.copy()

        xhat_minus, P_minus, _ = ekf_propagate_cr3bp_stm(
            mu=MU, x=xhat, P=P, t0=t_prev, t1=t_curr, q_acc=q_acc,
        )

        los_est       = _unit(r_body - xhat_minus[:3])
        los_cmd, R_cam = _compute_pointing(xhat_minus, r_body, pointing_cfg)
        R_frame_from_cam = R_cam.T

        pix = simulate_pixel_measurement(
            r_sc=r_sc_true, r_body=r_body,
            intrinsics=K, R_cam_from_frame=R_cam,
            sigma_px=float(cam_cfg["sigma_px"]),
            rng=rng, t=t_curr,
            dropout_p=float(cam_cfg["dropout_p"]),
            noise_mode="gaussian",
            out_of_frame=str(cam_cfg["out_of_frame"]),
            behind=str(cam_cfg["behind"]),
        )

        visible      = bool(pix.valid)
        update_used  = False
        sigma_theta  = float("nan")
        nis          = float("nan")

        if visible and np.isfinite(pix.u_px) and np.isfinite(pix.v_px):
            u_meas, sigma_theta = pixel_detection_to_bearing(
                u_px=float(pix.u_px),
                v_px=float(pix.v_px),
                sigma_px=float(pix.sigma_px),
                intrinsics=K,
                R_frame_from_cam=R_frame_from_cam,
                sigma_approx="fx_only",
            )
            sigma_theta = float(sigma_theta)

            upd = bearing_update_tangent(
                x=xhat_minus, P=P_minus,
                u_meas=u_meas, r_body=r_body,
                sigma_theta=sigma_theta,
            )
            nis = float(upd.nis)
            if upd.accepted:
                xhat = np.asarray(upd.x_upd, dtype=float)
                P    = np.asarray(upd.P_upd, dtype=float)
                update_used = True
            else:
                xhat = np.asarray(xhat_minus, dtype=float)
                P    = np.asarray(P_minus, dtype=float)
        else:
            xhat = np.asarray(xhat_minus, dtype=float)
            P    = np.asarray(P_minus, dtype=float)

        los_true = _unit(r_body - r_sc_true)
        _append_epoch_log(
            logs, K=K, t=t_curr,
            x_true=x_true, xhat=xhat, P=P,
            r_sc_true=r_sc_true, v_sc_true=v_sc_true, r_body_true=r_body,
            visible=visible, update_used=update_used, pix=pix,
            los_true=los_true, los_est=los_est, los_cmd=los_cmd,
            sigma_theta=sigma_theta, nis=nis,
        )

    results = _finalize_logs(logs)
    results["mu"]            = float(MU)
    results["active_pointing"] = bool(active_pointing)
    results["sanity_case"]   = bool(sanity)
    results["case_name"]     = _scenario_name(active_pointing=active_pointing, sanity=sanity)
    summary = _summarize_case(results)
    for k, v in summary.items():
        results[k] = float(v)
    return results



def _ax_style(ax: plt.Axes) -> None:
    ax.set_facecolor(_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(_BORDER)
    ax.grid(True)


def save_results(results: dict[str, Any], stem: str) -> Path:
    out_dir = OUTPUT["results_dir"]
    _ensure_dir(out_dir)
    out_path = out_dir / f"{stem}.npz"
    np.savez_compressed(out_path, **{
        k: v for k, v in results.items()
        if isinstance(v, (np.ndarray, float, int, bool, str))
    })
    return out_path


def make_static_plots(results: dict[str, Any], K: Intrinsics, stem: str) -> list[Path]:
    _apply_dark_theme()
    out_dir = OUTPUT["plots_dir"]
    _ensure_dir(out_dir)
    paths: list[Path] = []

    case_name = str(results["case_name"])
    t         = np.asarray(results["t_hist"], dtype=float)
    x_true    = np.asarray(results["x_true_hist"], dtype=float)
    xhat      = np.asarray(results["xhat_hist"],   dtype=float)
    u         = np.asarray(results["u_px_hist"],   dtype=float)
    v         = np.asarray(results["v_px_hist"],   dtype=float)
    visible   = np.asarray(results["visible_hist"],     dtype=bool)
    upd_used  = np.asarray(results["update_used_hist"], dtype=bool)
    r_img     = np.asarray(results["distance_from_principal_point"], dtype=float)
    ang_est   = np.rad2deg(np.asarray(results["angle_true_vs_est_hist"], dtype=float))
    ang_cmd   = np.rad2deg(np.asarray(results["angle_true_vs_cmd_hist"], dtype=float))
    nis       = np.asarray(results["nis_hist"],       dtype=float)
    Pdiag     = np.asarray(results["Pdiag_hist"],     dtype=float)

    pos_err = np.linalg.norm(xhat[:, :3] - x_true[:, :3], axis=1)
    vel_err = np.linalg.norm(xhat[:, 3:6] - x_true[:, 3:6], axis=1)
    sig_pos = 3.0 * np.sqrt(np.abs(Pdiag[:, 0]))

    fig, ax = plt.subplots(figsize=(6, 6))
    _ax_style(ax)
    ax.set_facecolor("#050709")
    ax.set_xlim(0, K.width); ax.set_ylim(K.height, 0)
    ax.plot([0, K.width, K.width, 0, 0], [0, 0, K.height, K.height, 0],
            color=_BORDER, lw=0.8)
    ax.axvline(K.cx, color=_DIM, lw=0.7, ls="--")
    ax.axhline(K.cy, color=_DIM, lw=0.7, ls="--")
    mask = np.isfinite(u) & np.isfinite(v)
    sc = ax.scatter(u[mask], v[mask], c=t[mask], cmap="plasma",
                    s=10, vmin=t[0], vmax=t[-1], zorder=3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("t [dimensionless CR3BP time]", color=_TEXT)
    cb.ax.yaxis.set_tick_params(color=_TEXT)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=_TEXT)
    ax.set_xlabel("u [px]"); ax.set_ylabel("v [px]")
    ax.set_title(f"Pixel track — {case_name}")
    p = out_dir / f"{stem}_pixel_track.png"
    fig.tight_layout(); fig.savefig(p, dpi=200); plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(10, 3))
    _ax_style(ax)
    ax.step(t, visible.astype(int), where="post", color=_GREEN, lw=1.5, label="visible")
    ax.step(t, upd_used.astype(int) * 0.85, where="post",
            color=_CYAN, lw=1.5, label="EKF update", alpha=0.9)
    ax.set_xlabel("t [dimensionless CR3BP time]"); ax.set_ylabel("flag")
    ax.set_title(f"Visibility / Update Timeline — {case_name}")
    ax.set_yticks([0, 1]); ax.legend()
    p = out_dir / f"{stem}_visibility.png"
    fig.tight_layout(); fig.savefig(p, dpi=200); plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(10, 4))
    _ax_style(ax)
    ax.fill_between(t, 0, sig_pos, color=_VIOLET, alpha=0.15, label="3σ (x)")
    ax.semilogy(t, pos_err + 1e-12, color=_CYAN,  lw=1.8, label="‖pos err‖")
    ax.semilogy(t, vel_err + 1e-12, color=_AMBER, lw=1.5, ls="--",
                label="‖vel err‖", alpha=0.85)
    ax.set_xlabel("t [dimensionless CR3BP time]"); ax.set_ylabel("Error norm [dimensionless CR3BP units]")
    ax.set_title(f"State Estimation Errors — {case_name}")
    ax.legend()
    p = out_dir / f"{stem}_errors.png"
    fig.tight_layout(); fig.savefig(p, dpi=200); plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(10, 4))
    _ax_style(ax)
    ax.plot(t, ang_est, color=_CYAN,  lw=1.8, label="true vs estimated LOS")
    ax.plot(t, ang_cmd, color=_AMBER, lw=1.5, ls="--",
            label="true vs commanded LOS", alpha=0.85)
    ax.set_xlabel("t [dimensionless CR3BP time]"); ax.set_ylabel("Angle [deg]")
    ax.set_title(f"Pointing Diagnostics — {case_name}")
    ax.legend()
    p = out_dir / f"{stem}_pointing.png"
    fig.tight_layout(); fig.savefig(p, dpi=200); plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    _ax_style(ax)
    nis_lo = chi2.ppf(0.025, df=2); nis_hi = chi2.ppf(0.975, df=2)
    ax.fill_between(t, nis_lo, nis_hi, color=_GREEN, alpha=0.10,
                    label=f"95% χ²(2): [{nis_lo:.2f}, {nis_hi:.2f}]")
    nis_ok = np.isfinite(nis)
    in_b  = nis_ok & (nis >= nis_lo) & (nis <= nis_hi)
    out_b = nis_ok & ~in_b
    ax.scatter(t[in_b],  nis[in_b],  s=10, c=_GREEN, zorder=4)
    ax.scatter(t[out_b], nis[out_b], s=10, c=_RED,   zorder=4)
    ax.set_xlabel("t [dimensionless CR3BP time]"); ax.set_ylabel("NIS")
    ax.set_title(f"NIS — {case_name}  (mean={np.nanmean(nis):.2f})")
    ax.set_ylim(0, 20); ax.legend()
    p = out_dir / f"{stem}_nis.png"
    fig.tight_layout(); fig.savefig(p, dpi=200); plt.close(fig)
    paths.append(p)

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                             gridspec_kw={"hspace": 0.08})
    fig.patch.set_facecolor(_BG)
    for ax in axs: _ax_style(ax)

    axs[0].step(t, visible.astype(int), where="post", color=_GREEN, lw=1.5, label="visible")
    axs[0].step(t, upd_used.astype(int) * 0.85, where="post",
                color=_CYAN, lw=1.5, alpha=0.85, label="update used")
    axs[0].set_ylabel("flag"); axs[0].set_title(f"{case_name}"); axs[0].legend()

    axs[1].plot(t, r_img, color=_AMBER, lw=1.5)
    axs[1].set_ylabel("px from principal point")

    axs[2].semilogy(t, pos_err + 1e-12, color=_CYAN, lw=1.8)
    axs[2].set_xlabel("t [dimensionless CR3BP time]"); axs[2].set_ylabel("‖r̂ − r‖ [dimensionless CR3BP length]")

    p = out_dir / f"{stem}_summary.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    paths.append(p)

    return paths


def make_comparison_plots(
    active_results: dict[str, Any],
    fixed_results:  dict[str, Any],
    *,
    stem: str,
) -> list[Path]:
    _apply_dark_theme()
    out_dir = OUTPUT["plots_dir"]
    _ensure_dir(out_dir)
    paths: list[Path] = []

    ta = np.asarray(active_results["t_hist"], dtype=float)
    tf = np.asarray(fixed_results["t_hist"],  dtype=float)

    xa_true = np.asarray(active_results["x_true_hist"], dtype=float)
    xa_hat  = np.asarray(active_results["xhat_hist"],   dtype=float)
    xf_true = np.asarray(fixed_results["x_true_hist"],  dtype=float)
    xf_hat  = np.asarray(fixed_results["xhat_hist"],    dtype=float)

    pos_err_a = np.linalg.norm(xa_hat[:, :3] - xa_true[:, :3], axis=1)
    pos_err_f = np.linalg.norm(xf_hat[:, :3] - xf_true[:, :3], axis=1)
    vel_err_a = np.linalg.norm(xa_hat[:, 3:6] - xa_true[:, 3:6], axis=1)
    vel_err_f = np.linalg.norm(xf_hat[:, 3:6] - xf_true[:, 3:6], axis=1)

    vis_a = np.asarray(active_results["visible_hist"],     dtype=bool).astype(int)
    vis_f = np.asarray(fixed_results["visible_hist"],      dtype=bool).astype(int)
    upd_a = np.asarray(active_results["update_used_hist"], dtype=bool).astype(int)
    upd_f = np.asarray(fixed_results["update_used_hist"],  dtype=bool).astype(int)

    nis_a = np.asarray(active_results["nis_hist"], dtype=float)
    nis_f = np.asarray(fixed_results["nis_hist"],  dtype=float)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"hspace": 0.08})
    fig.patch.set_facecolor(_BG)
    for ax in axs: _ax_style(ax)

    axs[0].step(ta, vis_a, where="post", color=_CYAN,   lw=1.4, label="active visible")
    axs[0].step(ta, upd_a * 0.85, where="post", color=_CYAN, lw=1.0, ls="--",
                alpha=0.8, label="active update")
    axs[0].step(tf, vis_f, where="post", color=_AMBER,  lw=1.4, label="fixed visible")
    axs[0].step(tf, upd_f * 0.85, where="post", color=_AMBER, lw=1.0, ls="--",
                alpha=0.8, label="fixed update")
    axs[0].set_ylabel("flag"); axs[0].legend(ncol=2, fontsize=9)
    axs[0].set_title("Active vs Fixed Pointing — Filter Comparison", fontsize=13)

    axs[1].semilogy(ta, pos_err_a + 1e-12, color=_CYAN,  lw=1.8, label="active ‖pos err‖")
    axs[1].semilogy(tf, pos_err_f + 1e-12, color=_AMBER, lw=1.8, ls="--",
                    label="fixed ‖pos err‖")
    axs[1].semilogy(ta, vel_err_a + 1e-12, color=_CYAN,  lw=1.2, ls=":",
                    alpha=0.7, label="active ‖vel err‖")
    axs[1].semilogy(tf, vel_err_f + 1e-12, color=_AMBER, lw=1.2, ls=":",
                    alpha=0.7, label="fixed ‖vel err‖")
    axs[1].set_ylabel("Error norm [dimensionless CR3BP units]"); axs[1].legend(ncol=2, fontsize=9)

    nis_lo = chi2.ppf(0.025, df=2); nis_hi = chi2.ppf(0.975, df=2)
    axs[2].fill_between(ta, nis_lo, nis_hi, color=_GREEN, alpha=0.08)
    nis_ok_a = np.isfinite(nis_a); nis_ok_f = np.isfinite(nis_f)
    axs[2].scatter(ta[nis_ok_a], nis_a[nis_ok_a], s=8, c=_CYAN,  alpha=0.7,
                   label="active NIS", zorder=3)
    axs[2].scatter(tf[nis_ok_f], nis_f[nis_ok_f], s=8, c=_AMBER, alpha=0.7,
                   label="fixed NIS",  zorder=3)
    axs[2].set_ylabel("NIS"); axs[2].set_xlabel("t [dimensionless CR3BP time]")
    axs[2].set_ylim(0, 20); axs[2].legend(ncol=2, fontsize=9)

    p = out_dir / f"{stem}_comparison.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_PANEL); ax.axis("off")
    for sp in ax.spines.values(): sp.set_edgecolor(_BORDER)

    lines = [
        "Fixed vs Active Pointing — Summary",
        "",
        f"  Fixed  : vis={fixed_results['visibility_fraction']:.3f}  "
        f"upd={fixed_results['update_fraction']:.3f}  "
        f"RMS_pos={fixed_results['rms_position_error']:.3e}  "
        f"RMS_vel={fixed_results['rms_velocity_error']:.3e}",
        f"  Active : vis={active_results['visibility_fraction']:.3f}  "
        f"upd={active_results['update_fraction']:.3f}  "
        f"RMS_pos={active_results['rms_position_error']:.3e}  "
        f"RMS_vel={active_results['rms_velocity_error']:.3e}",
    ]
    ax.text(0.04, 0.88, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=11, color=_TEXT,
            transform=ax.transAxes)

    p = out_dir / f"{stem}_comparison_summary.png"
    fig.tight_layout(); fig.savefig(p, dpi=200); plt.close(fig)
    paths.append(p)

    return paths



def _run_ffmpeg_from_frames(frames_dir: Path, out_path: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not installed or not on PATH.")
    cmd = [
        ffmpeg, "-y",
        "-framerate", str(int(fps)),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def make_tracking_feed_video(
    results: dict[str, Any],
    K: Intrinsics,
    *,
    stem: str,
    fps: int = 20,
    keep_frames: bool = True,
) -> Path:
    out_dir    = OUTPUT["videos_dir"]
    _ensure_dir(out_dir)
    out_path   = out_dir / f"{stem}_feed.mp4"
    frames_dir = out_dir / f"{stem}_feed_frames"
    _ensure_dir(frames_dir)

    t        = np.asarray(results["t_hist"],          dtype=float)
    u        = np.asarray(results["u_px_hist"],       dtype=float)
    v        = np.asarray(results["v_px_hist"],       dtype=float)
    u0       = np.asarray(results["u_px_ideal_hist"], dtype=float)
    v0       = np.asarray(results["v_px_ideal_hist"], dtype=float)
    visible  = np.asarray(results["visible_hist"],     dtype=bool)
    upd_used = np.asarray(results["update_used_hist"], dtype=bool)
    offb_deg = np.rad2deg(np.asarray(results["angle_true_vs_cmd_hist"], dtype=float))
    nis      = np.asarray(results["nis_hist"],         dtype=float)
    reasons  = np.asarray(results["reason_hist"],      dtype=object)
    pos_err  = np.linalg.norm(
        np.asarray(results["xhat_hist"],   dtype=float)[:, :3]
        - np.asarray(results["x_true_hist"], dtype=float)[:, :3], axis=1
    )

    dpi    = 100
    width  = int(K.width)
    height = int(K.height)

    for i in range(len(t)):
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="#050709")
        ax  = fig.add_axes([0, 0, 1, 1])
        blank = np.zeros((height, width), dtype=np.float32)
        ax.imshow(blank, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
        ax.set_axis_off()

        ax.plot([1, width-1, width-1, 1, 1], [1, 1, height-1, height-1, 1],
                color=_BORDER, lw=0.8, alpha=0.7)
        ax.plot([K.cx - 12, K.cx + 12], [K.cy, K.cy],   color=_DIM, lw=0.9)
        ax.plot([K.cx, K.cx], [K.cy - 12, K.cy + 12],   color=_DIM, lw=0.9)

        if np.isfinite(u0[i]) and np.isfinite(v0[i]):
            ax.scatter([u0[i]], [v0[i]], marker="x", s=30, color=_DIM, zorder=3)

        if visible[i] and np.isfinite(u[i]) and np.isfinite(v[i]):
            dot_color = _GREEN if upd_used[i] else _AMBER
            ax.scatter([u[i]], [v[i]], s=40, color=dot_color, zorder=5)
            ax.plot([u[i]-8, u[i]+8], [v[i], v[i]], color=dot_color, lw=1.2, zorder=4)
            ax.plot([u[i], u[i]], [v[i]-8, v[i]+8], color=dot_color, lw=1.2, zorder=4)

        status = "UPDATE" if upd_used[i] else ("DETECT" if visible[i] else "NO SIGNAL")
        sc = _GREEN if upd_used[i] else (_AMBER if visible[i] else _RED)
        txt = (
            f"case = {results['case_name']}\n"
            f"t = {t[i]:.3f} dimensionless\n"
            f"u,v = ({u[i]:.1f}, {v[i]:.1f})\n"
            f"{status}\n"
            f"off-boresight = {offb_deg[i]:.2f}°\n"
            f"NIS = {nis[i]:.3f}\n"
            f"‖pos err‖ = {pos_err[i]:.3e} dimensionless\n"
            f"reason = {reasons[i]}"
        )
        ax.text(12, 18, txt, color=sc, fontsize=10, va="top", ha="left",
                family="monospace",
                bbox=dict(facecolor=(0,0,0,0.5), edgecolor=(_BORDER+"80"),
                          boxstyle="round", pad=5))

        fig.savefig(frames_dir / f"frame_{i:05d}.png", dpi=dpi,
                    facecolor=fig.get_facecolor())
        plt.close(fig)

    _run_ffmpeg_from_frames(frames_dir, out_path, fps=fps)
    if not keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
    return out_path



def _run_and_emit_case(*, active_pointing: bool, sanity: bool,
                        K: Intrinsics) -> tuple[dict[str, Any], dict[str, Any]]:
    results    = run_case(active_pointing=active_pointing, sanity=sanity)
    stem       = f"{OUTPUT['stem_base']}_{results['case_name']}"
    npz_path   = save_results(results, stem=stem)
    fig_paths  = make_static_plots(results, K, stem=stem)
    video_path = make_tracking_feed_video(results, K, stem=stem, fps=20,
                                          keep_frames=True)
    meta = dict(npz_path=npz_path, fig_paths=fig_paths, video_path=video_path, stem=stem)
    return results, meta


def main() -> None:
    _apply_dark_theme()

    K = _build_intrinsics()
    active_results, active_meta = _run_and_emit_case(active_pointing=True,  sanity=False, K=K)
    fixed_results,  fixed_meta  = _run_and_emit_case(active_pointing=False, sanity=False, K=K)

    comparison_paths = make_comparison_plots(
        active_results, fixed_results,
        stem=f"{OUTPUT['stem_base']}_fixed_vs_active",
    )

    sanity_results = None
    if bool(SANITY["enabled"]):
        sanity_results, _ = _run_and_emit_case(active_pointing=True, sanity=True, K=K)

    print("07 active tracking complete.")
    print()
    print("Fixed-camera baseline:")
    print(f"  video: {fixed_meta['video_path']}")
    print(f"  vis={fixed_results['visibility_fraction']:.3f}  "
          f"upd={fixed_results['update_fraction']:.3f}  "
          f"RMS_pos={fixed_results['rms_position_error']:.3e}  "
          f"RMS_vel={fixed_results['rms_velocity_error']:.3e}")
    print()
    print("Active-pointing run:")
    print(f"  video: {active_meta['video_path']}")
    print(f"  vis={active_results['visibility_fraction']:.3f}  "
          f"upd={active_results['update_fraction']:.3f}  "
          f"RMS_pos={active_results['rms_position_error']:.3e}  "
          f"RMS_vel={active_results['rms_velocity_error']:.3e}")
    if sanity_results is not None:
        print()
        print("Zero-noise sanity run:")
        print(f"  vis={sanity_results['visibility_fraction']:.3f}  "
              f"upd={sanity_results['update_fraction']:.3f}  "
              f"RMS_pos={sanity_results['rms_position_error']:.3e}  "
              f"RMS_vel={sanity_results['rms_velocity_error']:.3e}")
    print()
    print("Comparison plots:")
    for p in comparison_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
