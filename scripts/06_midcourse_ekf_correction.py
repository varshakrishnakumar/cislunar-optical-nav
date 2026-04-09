from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import chi2

from dynamics.cr3bp import CR3BP
from dynamics.integrators import propagate
from nav.ekf import ekf_propagate_cr3bp_stm
from nav.measurements.bearing import bearing_update_tangent
from nav.measurements.pixel_bearing import pixel_detection_to_bearing
from cv.camera import Intrinsics
from cv.pointing import camera_dcm_from_boresight
from cv.sim_measurements import simulate_pixel_measurement
from guidance.targeting import solve_single_impulse_position_target


CameraMode = Literal["fixed", "truth_tracking", "estimate_tracking"]
_VALID_CAMERA_MODES = ("fixed", "truth_tracking", "estimate_tracking")


_BG     = "#080B14"
_PANEL  = "#0E1220"
_BORDER = "#1C2340"
_TEXT   = "#DCE0EC"
_DIM    = "#5A6080"
_CYAN   = "#22D3EE"
_AMBER  = "#F59E0B"
_GREEN  = "#10B981"
_VIOLET = "#8B5CF6"
_RED    = "#F43F5E"
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
    if isinstance(camera_mode, bool):
        return "fixed" if camera_mode else "estimate_tracking"
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
    P     = np.diag([1e-6] * 6).astype(float)
    q_acc = 1e-14

    x_hat_hist:  list[np.ndarray] = []
    nis_list:    list[float]       = []
    pos_err_list: list[float]      = []
    P_diag_list: list[np.ndarray]  = []
    valid_arr    = np.zeros(k_tc + 1, dtype=bool)

    for k in range(1, k_tc + 1):
        x_hat, P, _ = ekf_propagate_cr3bp_stm(
            mu=float(mu), x=x_hat, P=P,
            t0=float(t_meas[k - 1]), t1=float(t_meas[k]), q_acc=q_acc,
        )

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
                nis_list.append(float(upd.nis))
                valid_arr[k] = True
            else:
                nis_list.append(float("nan"))
        else:
            nis_list.append(float("nan"))

        x_hat_hist.append(x_hat.copy())
        pos_err_list.append(_norm(x_hat[:3] - xs_true[k, :3]))
        P_diag_list.append(np.diag(P).copy())

    x_hat_arr   = np.asarray(x_hat_hist)
    nis_arr     = np.asarray(nis_list)
    pos_err_arr = np.asarray(pos_err_list)
    P_diag_arr  = np.asarray(P_diag_list)

    x_true_tc    = xs_true[k_tc]
    x_hat_tc     = x_hat.copy()
    P_tc         = P.copy()
    pos_err_tc   = _norm(x_hat_tc[:3] - x_true_tc[:3])
    tracePpos_tc = float(np.trace(P_tc[:3, :3]))
    valid_rate   = float(np.mean(valid_arr[: k_tc + 1]))
    nis_finite   = nis_arr[np.isfinite(nis_arr)]
    nis_mean     = float(np.mean(nis_finite)) if nis_finite.size else float("nan")

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

    return {
        "tc":           tc_eff,
        "sigma_px":     float(sigma_px),
        "dropout_prob": float(dropout_prob),
        "camera_mode":  camera_mode,
        "dv_perfect_mag":   dv_perfect_mag,
        "dv_ekf_mag":       dv_ekf_mag,
        "dv_delta_mag":     dv_delta_mag,
        "dv_inflation":     dv_ekf_mag - dv_perfect_mag,
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
        "debug": {
            "t_meas":       t_meas,
            "k_tc":         k_tc,
            "xs_nom":       res_nom.x,
            "xs_true":      xs_true,
            "x_hat_hist":   x_hat_arr,
            "pos_err_hist": pos_err_arr,
            "P_diag_hist":  P_diag_arr,
            "nis_hist":     nis_arr,
            "xs_unc_tf":    res_unc.x,
            "xs_perf_tf":   res_perf.x,
            "xs_ekf_tf":    res_ekf.x,
            "dv_perf":      dv_perf,
            "dv_ekf":       dv_ekf,
            "r_target":     r_target,
        },
    }


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
    t_nom   = np.linspace(t0, tf, len(dbg["xs_nom"]))
    t_post  = np.linspace(float(t_meas[k_tc]), tf, len(dbg["xs_unc_tf"]))
    xs_nom  = dbg["xs_nom"]
    xs_true = dbg["xs_true"]
    x_hat   = dbg["x_hat_hist"]
    pos_err = dbg["pos_err_hist"]
    P_diag  = dbg["P_diag_hist"]
    nis     = dbg["nis_hist"]
    xs_unc  = dbg["xs_unc_tf"]
    xs_perf = dbg["xs_perf_tf"]
    xs_ekf  = dbg["xs_ekf_tf"]
    r_tgt   = dbg["r_target"]
    sig_pos = 3.0 * np.sqrt(np.abs(P_diag[:, 0]))

    print(f"  camera_mode   = {out['camera_mode']}")
    print(f"  |dv| perfect  = {out['dv_perfect_mag']:.4e} ND")
    print(f"  |dv| EKF      = {out['dv_ekf_mag']:.4e} ND")
    print(f"  miss_ekf      = {out['miss_ekf']:.4e} ND")
    print(f"  NIS mean      = {out['nis_mean']:.3f}")
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
    ax.set_xlabel("x  [ND]", color=_TEXT)
    ax.set_ylabel("y  [ND]", color=_TEXT)
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
    ax.set_xlabel("t  [ND]", color=_TEXT)
    ax.set_ylabel("Error  [ND]", color=_TEXT)
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
    ax.set_xlabel("t  [ND]", color=_TEXT)
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
    ax.set_ylabel("Miss distance  [ND]", color=_TEXT)
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
    ax.set_ylabel("Δv  [ND]", color=_TEXT)
    ax.grid(True, axis="y")
    infl = out["dv_inflation_pct"]
    if np.isfinite(infl):
        ax.text(0.98, 0.97, f"Inflation: {infl*100:+.2f}%",
                transform=ax.transAxes, ha="right", va="top",
                color=_ORANGE, fontsize=11,
                bbox=dict(facecolor=_PANEL, edgecolor=_BORDER, pad=5))
    fig.savefig(plots_dir / "06_dv_compare.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:")
    print(f"  {plots_dir / '06_midcourse_report.png'}")
    print(f"  {plots_dir / '06_dv_compare.png'}")


if __name__ == "__main__":
    main()
