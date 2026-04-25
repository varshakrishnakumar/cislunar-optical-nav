from __future__ import annotations

from pathlib import Path

from _common import ensure_src_on_path

ensure_src_on_path()

import numpy as np
import matplotlib.ticker as ticker
from scipy.stats import chi2
from visualization.style import plt

from dynamics.integrators import propagate
from dynamics.cr3bp import CR3BP
from nav.ekf import ekf_propagate_cr3bp_stm
from nav.measurements.bearing import los_unit, tangent_basis, bearing_update_tangent


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


def _apply_dark_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor":   _BG,
        "axes.facecolor":     _PANEL,
        "axes.edgecolor":     _BORDER,
        "axes.labelcolor":    _TEXT,
        "axes.titlecolor":    _TEXT,
        "text.color":         _TEXT,
        "xtick.color":        _TEXT,
        "ytick.color":        _TEXT,
        "grid.color":         _BORDER,
        "grid.alpha":         1.0,
        "grid.linestyle":     "--",
        "lines.linewidth":    2.0,
        "legend.facecolor":   _PANEL,
        "legend.edgecolor":   _BORDER,
        "legend.labelcolor":  _TEXT,
        "savefig.facecolor":  _BG,
        "savefig.edgecolor":  _BG,
        "font.size":          11,
        "axes.titlesize":     12,
        "axes.labelsize":     11,
        "legend.fontsize":    10,
    })


def add_angular_noise(u: np.ndarray, sigma_theta: float, rng: np.random.Generator) -> np.ndarray:
    u = np.asarray(u, dtype=float).reshape(3)
    u = u / np.linalg.norm(u)
    e1, e2 = tangent_basis(u)
    d = rng.normal(0.0, sigma_theta, size=2)
    up = u + d[0] * e1 + d[1] * e2
    return up / np.linalg.norm(up)


def main() -> None:
    _apply_dark_theme()

    Path("results/demos").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    mu = 0.0121505856
    model = CR3BP(mu=mu)

    L1x = model.lagrange_points()["L1"][0]
    x0_nom = np.array([L1x - 1e-3, 0.0, 0.0, 0.0, 0.05, 0.0], dtype=float)

    r_body = np.asarray(model.primary2, dtype=float)

    t0, tf    = 0.0, 6.0
    dt_meas   = 0.02
    t_meas    = np.arange(t0, tf + 1e-12, dt_meas)

    x_true0 = x0_nom.copy()
    x_true0[:3] += np.array([2e-4, -1e-4, 0.0])
    x_true0[3:] += np.array([0.0,  2e-3, 0.0])

    res = propagate(model.eom, (t0, tf), x_true0, t_eval=t_meas,
                    rtol=1e-11, atol=1e-13, method="DOP853")
    if not res.success:
        raise RuntimeError(f"Truth propagation failed: {res.message}")
    X_true = res.x

    sigma_theta = 2e-4
    U_meas = np.zeros((t_meas.size, 3), dtype=float)
    for k in range(t_meas.size):
        u_true, _ = los_unit(r_body, X_true[k, :3])
        U_meas[k] = add_angular_noise(u_true, sigma_theta, rng)

    x = x0_nom.copy()
    P = np.diag([1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8]).astype(float)
    q_acc = 1e-12

    N = t_meas.size
    X_hat   = np.zeros((N, 6), dtype=float)
    nis     = np.full(N, np.nan, dtype=float)
    P_diag  = np.zeros((N, 6), dtype=float)

    X_hat[0]  = x
    P_diag[0] = np.diag(P)
    t_prev    = t_meas[0]

    for k in range(1, N):
        tk = float(t_meas[k])
        x, P, _ = ekf_propagate_cr3bp_stm(
            mu=mu, x=x, P=P, t0=t_prev, t1=tk, q_acc=q_acc
        )

        upd = bearing_update_tangent(x, P, U_meas[k], r_body, sigma_theta)
        if upd.accepted:
            x, P = upd.x_upd, upd.P_upd
        nis[k] = upd.nis

        X_hat[k]  = x
        P_diag[k] = np.diag(P)
        t_prev    = tk

    pos_err  = np.linalg.norm(X_hat[:, :3] - X_true[:, :3], axis=1)
    vel_err  = np.linalg.norm(X_hat[:, 3:6] - X_true[:, 3:6], axis=1)

    sig_x = 3.0 * np.sqrt(np.abs(P_diag[:, 0]))
    sig_y = 3.0 * np.sqrt(np.abs(P_diag[:, 1]))

    err_x = X_hat[:, 0] - X_true[:, 0]
    err_y = X_hat[:, 1] - X_true[:, 1]

    nis_lo = chi2.ppf(0.025, df=2)
    nis_hi = chi2.ppf(0.975, df=2)
    nis_ok = np.isfinite(nis)

    print(f"Final ||pos error|| = {pos_err[-1]:.3e} dimensionless CR3BP length")
    print(f"Final ||vel error|| = {vel_err[-1]:.3e} dimensionless CR3BP velocity")
    print(f"Mean NIS (k≥1)      = {np.nanmean(nis[1:]):.3f}  (expect ~2)")

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True,
                             gridspec_kw={"hspace": 0.08})
    fig.patch.set_facecolor(_BG)

    ax = axs[0]
    ax.plot(t_meas, pos_err, color=_CYAN, lw=1.8, label="‖r̂ − r‖", zorder=3)
    ax.plot(t_meas, sig_x, color=_VIOLET, lw=1.0, ls="--", alpha=0.7, label="3σ x")
    ax.plot(t_meas, sig_y, color=_VIOLET, lw=1.0, ls=":",  alpha=0.7, label="3σ y")
    ax.set_ylabel("Position error  [dimensionless CR3BP length]", color=_TEXT)
    ax.set_title("Bearing-Only EKF — Near-L1 CR3BP Navigation", color=_TEXT,
                 fontsize=14, pad=10)
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(ticker.LogLocator(subs="all"))
    ax.tick_params(which="minor", length=3, color=_BORDER)

    ax = axs[1]
    ax.plot(t_meas, vel_err, color=_AMBER, lw=1.8, label="‖v̂ − v‖", zorder=3)
    ax.set_ylabel("Velocity error  [dimensionless CR3BP velocity]", color=_TEXT)
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.set_yscale("log")

    ax = axs[2]
    ax.fill_between(t_meas, nis_lo, nis_hi, color=_GREEN, alpha=0.12,
                    label=f"95% χ²(2) band [{nis_lo:.2f}, {nis_hi:.2f}]")
    ax.axhline(2.0, color=_GREEN, lw=1.0, ls="--", alpha=0.6)
    for k in np.where(nis_ok)[0]:
        c = _GREEN if nis_lo <= nis[k] <= nis_hi else _RED
        ax.scatter(t_meas[k], nis[k], s=8, color=c, zorder=4)
    ax.set_ylabel("NIS", color=_TEXT)
    ax.set_xlabel("Time  [dimensionless CR3BP time]", color=_TEXT)
    ax.legend(loc="upper right")
    ax.grid(True)

    for a in axs:
        a.set_facecolor(_PANEL)
        for spine in a.spines.values():
            spine.set_edgecolor(_BORDER)

    fig.savefig("results/demos/03_bearings_report.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                             gridspec_kw={"hspace": 0.08})
    fig.patch.set_facecolor(_BG)

    for ax, err, sig, label, col in [
        (axs[0], err_x, sig_x, "x-axis position error  [dimensionless CR3BP length]", _CYAN),
        (axs[1], err_y, sig_y, "y-axis position error  [dimensionless CR3BP length]", _AMBER),
    ]:
        ax.fill_between(t_meas, -sig, sig, color=_VIOLET, alpha=0.18, label="±3σ")
        ax.plot(t_meas, err, color=col, lw=1.5, label="error", zorder=3)
        ax.axhline(0.0, color=_BORDER, lw=0.8)
        ax.set_ylabel(label, color=_TEXT)
        ax.legend(loc="upper right")
        ax.grid(True)
        ax.set_facecolor(_PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(_BORDER)

    axs[0].set_title("Position 3σ Consistency — Bearing-Only EKF", color=_TEXT,
                      fontsize=13, pad=8)
    axs[1].set_xlabel("Time  [dimensionless CR3BP time]", color=_TEXT)

    fig.savefig("results/demos/03_bearings_consistency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:")
    print("  results/demos/03_bearings_report.png")
    print("  results/demos/03_bearings_consistency.png")


if __name__ == "__main__":
    main()
