from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from dynamics.integrators import propagate
from dynamics.cr3bp import CR3BP

from nav.ekf import ekf_propagate_cr3bp_stm
from nav.measurements.bearing import los_unit, tangent_basis, bearing_update_tangent


def add_angular_noise(u: np.ndarray, sigma_theta: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add small Gaussian angular noise around unit vector u using tangent-plane perturbation.
    """
    u = np.asarray(u, dtype=float).reshape(3,)
    u = u / np.linalg.norm(u)
    e1, e2 = tangent_basis(u)
    d = rng.normal(0.0, sigma_theta, size=(2,))
    up = u + d[0] * e1 + d[1] * e2
    return up / np.linalg.norm(up)


def main() -> None:
    rng = np.random.default_rng(0)

    mu = 0.0121505856
    model = CR3BP(mu=mu)

    # --- Time settings (nondimensional)
    t0 = 0.0
    tf = 6.0
    dt_meas = 0.02
    t_meas = np.arange(t0, tf + 1e-12, dt_meas)

    # --- Choose a target "body" position in rotating frame
    # easiest: Moon primary2 (constant in rotating frame)
    r_body = model.primary2

    # --- Truth initial condition (reuse near-L1 vibe)
    L1 = model.lagrange_points()["L1"]
    x_true0 = np.array([L1[0] - 1e-3, 0.0, 0.0, 0.0, 0.05, 0.0], dtype=float)

    # --- Propagate truth at measurement epochs
    res_truth = propagate(
        model.eom,
        (t0, tf),
        x_true0,
        t_eval=t_meas,
        rtol=1e-11,
        atol=1e-13,
        method="DOP853",
    )
    if not res_truth.success:
        raise RuntimeError(f"Truth propagation failed: {res_truth.message}")

    X_true = res_truth.x  # (N,6)

    # --- Simulate measurements: LOS unit vectors with angular noise
    sigma_theta = 2e-4  # radians (tune)
    U_meas = np.zeros((t_meas.size, 3), dtype=float)
    U_true = np.zeros((t_meas.size, 3), dtype=float)

    for k in range(t_meas.size):
        u_true, _ = los_unit(r_body, X_true[k, :3])
        U_true[k] = u_true
        U_meas[k] = add_angular_noise(u_true, sigma_theta, rng)

    # --- EKF init: truth + small perturbation
    x = X_true[0].copy()
    x[:3] += np.array([2e-4, -1e-4, 0.0])
    x[3:] += np.array([0.0, 2e-3, 0.0])

    P = np.diag([1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8]).astype(float)
    q_acc = 1e-12  # tune (process noise)

    X_hat = np.zeros_like(X_true)
    nis = np.full((t_meas.size,), np.nan)

    X_hat[0] = x
    t_prev = t_meas[0]

    for k in range(1, t_meas.size):
        tk = float(t_meas[k])

        # Propagate with STM and covariance
        x, P, _Phi = ekf_propagate_cr3bp_stm(mu=mu, x=x, P=P, t0=t_prev, t1=tk, q_acc=q_acc)

        # Update with bearing measurement
        x, P, y, nis_k = bearing_update_tangent(x, P, U_meas[k], r_body, sigma_theta)
        nis[k] = nis_k

        X_hat[k] = x
        t_prev = tk

    # --- Diagnostics
    pos_err = np.linalg.norm(X_hat[:, :3] - X_true[:, :3], axis=1)
    vel_err = np.linalg.norm(X_hat[:, 3:6] - X_true[:, 3:6], axis=1)
    print(f"Final |pos error| = {pos_err[-1]:.3e}")
    print(f"Final |vel error| = {vel_err[-1]:.3e}")
    print(f"Mean NIS (k>=1)   = {np.nanmean(nis[1:]):.3f}")

    # --- Plot (similar vibe to 02)
    fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax[0].plot(t_meas, pos_err, linewidth=1.2)
    ax[0].set_ylabel("||r_hat - r_true||")
    ax[0].grid(True, linewidth=0.5, alpha=0.6)
    ax[0].set_title("CR3BP bearings EKF (STM covariance)")

    ax[1].plot(t_meas, vel_err, linewidth=1.2)
    ax[1].set_xlabel("t (ND)")
    ax[1].set_ylabel("||v_hat - v_true||")
    ax[1].grid(True, linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig("results/plots/03_simulate_bearings_plot.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()