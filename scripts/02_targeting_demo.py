
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from dynamics.integrators import propagate
from dynamics.variational import cr3bp_eom_with_stm

from dynamics.cr3bp import CR3BP


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


def propagate_cr3bp(mu: float, t0: float, tf: float, x0: np.ndarray, *, t_eval=None, dense=False):
    return propagate(
        lambda t, x: CR3BP(mu=mu).eom(t, x),
        (t0, tf),
        x0,
        t_eval=t_eval,
        dense_output=dense,
        rtol=1e-11,
        atol=1e-13,
        method="DOP853",
    )


def propagate_cr3bp_with_stm(mu: float, t0: float, tf: float, z0: np.ndarray, *, dense=False):
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
    x0 = np.asarray(x0, dtype=float).reshape(6,)
    r_target = np.asarray(r_target, dtype=float).reshape(3,)

    z0 = pack_state_and_stm(x0, np.eye(6))
    res_tc = propagate_cr3bp_with_stm(mu, t0, tc, z0, dense=False)
    if not res_tc.success:
        raise RuntimeError(f"Prop to tc failed: {res_tc.message}")
    x_tc, _ = unpack_state_and_stm(res_tc.x[-1])

    dv = np.zeros(3) if dv0 is None else np.asarray(dv0, dtype=float).reshape(3,)
    hist = []

    for k in range(1, max_iter + 1):
        x_burn = x_tc.copy()
        x_burn[3:6] += dv

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

        try:
            delta = np.linalg.solve(Phi_rv, err)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(Phi_rv, err, rcond=None)

        dv = dv - delta

    return dv, {"converged": False, "iterations": max_iter, "history": hist, "final_err": hist[-1]["err"]}


def main() -> None:
    mu = 0.0121505856
    model = CR3BP(mu=mu)

    t0 = 0.0
    tf = 6.0
    tc = 2.0

    L = model.lagrange_points()
    L1 = L["L1"]
    x0_nom = np.array([L1[0] - 1e-3, 0.0, 0.0, 0.0, 0.05, 0.0], dtype=float)

    res_nom = propagate(model.eom, (t0, tf), x0_nom, dense_output=True, rtol=1e-11, atol=1e-13, method="DOP853")
    if not res_nom.success or res_nom.sol is None:
        raise RuntimeError(f"Nominal propagation failed: {res_nom.message}")
    x_tf_nom = res_nom.sol(tf).reshape(6,)
    r_target = x_tf_nom[:3]

    dx0 = np.array([2e-4, -1e-4, 0.0, 0.0, 2e-3, 0.0], dtype=float)
    x0_err = x0_nom + dx0

    dv, info = solve_single_impulse_position_target(
        mu=mu, x0=x0_err, t0=t0, tc=tc, tf=tf, r_target=r_target, max_iter=10, tol=1e-10
    )

    print("Targeting result:")
    print(f"  converged: {info['converged']}")
    print(f"  iterations: {info['iterations']}")
    print(f"  dv = {dv} (ND units)")
    print(f"  |dv| = {np.linalg.norm(dv):.6e}")
    print(f"  final position error = {info['final_err']}, norm={np.linalg.norm(info['final_err']):.3e}")

    t_plot = np.linspace(t0, tf, 4000)

    X_nom = res_nom.sol(t_plot).T

    res_unc = propagate(model.eom, (t0, tf), x0_err, dense_output=True, rtol=1e-11, atol=1e-13, method="DOP853")
    X_unc = res_unc.sol(t_plot).T

    res_to_tc = propagate(model.eom, (t0, tc), x0_err, dense_output=True, rtol=1e-11, atol=1e-13, method="DOP853")
    x_tc = res_to_tc.sol(tc).reshape(6,)
    x_tc[3:6] += dv
    res_cor = propagate(model.eom, (tc, tf), x_tc, dense_output=True, rtol=1e-11, atol=1e-13, method="DOP853")

    X_cor_1 = res_to_tc.sol(t_plot[t_plot <= tc]).T
    X_cor_2 = res_cor.sol(t_plot[t_plot >= tc]).T
    X_cor = np.vstack([X_cor_1, X_cor_2])

    p1 = model.primary1
    p2 = model.primary2

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(X_nom[:, 0], X_nom[:, 1], linewidth=1.2, label="Nominal")
    ax.plot(X_unc[:, 0], X_unc[:, 1], linewidth=1.0, linestyle="--", label="Off-nominal (uncorrected)")
    ax.plot(X_cor[:, 0], X_cor[:, 1], linewidth=1.2, label="Corrected (1 burn)")

    ax.scatter([p1[0]], [p1[1]], s=80, marker="o", label="Earth")
    ax.scatter([p2[0]], [p2[1]], s=60, marker="o", label="Moon")

    ax.scatter([r_target[0]], [r_target[1]], s=50, marker="x", label="Target r(tf)")
    ax.axvline(0.0, linewidth=0.6, alpha=0.4)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("CR3BP deterministic targeting: single midcourse burn")
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()

    plt.savefig('results/plots/02_targeting_demo_plot.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
