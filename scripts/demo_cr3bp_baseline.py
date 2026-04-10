from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import ensure_src_on_path, repo_path

ensure_src_on_path()

from dynamics.cr3bp import CR3BP
from dynamics.integrators import propagate


def create_and_save_plot(X: np.ndarray, model: CR3BP, L: dict, option: str) -> None:
    p1 = model.primary1
    p2 = model.primary2

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(X[:, 0], X[:, 1], linewidth=2.0, label=f"Trajectory ({option})", color="blue")

    ax.scatter([p1[0]], [p1[1]], s=100, marker="o", label="Primary 1 (Earth)", color="green")
    ax.scatter([p2[0]], [p2[1]], s=80, marker="o", label="Primary 2 (Moon)", color="orange")

    for k in ["L1", "L2", "L3", "L4", "L5"]:
        ax.scatter([L[k][0]], [L[k][1]], s=80, marker="x", label=k, color="red")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    ax.set_title(f"CR3BP Earth-Moon: {option} Trajectory", fontsize=16)
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()

    plots_dir = repo_path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / f"{option}_trajectory.png", dpi=300)
    plt.close()


def main() -> None:
    mu = 0.0121505856
    model = CR3BP(mu=mu, tiny=1e-12)

    L = model.lagrange_points()
    print("Lagrange points (x,y,z):")
    for k in ["L1", "L2", "L3", "L4", "L5"]:
        v = L[k]
        print(f"  {k}: {v}")

    L1 = L["L1"]
    x0_A = np.array(
        [
            L1[0] - 1e-3,
            0.0,
            0.0,
            0.0,
            0.05,
            0.0,
        ],
        dtype=float,
    )

    p1 = model.primary1
    x0_B = np.array([p1[0] + 0.05, 0.0, 0.0, 0.0, 0.6, 0.0], dtype=float)

    t0, tf = 0.0, 10.0
    t_plot = np.linspace(t0, tf, 5000)

    res_A = propagate(
        model.eom,
        (t0, tf),
        x0_A,
        dense_output=True,
        rtol=1e-11,
        atol=1e-13,
        method="DOP853",
    )

    if not res_A.success:
        print("Propagation FAILED for Option A:", res_A.message)
        return

    X_A = res_A.sol(t_plot).T

    create_and_save_plot(X_A, model, L, "baseline_cr3bp_nearL1")

    res_B = propagate(
        model.eom,
        (t0, tf),
        x0_B,
        dense_output=True,
        rtol=1e-11,
        atol=1e-13,
        method="DOP853",
    )

    if not res_B.success:
        print("Propagation FAILED for Option B:", res_B.message)
        return

    X_B = res_B.sol(t_plot).T

    create_and_save_plot(X_B, model, L, "baseline_cr3bp_nearEarth")


if __name__ == "__main__":
    main()
