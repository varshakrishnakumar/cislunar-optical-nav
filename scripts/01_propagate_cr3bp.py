from __future__ import annotations
import numpy as np
import os
from dynamics.cr3bp import CR3BP
from dynamics.integrators import propagate

import matplotlib.pyplot as plt


def create_and_save_plot(X: np.ndarray, model: CR3BP, L: dict, option: str) -> None:
    p1 = model.primary1  # (-mu, 0, 0)
    p2 = model.primary2  # (1-mu, 0, 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(X[:, 0], X[:, 1], linewidth=2.0, label=f"Trajectory ({option})", color='blue')

    # primaries
    ax.scatter([p1[0]], [p1[1]], s=100, marker="o", label="Primary 1 (Earth)", color='green')
    ax.scatter([p2[0]], [p2[1]], s=80, marker="o", label="Primary 2 (Moon)", color='orange')

    # L points
    for k in ["L1", "L2", "L3", "L4", "L5"]:
        ax.scatter([L[k][0]], [L[k][1]], s=80, marker="x", label=k, color='red')

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    ax.set_title(f"CR3BP Earth-Moon: {option} Trajectory", fontsize=16)
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    
    # Ensure the directory exists
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig(f'results/plots/{option}_trajectory.png', dpi=300)
    plt.close()

def main() -> None:
    mu = 0.0121505856
    model = CR3BP(mu=mu, tiny=1e-12)

    L = model.lagrange_points()  
    print("Lagrange points (x,y,z):")
    for k in ["L1", "L2", "L3", "L4", "L5"]:
        v = L[k]
        print(f"  {k}: {v}")

    # Option A: near L1 with a small offset and small tangential velocity
    L1 = L["L1"]
    x0_A = np.array([
        L1[0] - 1e-3,  # slightly left of L1
        0.0,
        0.0,
        0.0,
        0.05,          # small vy
        0.0
    ], dtype=float)

    # Option B: near Earth (primary1) 
    p1 = model.primary1  # (-mu, 0, 0)
    x0_B = np.array([p1[0] + 0.05, 0.0, 0.0, 0.0, 0.6, 0.0], dtype=float)

    # --- Propagate for Option A
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

    X_A = res_A.sol(t_plot).T  # (N,6)

    create_and_save_plot(X_A, model, L, "01_cr3bp_nearL1")

    # --- Propagate for Option B
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

    X_B = res_B.sol(t_plot).T  # (N,6)

    create_and_save_plot(X_B, model, L, "01_cr3bp_nearEarth")

if __name__ == "__main__":
    main()
