# scripts/01_propagate_cr3bp.py
"""
Propagate a simple CR3BP trajectory in the Earth–Moon system and visualize it.

Run:
  python scripts/01_propagate_cr3bp.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from dynamics.cr3bp import CR3BP
from dynamics.integrators import propagate



def main() -> None:
    # --- 1) Earth–Moon mass parameter
    mu = 0.0121505856
    model = CR3BP(mu=mu, tiny=1e-12)

    # --- 2) Lagrange points
    L = model.lagrange_points()  # dict: "L1".. "L5" -> (3,)
    print("Lagrange points (x,y,z):")
    for k in ["L1", "L2", "L3", "L4", "L5"]:
        v = L[k]
        print(f"  {k}: {v}")

    # --- 3) Simple initial condition
    # Option A: near L1 with a small offset and small tangential velocity
    L1 = L["L1"]
    x0 = np.array([
        L1[0] - 1e-3,  # slightly left of L1
        0.0,
        0.0,
        0.0,
        0.05,          # small vy
        0.0
    ], dtype=float)

    # (Option B: near Earth (primary1) 
    # p1 = model.primary1  # (-mu, 0, 0)
    # x0 = np.array([p1[0] + 0.05, 0.0, 0.0, 0.0, 0.6, 0.0], dtype=float)

    # --- 4) Propagate with SciPy wrapper, dense output enabled
    t0, tf = 0.0, 10.0

    # Use a reasonable output grid for plotting (integration itself uses adaptive steps)
    t_plot = np.linspace(t0, tf, 5000)

    res = propagate(
        model.eom,
        (t0, tf),
        x0,
        dense_output=True,
        # You can tune these if you want tighter drift:
        rtol=1e-11,
        atol=1e-13,
        method="DOP853",
    )

    if not res.success:
        print("Propagation FAILED:", res.message)
        return

    # Evaluate dense solution at plotting times
    X = res.sol(t_plot).T  # (N,6) since SciPy sol returns (6,N)

    # --- 5) Jacobi drift
    C0 = model.jacobi(x0)
    xf = X[-1]
    Cf = model.jacobi(xf)

    print("\nJacobi drift:")
    print(f"  C0 = {C0:.16e}")
    print(f"  Cf = {Cf:.16e}")
    print(f"  |Cf - C0| = {abs(Cf - C0):.16e}")

    # --- 6) Plot x–y trajectory + primaries + L points
    p1 = model.primary1  # (-mu, 0, 0)
    p2 = model.primary2  # (1-mu, 0, 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(X[:, 0], X[:, 1], linewidth=1.0, label="trajectory")

    # primaries
    ax.scatter([p1[0]], [p1[1]], s=80, marker="o", label="Primary 1 (Earth)")
    ax.scatter([p2[0]], [p2[1]], s=60, marker="o", label="Primary 2 (Moon)")

    # L points
    for k in ["L1", "L2", "L3", "L4", "L5"]:
        ax.scatter([L[k][0]], [L[k][1]], s=60, marker="x", label=k)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("CR3BP Earth–Moon: x–y trajectory with primaries and L points")
    ax.grid(True, linewidth=0.5, alpha=0.6)

    # Avoid a giant legend if you prefer; but it's handy at first.
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
