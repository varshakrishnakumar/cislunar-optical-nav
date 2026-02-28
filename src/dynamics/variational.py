import numpy as np


def cr3bp_eom_with_stm(t: float, z: np.ndarray, mu: float) -> np.ndarray:
    """
    CR3BP (a.k.a. Circular Restricted 3-Body Problem) EOM + 6x6 STM.

    Rotating barycentric frame, nondimensional units:
      - distance between primaries = 1
      - mean motion n = 1
      - total mass = 1
      - mu = m2 / (m1 + m2), with m1 = 1-mu, m2 = mu
      - primaries located at (-mu, 0, 0) and (1-mu, 0, 0)

    State ordering:
      X = [x, y, z, xd, yd, zd]
      Phi is flattened column-major (MATLAB style) in z[6:42].

    Returns:
      dz/dt = [dX/dt; dPhi/dt(:)]
    """
    z = np.asarray(z, dtype=float).reshape(-1,)
    if z.size != 42:
        raise ValueError(f"Expected z of length 42 (6 + 36), got {z.size}")

    # Unpack state
    x, y, zz, xd, yd, zd = z[:6]

    # Distances to primaries
    r1_vec = np.array([x + mu, y, zz], dtype=float)          # to m1 at (-mu,0,0)
    r2_vec = np.array([x - (1.0 - mu), y, zz], dtype=float)  # to m2 at (1-mu,0,0)
    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    # Guard against singularities
    if r1 == 0.0 or r2 == 0.0:
        raise ZeroDivisionError("CR3BP singularity: particle at a primary (r1=0 or r2=0).")

    # Effective potential Omega:
    # Omega = 0.5*(x^2 + y^2) + (1-mu)/r1 + mu/r2
    # Its gradient:
    mu1 = 1.0 - mu
    r1_3 = r1**3
    r2_3 = r2**3

    Omega_x = x - mu1 * (x + mu) / r1_3 - mu * (x - (1.0 - mu)) / r2_3
    Omega_y = y - mu1 * y / r1_3 - mu * y / r2_3
    Omega_z =     - mu1 * zz / r1_3 - mu * zz / r2_3

    # CR3BP acceleration = grad(Omega) + Coriolis terms
    xdd =  2.0 * yd + Omega_x
    ydd = -2.0 * xd + Omega_y
    zdd = Omega_z

    # --- Second derivatives of Omega (for STM A-matrix) ---
    # Helpful powers
    r1_5 = r1**5
    r2_5 = r2**5

    dx1, dy1, dz1 = r1_vec
    dx2, dy2, dz2 = r2_vec

    # Uxx, Uyy, Uzz, Uxy, Uxz, Uyz (Omega_xx, etc.)
    # For term m/r: d2/dx2 (m/r) = m*(3*dx^2/r^5 - 1/r^3)
    # Cross: d2/dxdy (m/r) = m*(3*dx*dy/r^5)
    def d2_terms(mass: float, dx: float, dy: float, dz: float, r3: float, r5: float):
        c = mass
        xx = c * (3.0 * dx * dx / r5 - 1.0 / r3)
        yy = c * (3.0 * dy * dy / r5 - 1.0 / r3)
        zz_ = c * (3.0 * dz * dz / r5 - 1.0 / r3)
        xy = c * (3.0 * dx * dy / r5)
        xz = c * (3.0 * dx * dz / r5)
        yz = c * (3.0 * dy * dz / r5)
        return xx, yy, zz_, xy, xz, yz

    t1 = d2_terms(mu1, dx1, dy1, dz1, r1_3, r1_5)
    t2 = d2_terms(mu,  dx2, dy2, dz2, r2_3, r2_5)

    # Omega includes +0.5*(x^2+y^2) => second deriv adds +1 to xx and yy, +0 to zz
    Omega_xx = 1.0 + (t1[0] + t2[0])
    Omega_yy = 1.0 + (t1[1] + t2[1])
    Omega_zz =       (t1[2] + t2[2])
    Omega_xy =       (t1[3] + t2[3])
    Omega_xz =       (t1[4] + t2[4])
    Omega_yz =       (t1[5] + t2[5])

    # --- Build A matrix for variational equations dPhi/dt = A Phi ---
    # State order: [x y z xd yd zd]
    O3 = np.zeros((3, 3), dtype=float)
    I3 = np.eye(3, dtype=float)

    # Coriolis coupling in partials of acceleration wrt velocity:
    # xdd = 2*yd + Omega_x  => d(xdd)/d(xd)=0, d(xdd)/d(yd)=2, d(xdd)/d(zd)=0
    # ydd = -2*xd + Omega_y => d(ydd)/d(xd)=-2, d(ydd)/d(yd)=0, d(ydd)/d(zd)=0
    # zdd = Omega_z         => zeros
    C = np.array([[0.0,  2.0, 0.0],
                  [-2.0, 0.0, 0.0],
                  [0.0,  0.0, 0.0]], dtype=float)

    # Acceleration partials wrt position (Hessian of Omega)
    U = np.array([[Omega_xx, Omega_xy, Omega_xz],
                  [Omega_xy, Omega_yy, Omega_yz],
                  [Omega_xz, Omega_yz, Omega_zz]], dtype=float)

    A = np.block([[O3, I3],
                  [U,  C]])

    # STM
    phi = z[6:].reshape((6, 6), order="F")
    dphidt = A @ phi

    # dX/dt
    dxdt = np.array([xd, yd, zd, xdd, ydd, zdd], dtype=float)

    return np.concatenate([dxdt, dphidt.reshape(-1, order="F")])

def cr3bp_eom(t: float, x: np.ndarray, mu: float) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(6,)
    z = np.concatenate([x, np.eye(6, dtype=float).reshape(-1, order="F")])
    return cr3bp_eom_with_stm(t, z, mu)[:6]