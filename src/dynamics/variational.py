from __future__ import annotations

from functools import lru_cache

import numpy as np

from dynamics.models import CR3BPDynamics


# ---------------------------------------------------------------------------
# Optional Numba JIT — ~10-50× faster RHS; also releases the GIL so the MC
# ThreadPoolExecutor can overlap integrations across CPU cores.
# Install with: pip install cislunar-optical-nav[fast]
# ---------------------------------------------------------------------------
try:
    import numba as _numba

    @_numba.njit(cache=True)
    def _cr3bp_eom_with_stm_nb(t: float, z: np.ndarray, mu: float) -> np.ndarray:
        """42-DOF CR3BP + STM right-hand side, compiled to native code.

        State layout (column-major STM): z[0:6] = [x,y,z,vx,vy,vz],
        z[6:42] = phi reshaped column-major, i.e. phi[:,j] = z[6+6j:12+6j].
        """
        tiny2 = 1e-24           # (1e-12)^2 softening term
        x,  y,  zc = z[0], z[1], z[2]
        vx, vy, vz = z[3], z[4], z[5]
        nu   = 1.0 - mu
        dx1  = x + mu
        dx2  = x - nu

        r1   = (dx1*dx1 + y*y + zc*zc + tiny2) ** 0.5
        r2   = (dx2*dx2 + y*y + zc*zc + tiny2) ** 0.5
        r1_3 = r1 * r1 * r1
        r2_3 = r2 * r2 * r2
        r1_5 = r1_3 * r1 * r1
        r2_5 = r2_3 * r2 * r2

        # grad Omega
        dOx = x  - nu * dx1 / r1_3 - mu * dx2 / r2_3
        dOy = y  - nu * y   / r1_3 - mu * y   / r2_3
        dOz =    - nu * zc  / r1_3 - mu * zc  / r2_3

        # Hessian of Omega (upper triangle, symmetric)
        Oxx = 1.0 + nu*(3.0*dx1*dx1/r1_5 - 1.0/r1_3) + mu*(3.0*dx2*dx2/r2_5 - 1.0/r2_3)
        Oyy = 1.0 + nu*(3.0*y*y    /r1_5 - 1.0/r1_3) + mu*(3.0*y*y    /r2_5 - 1.0/r2_3)
        Ozz =       nu*(3.0*zc*zc  /r1_5 - 1.0/r1_3) + mu*(3.0*zc*zc  /r2_5 - 1.0/r2_3)
        Oxy = 3.0*(nu*dx1*y  /r1_5 + mu*dx2*y  /r2_5)
        Oxz = 3.0*(nu*dx1*zc /r1_5 + mu*dx2*zc /r2_5)
        Oyz = 3.0*(nu*y*zc   /r1_5 + mu*y*zc   /r2_5)

        # Pack the 42-element output vector
        out = np.empty(42)
        out[0] = vx;  out[1] = vy;  out[2] = vz
        out[3] = 2.0*vy + dOx
        out[4] = -2.0*vx + dOy
        out[5] = dOz

        # dPhi/dt = A @ Phi, processed one column at a time to avoid reshape.
        # A = [[0,0,0,  1, 0,0],
        #      [0,0,0,  0, 1,0],
        #      [0,0,0,  0, 0,1],
        #      [Oxx,Oxy,Oxz, 0, 2,0],
        #      [Oxy,Oyy,Oyz,-2, 0,0],
        #      [Oxz,Oyz,Ozz, 0, 0,0]]
        for j in range(6):
            b  = 6 + 6*j
            p0, p1, p2 = z[b],   z[b+1], z[b+2]
            p3, p4, p5 = z[b+3], z[b+4], z[b+5]
            out[b]   = p3
            out[b+1] = p4
            out[b+2] = p5
            out[b+3] = Oxx*p0 + Oxy*p1 + Oxz*p2 + 2.0*p4
            out[b+4] = Oxy*p0 + Oyy*p1 + Oyz*p2 - 2.0*p3
            out[b+5] = Oxz*p0 + Oyz*p1 + Ozz*p2
        return out

    _NUMBA_AVAILABLE = True

except ImportError:
    _NUMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public entry point (called by EKF / targeting via propagate(..., args=(mu,)))
# ---------------------------------------------------------------------------
def cr3bp_eom_with_stm(t: float, z: np.ndarray, mu: float) -> np.ndarray:
    if _NUMBA_AVAILABLE:
        return _cr3bp_eom_with_stm_nb(t, np.asarray(z, dtype=np.float64), float(mu))
    return _cr3bp_model(float(mu)).eom_with_stm(t, z)


@lru_cache(maxsize=16)
def _cr3bp_model(mu: float) -> CR3BPDynamics:
    return CR3BPDynamics(mu=mu)