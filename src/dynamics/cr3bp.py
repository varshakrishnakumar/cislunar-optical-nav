# crbp.py
"""
CR3BP / crbp in normalized rotating frame with:
- singularity protection via softening length `tiny`
- NumPy arrays for speed and easy vectorization later

Normalized units:
- Distance between primaries = 1
- mu = m2 / (m1 + m2), 0 < mu < 0.5
- Rotating frame angular rate = 1
- Primaries at (-mu,0,0) and (1-mu,0,0)

State: [x, y, z, vx, vy, vz] in rotating frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class CR3BP:
    mu: float
    tiny: float = 1e-12  # softening length (normalized distance units)

    def __post_init__(self) -> None:
        if not (0.0 < self.mu < 0.5):
            raise ValueError("mu must be in (0, 0.5)")
        if not (self.tiny >= 0.0):
            raise ValueError("tiny must be >= 0")

    @property
    def primary1(self) -> Array:
        return np.array([-self.mu, 0.0, 0.0], dtype=float)

    @property
    def primary2(self) -> Array:
        return np.array([1.0 - self.mu, 0.0, 0.0], dtype=float)

    # -------------------------
    # Singularity-protected distances
    # -------------------------
    def r1_r2(self, x: float, y: float, z: float) -> Tuple[float, float]:
        """
        Returns (r1, r2) with softening:
            r = sqrt(dx^2 + dy^2 + dz^2 + tiny^2)
        so r never becomes exactly 0.
        """
        mu = self.mu
        tiny2 = self.tiny * self.tiny

        dx1 = x + mu
        dx2 = x - (1.0 - mu)

        r1 = np.sqrt(dx1 * dx1 + y * y + z * z + tiny2)
        r2 = np.sqrt(dx2 * dx2 + y * y + z * z + tiny2)
        return float(r1), float(r2)

    # -------------------------
    # Potential and derivatives
    # -------------------------
    def Omega(self, x: float, y: float, z: float) -> float:
        mu = self.mu
        r1, r2 = self.r1_r2(x, y, z)
        return float(0.5 * (x * x + y * y) + (1.0 - mu) / r1 + mu / r2)

    def grad_Omega(self, x: float, y: float, z: float) -> Array:
        """
        Gradient of Omega: [dOx, dOy, dOz]
        With softened r1, r2 so r^3 is bounded away from 0.
        """
        mu = self.mu

        dx1 = x + mu
        dx2 = x - (1.0 - mu)

        r1, r2 = self.r1_r2(x, y, z)
        r1_3 = r1 * r1 * r1
        r2_3 = r2 * r2 * r2

        dOx = x - (1.0 - mu) * dx1 / r1_3 - mu * dx2 / r2_3
        dOy = y - (1.0 - mu) * y / r1_3 - mu * y / r2_3
        dOz = - (1.0 - mu) * z / r1_3 - mu * z / r2_3
        return np.array([dOx, dOy, dOz], dtype=float)

    # -------------------------
    # Dynamics
    # -------------------------
    def eom(self, t: float, s: Array) -> Array:
        """
        Equations of motion in rotating frame.
        Input: s shape (6,)
        Output: ds/dt shape (6,)
        """
        # unpack
        x, y, z, vx, vy, vz = s
        dOx, dOy, dOz = self.grad_Omega(float(x), float(y), float(z))

        ax = 2.0 * vy + dOx
        ay = -2.0 * vx + dOy
        az = dOz

        return np.array([vx, vy, vz, ax, ay, az], dtype=float)

    def jacobi(self, s: Array) -> float:
        """
        Jacobi constant C = 2*Omega - v^2
        """
        x, y, z = float(s[0]), float(s[1]), float(s[2])
        v2 = float(np.dot(s[3:6], s[3:6]))
        return float(2.0 * self.Omega(x, y, z) - v2)

    # -------------------------
    # Lagrange points
    # -------------------------
    def _collinear_eq(self, x: float) -> float:
        # dOmega/dx at (x,0,0)
        return float(self.grad_Omega(x, 0.0, 0.0)[0])

    @staticmethod
    def _bisect(
        f: Callable[[float], float],
        a: float,
        b: float,
        *,
        tol: float = 1e-12,
        max_iter: int = 200
    ) -> float:
        fa = f(a)
        fb = f(b)
        if fa == 0.0:
            return a
        if fb == 0.0:
            return b
        if fa * fb > 0.0:
            raise ValueError("Bisection requires opposite signs at the bracket endpoints.")

        lo, hi = a, b
        flo, fhi = fa, fb
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            fmid = f(mid)
            if abs(fmid) < tol or 0.5 * (hi - lo) < tol:
                return mid
            if flo * fmid <= 0.0:
                hi, fhi = mid, fmid
            else:
                lo, flo = mid, fmid
        return 0.5 * (lo + hi)

    def lagrange_points(self) -> Dict[str, Array]:
        """
        Returns dict of L1..L5 positions as np arrays shape (3,).
        L4/L5 exact; L1/L2/L3 via bracketing + bisection.
        """
        mu = self.mu
        x1 = -mu
        x2 = 1.0 - mu
        eps = 1e-6

        f = self._collinear_eq

        # L1: between primaries
        a1, b1 = x1 + eps, x2 - eps
        L1x = self._bisect(f, a1, b1)

        # L2: right of m2
        a2 = x2 + eps
        b2 = x2 + 2.0
        fa2 = f(a2)
        fb2 = f(b2)
        grow = 0
        while fa2 * fb2 > 0.0 and grow < 60:
            b2 *= 1.5
            fb2 = f(b2)
            grow += 1
        if fa2 * fb2 > 0.0:
            raise RuntimeError("Failed to bracket L2 root; try expanding search or check mu.")
        L2x = self._bisect(f, a2, b2)

        # L3: left of m1
        b3 = x1 - eps
        a3 = x1 - 2.0
        fb3 = f(b3)
        fa3 = f(a3)
        grow = 0
        while fa3 * fb3 > 0.0 and grow < 60:
            a3 *= 1.5
            fa3 = f(a3)
            grow += 1
        if fa3 * fb3 > 0.0:
            raise RuntimeError("Failed to bracket L3 root; try expanding search or check mu.")
        L3x = self._bisect(f, a3, b3)

        L4 = np.array([0.5 - mu,  np.sqrt(3.0) / 2.0, 0.0], dtype=float)
        L5 = np.array([0.5 - mu, -np.sqrt(3.0) / 2.0, 0.0], dtype=float)

        return {
            "L1": np.array([L1x, 0.0, 0.0], dtype=float),
            "L2": np.array([L2x, 0.0, 0.0], dtype=float),
            "L3": np.array([L3x, 0.0, 0.0], dtype=float),
            "L4": L4,
            "L5": L5,
        }

    def propagate_rk4(
        self,
        s0: Array,
        t0: float,
        tf: float,
        dt: float,
        *,
        store: bool = True
    ):
        """
        Fixed-step RK4 integrator.

        Inputs:
          s0: array-like shape (6,)
          t0, tf: floats
          dt: positive float step size
          store: if True returns (times, traj) where
                 times shape (N,), traj shape (N,6)
                 else returns final state shape (6,)

        Note: This is not symplectic; for long-term energy/Jacobi behavior,
        consider higher-order or adaptive schemes later.
        """
        s = np.asarray(s0, dtype=float).reshape(6)
        if dt <= 0.0:
            raise ValueError("dt must be > 0")

        direction = 1.0 if tf >= t0 else -1.0
        h = direction * dt

        t = float(t0)

        if not store:
            while (t - tf) * direction < 0.0:
                hh = h
                if (t + hh - tf) * direction > 0.0:
                    hh = tf - t
                s = _rk4_step_np(self.eom, t, s, hh)
                t += hh
            return s

        times = [t]
        states = [s.copy()]

        while (t - tf) * direction < 0.0:
            hh = h
            if (t + hh - tf) * direction > 0.0:
                hh = tf - t
            s = _rk4_step_np(self.eom, t, s, hh)
            t += hh
            times.append(t)
            states.append(s.copy())

        return np.asarray(times, dtype=float), np.vstack(states)


def _rk4_step_np(f: Callable[[float, Array], Array], t: float, y: Array, h: float) -> Array:
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h,       y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

