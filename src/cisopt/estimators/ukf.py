"""Unscented Kalman filter (Wan/van der Merwe sigma-point UKF) for cisopt.

Same Scenario-driven dynamics and bearing-payload measurement model as the
IEKF, swapped onto sigma points instead of an STM-linearised propagation.
Used for item 7 (estimator zoo) so config ablations can compare EKF / IEKF /
UKF on identical scenario + sensor wiring.

Notes:
  - The UKF predict step does NOT return ``Phi_step`` -- there is no single
    STM. Observability accumulation in the trial runner relies on Phi_step,
    so ``run_trial(cfg, accumulate_gramian=True)`` will raise when used with
    a UKF. Use the IEKF/EKF estimators when you need the Gramian.
  - Process noise is the same continuous white-acceleration model as the EKF
    (``nav.ekf.Qd_white_accel``) so q_acc is comparable across filters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dynamics.integrators import propagate
from nav.ekf import Qd_white_accel
from nav.measurements.bearing import los_unit, tangent_basis

from ..protocols import Measurement, Scenario, StateEstimate


Array = np.ndarray


def _sigma_weights(n: int, alpha: float, beta: float, kappa: float) -> tuple[float, Array, Array]:
    lam = alpha * alpha * (n + kappa) - n
    c = n + lam
    Wm = np.full(2 * n + 1, 1.0 / (2.0 * c), dtype=float)
    Wc = np.full(2 * n + 1, 1.0 / (2.0 * c), dtype=float)
    Wm[0] = lam / c
    Wc[0] = lam / c + (1.0 - alpha * alpha + beta)
    return c, Wm, Wc


def _sigma_points(x: Array, P: Array, c: float) -> Array:
    n = x.size
    P_sym = 0.5 * (P + P.T)
    eigvals, eigvecs = np.linalg.eigh(P_sym)
    eigvals = np.maximum(eigvals, 0.0)
    sqrt_P = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
    L = np.linalg.cholesky(c * (sqrt_P @ sqrt_P) + 1e-18 * np.eye(n))
    sigmas = np.zeros((2 * n + 1, n), dtype=float)
    sigmas[0] = x
    for i in range(n):
        sigmas[1 + i] = x + L[:, i]
        sigmas[1 + n + i] = x - L[:, i]
    return sigmas


def _batched_eom(eom_fn, n: int, n_sigmas: int):
    def fn(t: float, y_flat: Array) -> Array:
        out = np.empty_like(y_flat)
        for i in range(n_sigmas):
            out[i * n:(i + 1) * n] = eom_fn(t, y_flat[i * n:(i + 1) * n])
        return out
    return fn


@dataclass
class UKFEstimator:
    scenario: Scenario
    q_acc: float = 1e-9
    alpha: float = 1.0e-3
    beta: float = 2.0
    kappa: float = 0.0
    rtol: float = 1.0e-10
    atol: float = 1.0e-12
    name: str = "ukf"

    def _propagate_sigmas(self, sigmas: Array, t0: float, t1: float) -> Array:
        n_sigmas, n = sigmas.shape
        if t0 == t1:
            return sigmas.copy()
        y0 = sigmas.reshape(-1)
        eom_fn = _batched_eom(self.scenario.dynamics.eom, n, n_sigmas)
        res = propagate(
            eom_fn,
            (float(t0), float(t1)),
            y0,
            rtol=float(self.rtol),
            atol=float(self.atol),
        )
        if not res.success:
            raise RuntimeError(f"UKF sigma propagation failed: {res.message}")
        return res.x[-1].reshape(n_sigmas, n)

    def predict(
        self, t1: float, est: StateEstimate,
    ) -> tuple[StateEstimate, dict[str, Any]]:
        n = int(est.x.size)
        c, Wm, Wc = _sigma_weights(n, self.alpha, self.beta, self.kappa)
        sigmas = _sigma_points(est.x, est.P, c)

        propagated = self._propagate_sigmas(sigmas, est.t_s, t1)
        x_pred = (Wm[:, None] * propagated).sum(axis=0)
        diff = propagated - x_pred
        P_pred = (Wc[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0)
        P_pred = 0.5 * (P_pred + P_pred.T)
        P_pred = P_pred + Qd_white_accel(float(t1) - float(est.t_s), float(self.q_acc))

        return (
            StateEstimate(t_s=float(t1), x=x_pred, P=P_pred),
            {"sigmas": propagated, "Wm": Wm, "Wc": Wc},
        )

    def update(
        self, est: StateEstimate, meas: Measurement,
    ) -> tuple[StateEstimate, dict[str, Any]]:
        if not meas.valid or meas.payload is None:
            return est.copy(), {"accepted": False, "nis": float("nan"), "reason": "invalid"}
        if meas.payload.get("kind") != "bearing":
            raise ValueError(
                f"UKFEstimator only handles 'bearing' payloads, got {meas.payload.get('kind')!r}"
            )

        u_meas = np.asarray(meas.payload["u_global"], dtype=float).reshape(3)
        r_body = np.asarray(meas.payload["r_body"], dtype=float).reshape(3)
        sigma_theta = float(meas.payload["sigma_theta"])

        n = int(est.x.size)
        c, Wm, Wc = _sigma_weights(n, self.alpha, self.beta, self.kappa)
        sigmas = _sigma_points(est.x, est.P, c)

        # 3-D LOS for each sigma, then projection onto tangent of weighted mean LOS.
        u_sigmas = np.zeros((sigmas.shape[0], 3), dtype=float)
        for i in range(sigmas.shape[0]):
            u_i, _ = los_unit(r_body, sigmas[i, :3])
            u_sigmas[i] = u_i

        u_pred_mean = (Wm[:, None] * u_sigmas).sum(axis=0)
        u_pred_mean = u_pred_mean / np.linalg.norm(u_pred_mean)

        e1, e2 = tangent_basis(u_pred_mean)
        E = np.vstack([e1, e2])

        z_sigmas = u_sigmas @ E.T
        z_pred = (Wm[:, None] * z_sigmas).sum(axis=0)
        z_meas = E @ u_meas
        innov = z_meas - z_pred

        diff_z = z_sigmas - z_pred
        diff_x = sigmas - est.x
        R = (sigma_theta * sigma_theta) * np.eye(2, dtype=float)

        S = (Wc[:, None, None] * diff_z[:, :, None] * diff_z[:, None, :]).sum(axis=0) + R
        S = 0.5 * (S + S.T)
        Cxz = (Wc[:, None, None] * diff_x[:, :, None] * diff_z[:, None, :]).sum(axis=0)

        try:
            K = np.linalg.solve(S.T, Cxz.T).T
        except np.linalg.LinAlgError:
            K = Cxz @ np.linalg.pinv(S)

        x_upd = est.x + K @ innov
        P_upd = est.P - K @ S @ K.T
        P_upd = 0.5 * (P_upd + P_upd.T)

        try:
            nis = float(innov @ np.linalg.solve(S, innov))
        except np.linalg.LinAlgError:
            nis = float("nan")

        return (
            StateEstimate(t_s=float(est.t_s), x=x_upd, P=P_upd),
            {
                "accepted": True,
                "nis": nis,
                "innovation": innov,
            },
        )


def build_ukf(params: dict[str, Any], scenario: Scenario) -> UKFEstimator:
    return UKFEstimator(scenario=scenario, **params)
