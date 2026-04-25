"""Single-trial runner: scenario + sensor + estimator + guidance -> TrialArtifact.

Mirrors the loop in scripts/06_midcourse_ekf_correction.py:run_case but goes
through the cisopt protocols, so the scenario/sensor/estimator/guidance
choices are determined by config rather than hard-coded.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from dynamics.integrators import propagate

from ..config import ExperimentCfg, config_hash, to_dict
from ..estimators import build_estimator
from ..guidance import build_guidance
from ..protocols import Estimator, Guidance, Measurement, Scenario, Sensor, StateEstimate
from ..results.artifact import TrialArtifact, TrialMetrics
from ..scenarios import build_scenario
from ..sensors import build_sensor


Array = np.ndarray


def _norm(x: Array) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=float)))


def _nearest_index(t_grid: Array, t: float) -> int:
    return int(np.argmin(np.abs(t_grid - t)))


def _sample_error(rng: np.random.Generator, sigma_r: float, sigma_v: float, planar: bool) -> Array:
    err = np.zeros(6, dtype=float)
    err[:3] = rng.normal(0.0, float(sigma_r), size=3)
    err[3:] = rng.normal(0.0, float(sigma_v), size=3)
    if planar:
        err[2] = 0.0
        err[5] = 0.0
    return err


def _propagate_truth(scenario: Scenario, x0: Array, t_grid: Array) -> Array:
    res = propagate(
        scenario.dynamics.eom,
        (float(t_grid[0]), float(t_grid[-1])),
        x0,
        t_eval=t_grid,
        rtol=1e-11,
        atol=1e-13,
    )
    if not res.success:
        raise RuntimeError(f"Truth propagation failed: {res.message}")
    return res.x


def _miss_after_burn(scenario: Scenario, x0: Array, dv: Array | None) -> float:
    x = x0.copy()
    if dv is not None:
        x[3:6] += np.asarray(dv, dtype=float)
    res = propagate(
        scenario.dynamics.eom,
        (float(scenario.tc_s), float(scenario.tf_s)),
        x,
        t_eval=np.linspace(scenario.tc_s, scenario.tf_s, 2001),
        rtol=1e-11,
        atol=1e-13,
    )
    if not res.success:
        raise RuntimeError(f"Post-burn propagation failed: {res.message}")
    return _norm(res.x[-1, :3] - scenario.target_position())


def _run_filter_loop(
    scenario: Scenario,
    sensor: Sensor,
    estimator: Estimator,
    *,
    xs_truth: Array,
    t_meas: Array,
    k_tc: int,
    est0: StateEstimate,
    rng: np.random.Generator,
    accumulate_gramian: bool = False,
) -> tuple[StateEstimate, dict[str, np.ndarray]]:
    est = est0.copy()

    nis_list: list[float] = []
    nees_list: list[float] = []
    pos_err_list: list[float] = []
    valid_arr = np.zeros(k_tc + 1, dtype=bool)

    Phi_cum = np.eye(6, dtype=float) if accumulate_gramian else None
    W_obs = np.zeros((6, 6), dtype=float) if accumulate_gramian else None
    gramian_eig_hist: list[Array] = []

    for k in range(1, k_tc + 1):
        est, pred_info = estimator.predict(float(t_meas[k]), est)
        if accumulate_gramian:
            Phi_step = pred_info.get("Phi_step")
            if Phi_step is None:
                raise RuntimeError(
                    "accumulate_gramian=True but estimator did not return "
                    "'Phi_step' from predict(). Use an estimator with an STM."
                )
            Phi_cum = np.asarray(Phi_step, dtype=float) @ Phi_cum

        x_truth_k = xs_truth[k]
        meas: Measurement = sensor.measure(
            float(t_meas[k]), x_truth_k, est.x, rng=rng,
        )

        if meas.valid:
            est, info = estimator.update(est, meas)
            if info.get("accepted", False):
                valid_arr[k] = True
                if accumulate_gramian and "H" in info:
                    H = np.asarray(info["H"], dtype=float)
                    W_obs += Phi_cum.T @ H.T @ H @ Phi_cum
            nis_list.append(float(info.get("nis", float("nan"))))
        else:
            nis_list.append(float("nan"))

        err6 = est.x - x_truth_k
        try:
            nees_val = float(err6 @ np.linalg.solve(est.P, err6))
        except np.linalg.LinAlgError:
            nees_val = float("nan")
        nees_list.append(nees_val)
        pos_err_list.append(_norm(est.x[:3] - x_truth_k[:3]))
        if accumulate_gramian:
            gramian_eig_hist.append(np.linalg.eigvalsh(W_obs).copy())

    timeseries = {
        "t_meas": np.asarray(t_meas[: k_tc + 1], dtype=float),
        "nis_hist": np.asarray(nis_list, dtype=float),
        "nees_hist": np.asarray(nees_list, dtype=float),
        "pos_err_hist": np.asarray(pos_err_list, dtype=float),
        "valid_arr": valid_arr.astype(np.int8),
    }
    if accumulate_gramian:
        timeseries["W_obs_final"] = np.asarray(W_obs, dtype=float)
        if gramian_eig_hist:
            timeseries["gramian_eig_hist"] = np.asarray(gramian_eig_hist, dtype=float)
    return est, timeseries


def run_trial(
    cfg: ExperimentCfg,
    *,
    accumulate_gramian: bool = False,
) -> TrialArtifact:
    scenario = build_scenario(cfg.scenario)
    sensor = build_sensor(cfg.sensor, scenario)
    estimator = build_estimator(cfg.estimator, scenario)
    guidance: Guidance = build_guidance(cfg.guidance, scenario)

    rng = np.random.default_rng(int(cfg.trial.seed))

    dx0 = _sample_error(
        rng, cfg.trial.sigma_r_inj, cfg.trial.sigma_v_inj, cfg.trial.planar_only,
    )
    est_err = _sample_error(
        rng, cfg.trial.sigma_r_est, cfg.trial.sigma_v_est, cfg.trial.planar_only,
    )

    x0_truth = scenario.initial_truth(dx0=dx0)
    est0 = scenario.initial_estimate(est_err=est_err)

    t_meas = np.arange(
        float(scenario.t0_s), float(scenario.tf_s) + 1e-12, float(scenario.dt_meas_s),
    )
    xs_truth = _propagate_truth(scenario, x0_truth, t_meas)
    k_tc = _nearest_index(t_meas, float(scenario.tc_s))

    est_tc, ts = _run_filter_loop(
        scenario, sensor, estimator,
        xs_truth=xs_truth, t_meas=t_meas, k_tc=k_tc, est0=est0, rng=rng,
        accumulate_gramian=accumulate_gramian,
    )

    x_truth_tc = xs_truth[k_tc]
    pos_err_tc = _norm(est_tc.x[:3] - x_truth_tc[:3])
    trace_P_pos_tc = float(np.trace(est_tc.P[:3, :3]))

    truth_state_at_tc = StateEstimate(t_s=float(t_meas[k_tc]), x=x_truth_tc, P=est_tc.P)
    dv_perfect, _ = guidance.solve(truth_state_at_tc, scenario)
    dv_ekf, _ = guidance.solve(est_tc, scenario)

    miss_uncorrected = _miss_after_burn(scenario, x_truth_tc, None)
    miss_perfect = _miss_after_burn(scenario, x_truth_tc, dv_perfect)
    miss_ekf = _miss_after_burn(scenario, x_truth_tc, dv_ekf)

    dv_perfect_mag = _norm(dv_perfect)
    dv_ekf_mag = _norm(dv_ekf)
    dv_delta_mag = _norm(dv_ekf - dv_perfect)
    dv_inflation_pct = (
        float("nan") if dv_perfect_mag == 0.0 else dv_ekf_mag / dv_perfect_mag - 1.0
    )

    nis_arr = ts["nis_hist"]
    nees_arr = ts["nees_hist"]
    valid_arr = ts["valid_arr"].astype(bool)

    nis_finite = nis_arr[np.isfinite(nis_arr)]
    nees_finite = nees_arr[np.isfinite(nees_arr)]
    nis_mean = float(np.mean(nis_finite)) if nis_finite.size else float("nan")
    nees_mean = float(np.mean(nees_finite)) if nees_finite.size else float("nan")
    valid_rate = float(np.mean(valid_arr))

    metrics = TrialMetrics(
        dv_perfect_mag=dv_perfect_mag,
        dv_ekf_mag=dv_ekf_mag,
        dv_delta_mag=dv_delta_mag,
        dv_inflation_pct=dv_inflation_pct,
        miss_uncorrected=miss_uncorrected,
        miss_perfect=miss_perfect,
        miss_ekf=miss_ekf,
        pos_err_tc=pos_err_tc,
        trace_P_pos_tc=trace_P_pos_tc,
        nis_mean=nis_mean,
        nees_mean=nees_mean,
        valid_rate=valid_rate,
    )

    artifact = TrialArtifact(
        config=to_dict(cfg),
        config_hash=config_hash(cfg),
        seed=int(cfg.trial.seed),
        metrics=metrics,
        units=getattr(scenario, "units", lambda: {})(),
        timeseries=ts if cfg.output.save_debug else {},
        notes={
            "dv_perfect": np.asarray(dv_perfect, dtype=float).tolist(),
            "dv_ekf": np.asarray(dv_ekf, dtype=float).tolist(),
            "dx0": dx0.tolist(),
            "est_err": est_err.tolist(),
        },
    )
    return artifact
