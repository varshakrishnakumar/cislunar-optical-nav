from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from dynamics.integrators import propagate

from ..protocols import Guidance, Scenario, StateEstimate


Array = np.ndarray


@dataclass(frozen=True)
class CouplingRow:
    sigma_r: float
    sigma_v: float
    est_err_norm_r: float
    est_err_norm_v: float
    dv_perfect_mag: float
    dv_offset_mag: float
    dv_delta_mag: float
    dv_inflation_pct: float
    miss_perfect: float
    miss_offset: float
    miss_inflation: float

    def asdict(self) -> dict:
        return asdict(self)


def _truth_at_tc(scenario: Scenario, dx0: Array | None = None) -> Array:
    res = propagate(
        scenario.dynamics.eom,
        (float(scenario.t0_s), float(scenario.tc_s)),
        scenario.initial_truth(dx0=dx0),
        t_eval=np.array([scenario.t0_s, scenario.tc_s]),
        rtol=1e-11,
        atol=1e-13,
    )
    if not res.success:
        raise RuntimeError(f"Truth propagation to tc failed: {res.message}")
    return res.x[-1]


def _miss_after_burn(scenario: Scenario, x_truth_tc: Array, dv: Array) -> float:
    x = x_truth_tc.copy()
    x[3:6] += np.asarray(dv, dtype=float)
    res = propagate(
        scenario.dynamics.eom,
        (float(scenario.tc_s), float(scenario.tf_s)),
        x,
        t_eval=np.array([scenario.tc_s, scenario.tf_s]),
        rtol=1e-11,
        atol=1e-13,
    )
    if not res.success:
        raise RuntimeError(f"Post-burn propagation failed: {res.message}")
    return float(np.linalg.norm(res.x[-1, :3] - scenario.target_position()))


def navigation_to_burn(
    scenario: Scenario,
    guidance: Guidance,
    est_err: Array,
    *,
    dx0: Array | None = None,
    P_at_tc: Array | None = None,
) -> CouplingRow:
    """Single navigation-error → burn-error mapping.

    ``dx0`` is the truth dispersion at t0 (launch error) -- without it the
    nominal trajectory hits the target by construction and dv_perfect=0,
    making inflation ratios meaningless. Defaults to all-zero, in which case
    the caller is expected to interpret the absolute Δv shift, not the ratio.

    ``est_err`` is a 6-vector added to the truth state at tc to model what
    the filter would have estimated. We solve guidance from both states and
    compare burns + miss distances.
    """
    err = np.asarray(est_err, dtype=float).reshape(6)
    sigma_r = float(np.linalg.norm(err[:3]))
    sigma_v = float(np.linalg.norm(err[3:]))

    x_truth_tc = _truth_at_tc(scenario, dx0=dx0)
    x_offset_tc = x_truth_tc + err

    P = (
        np.eye(6, dtype=float) if P_at_tc is None
        else np.asarray(P_at_tc, dtype=float).reshape(6, 6)
    )

    dv_perfect, _ = guidance.solve(
        StateEstimate(t_s=float(scenario.tc_s), x=x_truth_tc, P=P), scenario,
    )
    dv_offset, _ = guidance.solve(
        StateEstimate(t_s=float(scenario.tc_s), x=x_offset_tc, P=P), scenario,
    )

    dv_perfect = np.asarray(dv_perfect, dtype=float)
    dv_offset = np.asarray(dv_offset, dtype=float)

    dv_perfect_mag = float(np.linalg.norm(dv_perfect))
    dv_offset_mag = float(np.linalg.norm(dv_offset))
    dv_delta_mag = float(np.linalg.norm(dv_offset - dv_perfect))
    dv_inflation_pct = (
        float("nan") if dv_perfect_mag == 0.0
        else dv_offset_mag / dv_perfect_mag - 1.0
    )

    miss_perfect = _miss_after_burn(scenario, x_truth_tc, dv_perfect)
    miss_offset = _miss_after_burn(scenario, x_truth_tc, dv_offset)
    miss_inflation = (
        float("nan") if miss_perfect == 0.0
        else miss_offset / miss_perfect - 1.0
    )

    return CouplingRow(
        sigma_r=sigma_r,
        sigma_v=sigma_v,
        est_err_norm_r=sigma_r,
        est_err_norm_v=sigma_v,
        dv_perfect_mag=dv_perfect_mag,
        dv_offset_mag=dv_offset_mag,
        dv_delta_mag=dv_delta_mag,
        dv_inflation_pct=dv_inflation_pct,
        miss_perfect=miss_perfect,
        miss_offset=miss_offset,
        miss_inflation=miss_inflation,
    )


def coupling_grid_random(
    scenario: Scenario,
    guidance: Guidance,
    *,
    sigma_r_grid: Iterable[float],
    sigma_v_grid: Iterable[float],
    n_samples: int = 50,
    base_seed: int = 7,
    planar_only: bool = False,
    dx0: Array | None = None,
    sigma_r_inj: float = 1e-4,
    sigma_v_inj: float = 0.0,
) -> list[CouplingRow]:
    """Sample n random est_err vectors per (sigma_r, sigma_v) pair.

    A single launch-dispersion ``dx0`` is sampled once (using sigma_*_inj) and
    held fixed across every grid cell so all rows share the same underlying
    truth trajectory; only the navigation error varies. Pass ``dx0`` directly
    to override the sampling.

    Output is one row per *sample*; aggregate downstream with cisopt.sweeps.query
    to get mean/std per (sigma_r, sigma_v) cell.
    """
    rng_master = np.random.default_rng(int(base_seed))

    if dx0 is None:
        dx0_arr = np.zeros(6, dtype=float)
        dx0_arr[:3] = rng_master.normal(0.0, float(sigma_r_inj), size=3)
        dx0_arr[3:] = rng_master.normal(0.0, float(sigma_v_inj), size=3)
        if planar_only:
            dx0_arr[2] = 0.0
            dx0_arr[5] = 0.0
    else:
        dx0_arr = np.asarray(dx0, dtype=float).reshape(6)

    rows: list[CouplingRow] = []

    for sigma_r in sigma_r_grid:
        for sigma_v in sigma_v_grid:
            rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
            for _ in range(int(n_samples)):
                err = np.zeros(6, dtype=float)
                err[:3] = rng.normal(0.0, float(sigma_r), size=3)
                err[3:] = rng.normal(0.0, float(sigma_v), size=3)
                if planar_only:
                    err[2] = 0.0
                    err[5] = 0.0
                row = navigation_to_burn(scenario, guidance, err, dx0=dx0_arr)
                rows.append(
                    CouplingRow(
                        sigma_r=float(sigma_r),
                        sigma_v=float(sigma_v),
                        est_err_norm_r=row.est_err_norm_r,
                        est_err_norm_v=row.est_err_norm_v,
                        dv_perfect_mag=row.dv_perfect_mag,
                        dv_offset_mag=row.dv_offset_mag,
                        dv_delta_mag=row.dv_delta_mag,
                        dv_inflation_pct=row.dv_inflation_pct,
                        miss_perfect=row.miss_perfect,
                        miss_offset=row.miss_offset,
                        miss_inflation=row.miss_inflation,
                    )
                )
    return rows


def coupling_grid_structured(
    scenario: Scenario,
    guidance: Guidance,
    *,
    err_directions: Iterable[Array],
    err_magnitudes: Iterable[float],
    dx0: Array | None = None,
) -> list[CouplingRow]:
    """Walk along fixed error *directions* at given magnitudes.

    Useful when you want to know "which axis of estimation error matters
    most" -- pair this with the observability module's weak_directions to
    probe the navigation-error to burn-error coupling along low-observability
    axes specifically.
    """
    rows: list[CouplingRow] = []
    for direction in err_directions:
        d = np.asarray(direction, dtype=float).reshape(6)
        n = float(np.linalg.norm(d))
        if n <= 0.0:
            raise ValueError("err_directions must be non-zero 6-vectors")
        d_unit = d / n
        for mag in err_magnitudes:
            rows.append(
                navigation_to_burn(scenario, guidance, d_unit * float(mag), dx0=dx0)
            )
    return rows
