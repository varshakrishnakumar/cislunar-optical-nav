from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import chi2

from diagnostics.jacobians import JacobianComparison
from diagnostics.measurement_checks import MeasurementCheckSuiteResult
from diagnostics.stm_checks import STMComparison
from diagnostics.types import (
    HealthRecord,
    HypothesisResult,
    RunResult,
    RunTrace,
    UpdateRecord,
)


Array = np.ndarray


@dataclass(frozen=True)
class HypothesisConfig:
    symmetry_tol: float = 1e-9
    min_eig_tol: float = -1e-10
    max_condition_warn: float = 1e12

    nis_probability: float = 0.95
    nees_probability: float = 0.95

    gate_reject_rate_warn: float = 0.50
    min_update_rate_warn: float = 0.10

    los_angle_warn_rad: float = 1e-2
    final_pos_err_warn: float = 1e-2
    final_vel_err_warn: float = 1e-2

    jacobian_atol: float = 1e-8
    jacobian_rtol: float = 1e-5
    stm_atol: float = 1e-6
    stm_rtol: float = 1e-4


def _finite_values(x: Array) -> Array:
    x = np.asarray(x, dtype=float).reshape(-1)
    return x[np.isfinite(x)]


def _safe_mean(x: Array) -> float:
    vals = _finite_values(x)
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def _safe_frac(mask: Array) -> float:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size == 0:
        return float("nan")
    return float(np.mean(mask))


def _chi2_bounds(probability: float, dof: int) -> tuple[float, float]:
    alpha = 1.0 - float(probability)
    lo = float(chi2.ppf(alpha / 2.0, dof))
    hi = float(chi2.ppf(1.0 - alpha / 2.0, dof))
    return lo, hi


def _severity_from_pass(passed: bool) -> str:
    return "info" if passed else "failure"





def _check_health_records(
    records: Sequence[HealthRecord | None],
    *,
    name: str,
    symmetry_tol: float,
    min_eig_tol: float,
    max_condition_warn: float,
) -> list[HypothesisResult]:
    valid = [r for r in records if r is not None]
    if not valid:
        return [
            HypothesisResult(
                name=f"{name}_records_present",
                passed=False,
                severity="failure",
                summary=f"No {name} health records were available.",
                details={},
            )
        ]

    max_sym    = max(r.symmetry_error_fro for r in valid)
    min_eig    = min(r.min_eig for r in valid)
    worst_cond = (
        max((r.cond for r in valid if np.isfinite(r.cond)), default=float("inf"))
    )
    all_finite = all(r.is_finite for r in valid)
    all_spd    = all(r.is_spd or r.chol_ok for r in valid)

    sym_ok  = bool(max_sym <= symmetry_tol)
    psd_ok  = bool(min_eig >= min_eig_tol and all_spd)
    cond_ok = bool(worst_cond <= max_condition_warn)

    return [
        HypothesisResult(
            name=f"{name}_finite",
            passed=all_finite,
            severity=_severity_from_pass(all_finite),
            summary=(
                f"All {name} matrices remained finite."
                if all_finite
                else f"At least one {name} matrix contained non-finite values."
            ),
            details={"num_records": len(valid)},
        ),
        HypothesisResult(
            name=f"{name}_symmetric",
            passed=sym_ok,
            severity=_severity_from_pass(sym_ok),
            summary=(
                f"{name} symmetry error stayed within tolerance."
                if sym_ok
                else f"{name} symmetry error exceeded tolerance (max={max_sym:.3e})."
            ),
            details={
                "max_symmetry_error_fro": max_sym,
                "symmetry_tol": symmetry_tol,
            },
        ),
        HypothesisResult(
            name=f"{name}_psd",
            passed=psd_ok,
            severity=_severity_from_pass(psd_ok),
            summary=(
                f"{name} remained positive semidefinite within tolerance."
                if psd_ok
                else (
                    f"{name} violated PSD / Cholesky health checks "
                    f"(min_eig={min_eig:.3e}, tolerance={min_eig_tol:.3e})."
                )
            ),
            details={
                "min_eig": min_eig,
                "min_eig_tol": min_eig_tol,
                "all_spd_or_chol_ok": all_spd,
            },
        ),
        HypothesisResult(
            name=f"{name}_condition",
            passed=cond_ok,
            severity="warning" if not cond_ok else "info",
            summary=(
                f"{name} condition number stayed within warning threshold."
                if cond_ok
                else f"{name} became poorly conditioned (cond={worst_cond:.3e})."
            ),
            details={
                "worst_condition_number": worst_cond,
                "max_condition_warn": max_condition_warn,
            },
        ),
    ]





def check_run_health(
    run: RunResult,
    *,
    cfg: HypothesisConfig = HypothesisConfig(),
) -> list[HypothesisResult]:
    trace = run.trace
    out: list[HypothesisResult] = []

    out.extend(_check_health_records(
        trace.P_minus_health,
        name="P_minus",
        symmetry_tol=cfg.symmetry_tol,
        min_eig_tol=cfg.min_eig_tol,
        max_condition_warn=cfg.max_condition_warn,
    ))
    out.extend(_check_health_records(
        trace.P_plus_health,
        name="P_plus",
        symmetry_tol=cfg.symmetry_tol,
        min_eig_tol=cfg.min_eig_tol,
        max_condition_warn=cfg.max_condition_warn,
    ))

    s_records = [u.S_health for u in trace.updates if u.S_health is not None]
    out.extend(_check_health_records(
        s_records,
        name="S",
        symmetry_tol=cfg.symmetry_tol,
        min_eig_tol=cfg.min_eig_tol,
        max_condition_warn=cfg.max_condition_warn,
    ))

    return out


def check_nis_consistency(
    run: RunResult,
    *,
    cfg: HypothesisConfig = HypothesisConfig(),
) -> HypothesisResult:
    updates = [u for u in run.trace.updates if np.isfinite(u.nis)]
    nis = np.array([u.nis for u in updates], dtype=float)

    if nis.size == 0:
        return HypothesisResult(
            name="nis_consistency",
            passed=False,
            severity="failure",
            summary="No finite NIS values were available.",
            details={},
        )

    dof = 2
    lo, hi = _chi2_bounds(cfg.nis_probability, dof)
    mean_nis = float(np.mean(nis))
    frac_in_bounds = float(np.mean((nis >= lo) & (nis <= hi)))
    passed = bool(lo <= mean_nis <= hi)

    return HypothesisResult(
        name="nis_consistency",
        passed=passed,
        severity=_severity_from_pass(passed),
        summary=(
            f"Mean NIS ({mean_nis:.3f}) falls within the expected χ²({dof}) "
            f"confidence interval [{lo:.3f}, {hi:.3f}]."
            if passed
            else
            f"Mean NIS ({mean_nis:.3f}) falls outside the expected χ²({dof}) "
            f"confidence interval [{lo:.3f}, {hi:.3f}]."
        ),
        details={
            "num_updates": int(nis.size),
            "mean_nis": mean_nis,
            "bounds": (lo, hi),
            "fraction_in_bounds": frac_in_bounds,
            "probability": cfg.nis_probability,
            "dof": dof,
        },
    )


def check_nees_consistency(
    run: RunResult,
    *,
    which: str = "plus",
    cfg: HypothesisConfig = HypothesisConfig(),
) -> HypothesisResult:
    if which == "plus":
        vals = run.trace.nees_plus_hist
        name = "nees_plus_consistency"
    elif which == "minus":
        vals = run.trace.nees_minus_hist
        name = "nees_minus_consistency"
    else:
        raise ValueError(f"which must be 'plus' or 'minus', got {which!r}")

    nees = _finite_values(vals)
    if nees.size == 0:
        return HypothesisResult(
            name=name,
            passed=False,
            severity="failure",
            summary=f"No finite {name} values were available.",
            details={},
        )

    dof = 6
    lo, hi = _chi2_bounds(cfg.nees_probability, dof)
    mean_nees = float(np.mean(nees))
    frac_in_bounds = float(np.mean((nees >= lo) & (nees <= hi)))
    passed = bool(lo <= mean_nees <= hi)

    return HypothesisResult(
        name=name,
        passed=passed,
        severity=_severity_from_pass(passed),
        summary=(
            f"Mean {name} ({mean_nees:.3f}) falls within the expected χ²({dof}) "
            f"confidence interval [{lo:.3f}, {hi:.3f}]."
            if passed
            else
            f"Mean {name} ({mean_nees:.3f}) falls outside the expected χ²({dof}) "
            f"confidence interval [{lo:.3f}, {hi:.3f}]."
        ),
        details={
            "num_samples": int(nees.size),
            "mean_nees": mean_nees,
            "bounds": (lo, hi),
            "fraction_in_bounds": frac_in_bounds,
            "probability": cfg.nees_probability,
            "dof": dof,
        },
    )





def check_update_and_gate_rates(
    run: RunResult,
    *,
    cfg: HypothesisConfig = HypothesisConfig(),
) -> list[HypothesisResult]:
    updates = run.trace.updates
    valid  = np.array([u.valid_measurement for u in updates], dtype=bool)
    used   = np.array([u.update_used       for u in updates], dtype=bool)
    gated  = np.array(
        [False if u.gate is None else (not u.gate.accepted) for u in updates],
        dtype=bool,
    )

    valid_rate  = _safe_frac(valid[1:]) if valid.size > 1 else _safe_frac(valid)
    update_rate = _safe_frac(used[1:])  if used.size  > 1 else _safe_frac(used)

    valid_nonzero    = int(np.sum(valid))
    gate_reject_rate = (
        float(np.sum(gated & valid) / valid_nonzero)
        if valid_nonzero > 0
        else float("nan")
    )

    passed_update = bool(np.isnan(update_rate) or update_rate >= cfg.min_update_rate_warn)
    passed_gate   = bool(np.isnan(gate_reject_rate) or gate_reject_rate <= cfg.gate_reject_rate_warn)

    return [
        HypothesisResult(
            name="update_rate_health",
            passed=passed_update,
            severity="warning" if not passed_update else "info",
            summary=(
                f"Update rate ({update_rate:.1%}) stayed above the warning threshold "
                f"({cfg.min_update_rate_warn:.1%})."
                if passed_update
                else
                f"Update rate ({update_rate:.1%}) fell below the warning threshold "
                f"({cfg.min_update_rate_warn:.1%})."
            ),
            details={
                "valid_rate": valid_rate,
                "update_rate": update_rate,
                "min_update_rate_warn": cfg.min_update_rate_warn,
            },
        ),
        HypothesisResult(
            name="gate_reject_rate_health",
            passed=passed_gate,
            severity="warning" if not passed_gate else "info",
            summary=(
                f"Gate reject rate ({gate_reject_rate:.1%}) stayed below the warning "
                f"threshold ({cfg.gate_reject_rate_warn:.1%})."
                if passed_gate
                else
                f"Gate reject rate ({gate_reject_rate:.1%}) was unusually high "
                f"(threshold: {cfg.gate_reject_rate_warn:.1%})."
            ),
            details={
                "gate_reject_rate": gate_reject_rate,
                "gate_reject_rate_warn": cfg.gate_reject_rate_warn,
            },
        ),
    ]


def check_final_tracking_error(
    run: RunResult,
    *,
    cfg: HypothesisConfig = HypothesisConfig(),
) -> list[HypothesisResult]:
    s = run.summary

    pos_ok = bool(np.isfinite(s.final_pos_err) and s.final_pos_err <= cfg.final_pos_err_warn)
    vel_ok = bool(np.isfinite(s.final_vel_err) and s.final_vel_err <= cfg.final_vel_err_warn)
    los_ok = bool(np.isfinite(s.final_los_angle) and s.final_los_angle <= cfg.los_angle_warn_rad)

    return [
        HypothesisResult(
            name="final_position_error",
            passed=pos_ok,
            severity="warning" if not pos_ok else "info",
            summary=(
                f"Final position error ({s.final_pos_err:.3e} dimensionless CR3BP length) within threshold "
                f"({cfg.final_pos_err_warn:.3e} dimensionless CR3BP length)."
                if pos_ok
                else
                f"Final position error ({s.final_pos_err:.3e} dimensionless CR3BP length) exceeded threshold "
                f"({cfg.final_pos_err_warn:.3e} dimensionless CR3BP length)."
            ),
            details={"final_pos_err": s.final_pos_err, "threshold": cfg.final_pos_err_warn},
        ),
        HypothesisResult(
            name="final_velocity_error",
            passed=vel_ok,
            severity="warning" if not vel_ok else "info",
            summary=(
                f"Final velocity error ({s.final_vel_err:.3e} dimensionless CR3BP velocity) within threshold "
                f"({cfg.final_vel_err_warn:.3e} dimensionless CR3BP velocity)."
                if vel_ok
                else
                f"Final velocity error ({s.final_vel_err:.3e} dimensionless CR3BP velocity) exceeded threshold "
                f"({cfg.final_vel_err_warn:.3e} dimensionless CR3BP velocity)."
            ),
            details={"final_vel_err": s.final_vel_err, "threshold": cfg.final_vel_err_warn},
        ),
        HypothesisResult(
            name="final_los_angle",
            passed=los_ok,
            severity="warning" if not los_ok else "info",
            summary=(
                f"Final LOS angle ({s.final_los_angle:.3e} rad) within threshold "
                f"({cfg.los_angle_warn_rad:.3e} rad)."
                if los_ok
                else
                f"Final LOS angle ({s.final_los_angle:.3e} rad) exceeded threshold "
                f"({cfg.los_angle_warn_rad:.3e} rad)."
            ),
            details={"final_los_angle": s.final_los_angle, "threshold": cfg.los_angle_warn_rad},
        ),
    ]





def check_measurement_suite(
    suite: MeasurementCheckSuiteResult,
) -> list[HypothesisResult]:
    return suite.to_hypotheses()


def check_jacobian_results(
    results: Sequence[JacobianComparison],
    *,
    cfg: HypothesisConfig = HypothesisConfig(),
) -> list[HypothesisResult]:
    out: list[HypothesisResult] = []
    for r in results:
        out.append(
            HypothesisResult(
                name=f"{r.name}_jacobian_check",
                passed=r.passed,
                severity=_severity_from_pass(r.passed),
                summary=(
                    f"{r.name} Jacobian agrees with numeric differentiation within tolerance."
                    if r.passed
                    else (
                        f"{r.name} Jacobian differs from numeric differentiation: "
                        f"worst entry {r.worst_index} has abs error {r.max_abs_error:.3e}, "
                        f"rel error {r.max_rel_error:.3e}."
                    )
                ),
                details={
                    "shape":           r.shape,
                    "max_abs_error":   r.max_abs_error,
                    "max_rel_error":   r.max_rel_error,
                    "rms_abs_error":   r.rms_abs_error,
                    "worst_index":     r.worst_index,
                    "analytic_value":  r.analytic_value,
                    "numeric_value":   r.numeric_value,
                    "diff_value":      r.diff_value,
                    "atol":            cfg.jacobian_atol,
                    "rtol":            cfg.jacobian_rtol,
                },
            )
        )
    return out


def check_stm_results(
    results: Sequence[STMComparison],
    *,
    cfg: HypothesisConfig = HypothesisConfig(),
) -> list[HypothesisResult]:
    out: list[HypothesisResult] = []
    for r in results:
        out.append(
            HypothesisResult(
                name=f"{r.name}_stm_check",
                passed=r.passed,
                severity=_severity_from_pass(r.passed),
                summary=(
                    f"{r.name} STM agrees with finite-difference sensitivity within tolerance."
                    if r.passed
                    else (
                        f"{r.name} STM differs from finite-difference sensitivity: "
                        f"worst entry {r.worst_index} has abs error {r.max_abs_error:.3e}, "
                        f"rel error {r.max_rel_error:.3e}."
                    )
                ),
                details={
                    "shape":          r.shape,
                    "dt":             r.dt,
                    "fd_eps":         r.fd_eps,
                    "max_abs_error":  r.max_abs_error,
                    "max_rel_error":  r.max_rel_error,
                    "fro_abs_error":  r.fro_abs_error,
                    "fro_rel_error":  r.fro_rel_error,
                    "worst_index":    r.worst_index,
                    "analytic_value": r.analytic_value,
                    "numeric_value":  r.numeric_value,
                    "diff_value":     r.diff_value,
                    "atol":           cfg.stm_atol,
                    "rtol":           cfg.stm_rtol,
                },
            )
        )
    return out





def run_all_hypotheses(
    *,
    run: RunResult,
    measurement_suite: MeasurementCheckSuiteResult | None = None,
    jacobian_results: Sequence[JacobianComparison] | None = None,
    stm_results: Sequence[STMComparison] | None = None,
    cfg: HypothesisConfig = HypothesisConfig(),
) -> list[HypothesisResult]:
    out: list[HypothesisResult] = []

    out.extend(check_run_health(run, cfg=cfg))
    out.append(check_nis_consistency(run, cfg=cfg))
    out.append(check_nees_consistency(run, which="minus", cfg=cfg))
    out.append(check_nees_consistency(run, which="plus",  cfg=cfg))
    out.extend(check_update_and_gate_rates(run, cfg=cfg))
    out.extend(check_final_tracking_error(run, cfg=cfg))

    if measurement_suite is not None:
        out.extend(check_measurement_suite(measurement_suite))
    if jacobian_results is not None:
        out.extend(check_jacobian_results(jacobian_results, cfg=cfg))
    if stm_results is not None:
        out.extend(check_stm_results(stm_results, cfg=cfg))

    return out


def summarize_hypotheses(results: Sequence[HypothesisResult]) -> dict[str, object]:
    total  = len(results)
    passed = sum(r.passed for r in results)
    failed = total - passed

    by_severity = {
        "info":    sum(r.severity == "info"    for r in results),
        "warning": sum(r.severity == "warning" for r in results),
        "failure": sum(r.severity == "failure" for r in results),
    }

    failed_names = [r.name for r in results if not r.passed]

    return {
        "total":        total,
        "passed":       passed,
        "failed":       failed,
        "by_severity":  by_severity,
        "failed_names": failed_names,
    }
