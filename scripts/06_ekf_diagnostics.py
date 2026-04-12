
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from _common import ensure_src_on_path

ensure_src_on_path()

import numpy as np

from diagnostics.config import (
    CaseConfig,
    FaultInjectionConfig,
    GatingConfig,
    NoiseConfig,
    OutputConfig,
)
from diagnostics.hypotheses import HypothesisConfig, run_all_hypotheses, summarize_hypotheses
from diagnostics.jacobians import check_bearing_measurement_jacobian
from diagnostics.measurement_checks import run_measurement_check_suite
from diagnostics.plots import save_all_plots
from diagnostics.runner import run_case
from diagnostics.stm_checks import compare_stm_to_finite_difference
from diagnostics.types import HypothesisResult, RunResult


Array = np.ndarray

_X0_NOM = np.array([0.8359, 0.0, 0.0, 0.0, 0.05, 0.0], dtype=float)

_Q_ACC_FLOOR = 1e-14



def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj



def make_case_config(
    *,
    camera_mode: str,
    gate_probability: float,
    sigma_px: float = 1.5,
    q_acc: float = _Q_ACC_FLOOR,
    dropout_prob: float = 0.0,
    outlier_prob: float = 0.0,
    outlier_sigma_scale: float = 10.0,
    measurement_delay_steps: int = 0,
    seed: int = 7,
) -> CaseConfig:
    return CaseConfig(
        mu=0.0121505856,
        t0=0.0,
        tf=6.0,
        dt_meas=0.02,
        seed=seed,
        x0_nom=_X0_NOM.copy(),
        dx0=np.array([1e-4, -1e-4, 0.0, 0.0, 0.0, 0.0], dtype=float),
        est_err=np.array([1e-4,  1e-4, 0.0, 0.0, 0.0, 0.0], dtype=float),
        camera_mode=camera_mode,
        noise=NoiseConfig(
            sigma_px=sigma_px,
            q_acc=q_acc,
            p0_diag=(1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6),
        ),
        gating=GatingConfig(
            enabled=True,
            probability=gate_probability,
            measurement_dim=2,
            reject_on_nan=True,
            preset="baseline" if np.isclose(gate_probability, 0.9973) else "loose_debug",
        ),
        faults=FaultInjectionConfig(
            dropout_prob=dropout_prob,
            outlier_prob=outlier_prob,
            outlier_sigma_scale=outlier_sigma_scale,
            measurement_delay_steps=measurement_delay_steps,
        ),
    )


def build_case_suite() -> dict[str, CaseConfig]:
    return {
        "fixed_baseline": make_case_config(
            camera_mode="fixed",
            gate_probability=0.9973,
        ),
        "truth_tracking_baseline": make_case_config(
            camera_mode="truth_tracking",
            gate_probability=0.9973,
        ),
        "estimate_tracking_baseline": make_case_config(
            camera_mode="estimate_tracking",
            gate_probability=0.9973,
        ),
        "estimate_tracking_loose_debug": make_case_config(
            camera_mode="estimate_tracking",
            gate_probability=0.95,
        ),
        "estimate_tracking_dropout": make_case_config(
            camera_mode="estimate_tracking",
            gate_probability=0.9973,
            dropout_prob=0.05,
        ),
        "estimate_tracking_outliers": make_case_config(
            camera_mode="estimate_tracking",
            gate_probability=0.9973,
            outlier_prob=0.03,
            outlier_sigma_scale=12.0,
        ),
        "estimate_tracking_delay": make_case_config(
            camera_mode="estimate_tracking",
            gate_probability=0.9973,
            measurement_delay_steps=1,
        ),
        "estimate_tracking_debug_open_gate": CaseConfig(
            mu=0.0121505856,
            t0=0.0,
            tf=6.0,
            dt_meas=0.02,
            seed=7,
            x0_nom=_X0_NOM.copy(),
            dx0=np.array([1e-4, -1e-4, 0.0, 0.0, 0.0, 0.0], dtype=float),
            est_err=np.array([1e-4,  1e-4, 0.0, 0.0, 0.0, 0.0], dtype=float),
            camera_mode="estimate_tracking",
            noise=NoiseConfig(
                sigma_px=3.0,
                q_acc=_Q_ACC_FLOOR,
                p0_diag=(1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4),
            ),
            gating=GatingConfig(
                enabled=False,
                probability=0.95,
                measurement_dim=2,
                reject_on_nan=True,
                preset="loose_debug",
            ),
            faults=FaultInjectionConfig(),
        ),
    }



def run_single_case_bundle(
    name: str,
    cfg: CaseConfig,
    out_root: Path,
) -> dict[str, Any]:
    case_dir = out_root / name
    plots_dir = case_dir / "plots"
    _ensure_dir(case_dir)
    _ensure_dir(plots_dir)

    run_result: RunResult = run_case(cfg)

    x_ref = run_result.trace.xhat_plus_hist[1].copy()
    if not np.all(np.isfinite(x_ref)):
        x_ref = run_result.trace.xhat_plus_hist[0].copy()

    r_body = np.array([1.0 - float(cfg.mu), 0.0, 0.0], dtype=float)

    measurement_suite = run_measurement_check_suite(
        mu=float(cfg.mu),
        r_sc_true=run_result.trace.x_true_hist[1, :3],
        x_hat_for_pointing=x_ref,
    )

    jacobian_result = check_bearing_measurement_jacobian(
        x=x_ref,
        r_body=r_body,
        sigma_theta=max(float(cfg.noise.sigma_px) / 400.0, 1e-12),
    )

    stm_result = compare_stm_to_finite_difference(
        mu=float(cfg.mu),
        x0=run_result.trace.x_true_hist[0],
        t0=float(cfg.t0),
        t1=min(float(cfg.t0) + 0.2, float(cfg.tf)),
        fd_eps=1e-7,
        method="central",
        atol=1e-6,
        rtol=1e-4,
        name=f"{name}_stm",
    )

    hypotheses = run_all_hypotheses(
        run=run_result,
        measurement_suite=measurement_suite,
        jacobian_results=[jacobian_result],
        stm_results=[stm_result],
        cfg=HypothesisConfig(),
    )

    saved_plots = save_all_plots(
        run_result,
        plots_dir,
        hypotheses=hypotheses,
    )

    summary_payload = {
        "case_name": name,
        "config": _jsonable(run_result.config),
        "summary": _jsonable(asdict(run_result.summary)),
        "hypotheses_summary": _jsonable(summarize_hypotheses(hypotheses)),
        "hypotheses": _jsonable([asdict(h) for h in hypotheses]),
        "measurement_checks": _jsonable([
            {
                "name": c.name,
                "passed": c.passed,
                "summary": c.summary,
                "details": c.details,
            }
            for c in measurement_suite.checks
        ]),
        "jacobian_check": _jsonable({
            "name": jacobian_result.name,
            "passed": jacobian_result.passed,
            "shape": jacobian_result.shape,
            "max_abs_error": jacobian_result.max_abs_error,
            "max_rel_error": jacobian_result.max_rel_error,
            "rms_abs_error": jacobian_result.rms_abs_error,
            "worst_index": jacobian_result.worst_index,
        }),
        "stm_check": _jsonable({
            "name": stm_result.name,
            "passed": stm_result.passed,
            "shape": stm_result.shape,
            "max_abs_error": stm_result.max_abs_error,
            "max_rel_error": stm_result.max_rel_error,
            "fro_abs_error": stm_result.fro_abs_error,
            "fro_rel_error": stm_result.fro_rel_error,
            "worst_index": stm_result.worst_index,
        }),
        "saved_plots": _jsonable(saved_plots),
    }

    summary_path = case_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    return {
        "name": name,
        "run_result": run_result,
        "hypotheses": hypotheses,
        "summary_path": summary_path,
        "saved_plots": saved_plots,
    }



def save_suite_index(
    suite_results: dict[str, dict[str, Any]],
    out_root: Path,
) -> Path:
    index = {}
    for name, bundle in suite_results.items():
        run_result: RunResult = bundle["run_result"]
        index[name] = {
            "summary": _jsonable(asdict(run_result.summary)),
            "summary_path": str(bundle["summary_path"]),
            "saved_plots": {k: str(v) for k, v in bundle["saved_plots"].items()},
            "num_failed_hypotheses": int(sum(not h.passed for h in bundle["hypotheses"])),
        }

    path = out_root / "suite_index.json"
    path.write_text(json.dumps(index, indent=2))
    return path


def print_console_summary(suite_results: dict[str, dict[str, Any]]) -> None:
    print("\n06 EKF diagnostics suite complete.")
    print("=" * 64)
    for name, bundle in suite_results.items():
        run_result: RunResult = bundle["run_result"]
        s = run_result.summary
        hyp = bundle["hypotheses"]
        failed = [h.name for h in hyp if not h.passed]

        print(f"\n[{name}]")
        print(f"  camera_mode       = {s.camera_mode}")
        print(f"  valid_rate        = {s.valid_rate:.4f}")
        print(f"  update_rate       = {s.update_rate:.4f}")
        print(f"  gate_accept_rate  = {s.gate_accept_rate:.4f}")
        print(f"  nis_mean          = {s.nis_mean:.4e}")
        print(f"  nees_minus_mean   = {s.nees_minus_mean:.4e}")
        print(f"  nees_plus_mean    = {s.nees_plus_mean:.4e}")
        print(f"  final_pos_err     = {s.final_pos_err:.4e} dimensionless CR3BP length")
        print(f"  final_vel_err     = {s.final_vel_err:.4e} dimensionless CR3BP velocity")
        print(f"  final_los_angle   = {s.final_los_angle:.4e} rad")
        print(f"  failed_hypotheses = {len(failed)}")
        if failed:
            for fname in failed[:8]:
                print(f"    ✗ {fname}")



def main() -> None:
    output = OutputConfig(root_dir=Path("results/diagnostics/06_ekf"))
    out_root = output.root_dir
    _ensure_dir(out_root)

    suite_cfg = build_case_suite()
    suite_results: dict[str, dict[str, Any]] = {}

    for name, cfg in suite_cfg.items():
        print(f"\nRunning case: {name} ...")
        suite_results[name] = run_single_case_bundle(
            name=name,
            cfg=cfg,
            out_root=out_root,
        )

    suite_index_path = save_suite_index(suite_results, out_root)
    print_console_summary(suite_results)
    print(f"\nSuite index written to: {suite_index_path}")


if __name__ == "__main__":
    main()
