#!/usr/bin/env python3
"""Re-run the canonical cisopt experiment set used in the paper / report.

Goal of this script: a single command that regenerates every Parquet table
the analysis depends on, with deterministic seeds, so peer reviewers can
verify each claim by re-executing this and diffing results.

Usage:
    python reproduce_paper.py                  # run full set
    python reproduce_paper.py --quick          # 8 trials/combo for smoke
    python reproduce_paper.py --include mc     # subset by name
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def _ensure_src_on_path() -> None:
    src = Path(__file__).resolve().parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


@dataclass
class ReproStep:
    name: str
    description: str
    runner: Callable[..., Any]
    out_subdir: str
    quick_kwargs: dict[str, Any] = field(default_factory=dict)
    full_kwargs: dict[str, Any] = field(default_factory=dict)


def _load(path: str):
    from cisopt.config import load_config
    return load_config(path)


def _step_baseline_single(out_root: Path, *, quick: bool, **_: Any) -> dict:
    import numpy as np
    from cisopt.observability import compute_gramian
    from cisopt.results import save_artifact
    from cisopt.runner import run_trial
    cfg = _load("configs/examples/halo_l1_baseline.yaml")
    art = run_trial(cfg, accumulate_gramian=True)
    out = out_root / "halo_l1_baseline_single"
    save_artifact(art, out)
    # Stand-alone observability sidecar so the viz layer can pick it up.
    if "W_obs_final" in art.timeseries:
        g = compute_gramian(art.timeseries["W_obs_final"])
        np.savez(
            out / "observability.npz",
            W=g.W,
            eigvals=g.eigvals,
            weak_directions=g.weak_directions,
        )
    return {"miss_ekf": art.metrics.miss_ekf, "out": str(out)}


def _step_mc(out_root: Path, *, quick: bool, n_trials: int, **_: Any) -> dict:
    from cisopt.sweeps import MCSweepCfg, run_mc
    cfg = _load("configs/examples/halo_l1_baseline.yaml")
    sweep = MCSweepCfg(base_cfg=cfg, n_trials=int(n_trials), n_workers=4)
    res = run_mc(sweep, out_dir=out_root / "halo_l1_mc")
    return {"n_trials": res.n_trials, "out": str(res.out_dir)}


def _step_ablation(out_root: Path, *, quick: bool, n_trials_per_combo: int, **_: Any) -> dict:
    import yaml
    from cisopt.ablation import AblationCfg, run_ablation
    from cisopt.config import from_dict
    doc = yaml.safe_load(open("configs/examples/halo_l1_ablation.yaml").read())
    base_cfg = from_dict(doc["base"])
    abl = AblationCfg(
        base_cfg=base_cfg,
        axes={k: list(v) for k, v in doc["axes"].items()},
        n_trials_per_combo=int(n_trials_per_combo),
        base_seed=int(doc.get("base_seed", 7)),
        n_workers=4,
    )
    res = run_ablation(abl, out_dir=out_root / "halo_l1_ablation")
    return {"n_trials": res.n_trials, "n_combos": len(res.combos), "out": str(res.out_dir)}


def _step_coupling(out_root: Path, *, quick: bool, n_samples: int, **_: Any) -> dict:
    from cisopt.config import load_config
    from cisopt.coupling import coupling_grid_random
    from cisopt.guidance import build_guidance
    from cisopt.scenarios import build_scenario
    from dataclasses import asdict
    import pyarrow as pa
    import pyarrow.parquet as pq

    cfg = load_config("configs/examples/halo_l1_baseline.yaml")
    scenario = build_scenario(cfg.scenario)
    guidance = build_guidance(cfg.guidance, scenario)
    rows = coupling_grid_random(
        scenario, guidance,
        sigma_r_grid=[1e-5, 1e-4, 1e-3, 1e-2],
        sigma_v_grid=[0.0, 1e-5],
        n_samples=int(n_samples),
        base_seed=7,
        planar_only=True,
        sigma_r_inj=1e-4,
    )
    out = out_root / "halo_l1_coupling"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([asdict(r) for r in rows]),
        out / "coupling_random.parquet",
        compression="zstd",
    )
    return {"n_rows": len(rows), "out": str(out)}


def _step_estimator_zoo(out_root: Path, *, quick: bool, n_trials: int, **_: Any) -> dict:
    from cisopt.config import load_config, patch_cfg
    from cisopt.sweeps import MCSweepCfg, run_mc

    bundles = [
        ("ekf",  {"q_acc": 1e-9, "rtol": 1e-10, "atol": 1e-12,
                  "gating_enabled": False}),
        ("iekf", {"q_acc": 1e-9, "rtol": 1e-10, "atol": 1e-12,
                  "max_iterations": 3, "gating_enabled": False}),
        ("ukf",  {"q_acc": 1e-9, "rtol": 1e-10, "atol": 1e-12,
                  "alpha": 1e-3, "beta": 2.0, "kappa": 0.0}),
    ]

    summaries: dict[str, dict] = {}

    for fidelity, base_yaml, q_default in (
        ("cr3bp", "configs/examples/halo_l1_baseline.yaml", None),
        ("spice", "configs/examples/halo_l1_spice_baseline.yaml", 1.0e-12),
    ):
        base = load_config(base_yaml)
        out_fidelity = out_root / "estimator_zoo" / fidelity
        # SPICE's furnsh / kernel-pool is not thread-safe -- multiple worker
        # threads loading kernels concurrently trips SPICE(INVALIDDIVISOR).
        # Force single-threaded for SPICE; CR3BP can stay parallel.
        n_workers = 1 if fidelity == "spice" else 4
        for est_name, params in bundles:
            est_params = dict(params)
            # SPICE units want a smaller q_acc; override unless caller specified.
            if q_default is not None:
                est_params["q_acc"] = q_default
            cfg = patch_cfg(base, {
                "name": f"halo_l1_{fidelity}_{est_name}",
                "estimator.name": est_name,
                "estimator.params": est_params,
            })
            sweep = MCSweepCfg(base_cfg=cfg, n_trials=int(n_trials), n_workers=n_workers)
            res = run_mc(sweep, out_dir=out_fidelity / est_name)
            summaries[f"{fidelity}_{est_name}"] = {
                "n": res.n_trials, "n_failed": res.n_failed,
                "out": str(res.out_dir),
            }
    return {"summaries": summaries, "out": str(out_root / "estimator_zoo")}


_STEPS: list[ReproStep] = [
    ReproStep("baseline_single", "Single trial w/ observability Gramian",
              _step_baseline_single, "single",
              quick_kwargs={}, full_kwargs={}),
    ReproStep("mc", "Halo-L1 baseline MC sweep",
              _step_mc, "mc",
              quick_kwargs={"n_trials": 16}, full_kwargs={"n_trials": 200}),
    ReproStep("ablation", "Pointing × q_acc × σ_px ablation",
              _step_ablation, "ablation",
              quick_kwargs={"n_trials_per_combo": 4},
              full_kwargs={"n_trials_per_combo": 32}),
    ReproStep("coupling", "Navigation→burn coupling map",
              _step_coupling, "coupling",
              quick_kwargs={"n_samples": 8}, full_kwargs={"n_samples": 64}),
    ReproStep("estimator_zoo", "EKF / IEKF / UKF MC comparison (CR3BP + SPICE)",
              _step_estimator_zoo, "estimator_zoo",
              quick_kwargs={"n_trials": 32}, full_kwargs={"n_trials": 96}),
]


def main() -> int:
    _ensure_src_on_path()

    parser = argparse.ArgumentParser(description="Reproduce cisopt paper experiments.")
    parser.add_argument("--out", type=Path, default=Path("results/cisopt/paper"))
    parser.add_argument("--quick", action="store_true",
                        help="Run smoke versions (small N) instead of full sweeps.")
    parser.add_argument("--include", nargs="+", default=None,
                        help="Subset of step names to run.")
    args = parser.parse_args()

    out_root = args.out.expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    selected = (
        _STEPS if args.include is None
        else [s for s in _STEPS if s.name in args.include]
    )
    if not selected:
        names = [s.name for s in _STEPS]
        raise SystemExit(f"No steps matched --include. Available: {names}")

    overall_t0 = time.perf_counter()
    for step in selected:
        kwargs = step.quick_kwargs if args.quick else step.full_kwargs
        print(f"\n>> {step.name}: {step.description}")
        t0 = time.perf_counter()
        res = step.runner(out_root, quick=args.quick, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"   done in {elapsed:.1f}s -- {res}")

    total = time.perf_counter() - overall_t0
    print(f"\n== reproduction complete in {total:.1f}s, results under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
