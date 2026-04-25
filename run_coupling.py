#!/usr/bin/env python3
"""Run a guidance--navigation coupling sweep.

Generates a "navigation_error -> burn_error" map (item 8 of the refactor
brief) by sweeping random est_err vectors at each (sigma_r, sigma_v) cell on
top of a fixed launch-dispersion truth trajectory.

Output is a Parquet table compatible with cisopt.sweeps.query helpers. When
``--with-observability`` is set, the script also runs one filter trial to
extract the observability Gramian and walks the two weakest directions, so
you can visualise nav-axis sensitivity vs observability rank.

Usage:
    python run_coupling.py configs/examples/halo_l1_baseline.yaml \\
        --sigma-r 1e-5,1e-4,1e-3,1e-2 \\
        --n-samples 30 \\
        --out results/cisopt/coupling/halo_l1_baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def _ensure_src_on_path() -> None:
    src = Path(__file__).resolve().parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a cisopt navigation→burn coupling sweep.")
    p.add_argument("config", type=Path)
    p.add_argument("--sigma-r", type=str, default="1e-5,1e-4,1e-3,1e-2",
                   help="Comma-separated σ_r values to sweep (estimation-error scale).")
    p.add_argument("--sigma-v", type=str, default="0.0",
                   help="Comma-separated σ_v values to sweep.")
    p.add_argument("--n-samples", type=int, default=30)
    p.add_argument("--base-seed", type=int, default=7)
    p.add_argument("--sigma-r-inj", type=float, default=1e-4,
                   help="Launch dispersion σ_r — sets the truth trajectory deflection.")
    p.add_argument("--sigma-v-inj", type=float, default=0.0)
    p.add_argument("--planar-only", action="store_true")
    p.add_argument("--with-observability", action="store_true",
                   help="Also run one filter trial and probe weakest Gramian directions.")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    _ensure_src_on_path()
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from cisopt.config import load_config
    from cisopt.coupling import coupling_grid_random, coupling_grid_structured
    from cisopt.guidance import build_guidance
    from cisopt.observability import compute_gramian
    from cisopt.runner import run_trial
    from cisopt.scenarios import build_scenario

    args = _parse_args()

    cfg = load_config(args.config)
    scenario = build_scenario(cfg.scenario)
    guidance = build_guidance(cfg.guidance, scenario)

    sigma_r_grid = _parse_floats(args.sigma_r)
    sigma_v_grid = _parse_floats(args.sigma_v)

    out_dir = args.out if args.out is not None else (
        Path(cfg.output.out_dir) / "coupling" / cfg.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = coupling_grid_random(
        scenario, guidance,
        sigma_r_grid=sigma_r_grid,
        sigma_v_grid=sigma_v_grid,
        n_samples=int(args.n_samples),
        base_seed=int(args.base_seed),
        planar_only=bool(args.planar_only),
        sigma_r_inj=float(args.sigma_r_inj),
        sigma_v_inj=float(args.sigma_v_inj),
    )
    print(f"== {cfg.name}: {len(rows)} coupling rows "
          f"({len(sigma_r_grid)}×{len(sigma_v_grid)} cells × {args.n_samples} samples)")

    table = pa.Table.from_pylist([asdict(r) for r in rows])
    pq.write_table(table, out_dir / "coupling_random.parquet", compression="zstd")

    summary = {}
    for sigma_r in sigma_r_grid:
        for sigma_v in sigma_v_grid:
            cell = [r for r in rows
                    if abs(r.sigma_r - sigma_r) < 1e-15 and abs(r.sigma_v - sigma_v) < 1e-15]
            miss_arr = np.array([c.miss_offset for c in cell])
            dv_arr = np.array([c.dv_inflation_pct for c in cell])
            summary[f"{sigma_r:g}|{sigma_v:g}"] = {
                "n": len(cell),
                "miss_offset_mean": float(np.nanmean(miss_arr)),
                "miss_offset_std":  float(np.nanstd(miss_arr)),
                "dv_inflation_mean": float(np.nanmean(dv_arr)),
                "dv_inflation_std":  float(np.nanstd(dv_arr)),
            }
    (out_dir / "coupling_random_summary.json").write_text(json.dumps(summary, indent=2))

    if args.with_observability:
        artifact = run_trial(cfg, accumulate_gramian=True)
        g = compute_gramian(artifact.timeseries["W_obs_final"])
        struct = coupling_grid_structured(
            scenario, guidance,
            err_directions=[g.weak_directions[:, 0], g.weak_directions[:, 1]],
            err_magnitudes=[1e-5, 1e-4, 1e-3, 1e-2],
        )
        struct_table = pa.Table.from_pylist([asdict(r) for r in struct])
        pq.write_table(struct_table, out_dir / "coupling_weak_dirs.parquet", compression="zstd")
        np.savez(
            out_dir / "observability.npz",
            W=g.W,
            eigvals=g.eigvals,
            weak_directions=g.weak_directions,
        )
        print(f"   observability: cond_number={g.condition_number:.3e}, "
              f"smallest_eig={g.smallest_eig:.3e}")

    print(f"   out: {out_dir}")
    print(f"\n   {'sigma_r':>10s}  {'sigma_v':>10s}  {'miss_mean':>14s}  {'dv_inflation':>14s}")
    for key, s in summary.items():
        sr, sv = key.split("|")
        print(f"   {float(sr):>10.1e}  {float(sv):>10.1e}  {s['miss_offset_mean']:>14.3e}  {s['dv_inflation_mean']:>14.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
