#!/usr/bin/env python3
"""Run a Monte Carlo sweep over a base config.

Usage:
    python run_mc.py configs/examples/halo_l1_mc.yaml
    python run_mc.py configs/examples/halo_l1_baseline.json --n-trials 200 --workers 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _ensure_src_on_path() -> None:
    src = Path(__file__).resolve().parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a cisopt Monte Carlo sweep.")
    p.add_argument("config", type=Path, help="Path to base experiment config (.json/.yaml/.toml)")
    p.add_argument("--out", type=Path, default=None, help="Output directory")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--base-seed", type=int, default=7)
    p.add_argument("--workers", type=int, default=1, help="Worker threads (-1 = all cores)")
    p.add_argument("--on-error", choices=("warn", "raise", "skip"), default="warn")
    return p.parse_args()


def main() -> int:
    _ensure_src_on_path()
    from cisopt.config import load_config
    from cisopt.sweeps import MCSweepCfg, query, run_mc

    args = _parse_args()

    cfg = load_config(args.config)
    sweep = MCSweepCfg(
        base_cfg=cfg,
        n_trials=int(args.n_trials),
        base_seed=int(args.base_seed),
        n_workers=int(args.workers),
    )
    out_dir = args.out if args.out is not None else Path(cfg.output.out_dir) / "mc" / cfg.name

    t0 = time.perf_counter()
    result = run_mc(sweep, out_dir=out_dir, on_trial_error=args.on_error)
    elapsed = time.perf_counter() - t0

    metrics_to_print = [
        "miss_ekf", "miss_perfect", "miss_uncorrected",
        "dv_ekf_mag", "dv_inflation_pct",
        "nis_mean", "nees_mean", "valid_rate",
    ]
    print(f"== {cfg.name}: {result.n_trials} trials ({result.n_failed} failed) in {elapsed:.1f}s")
    print(f"   run_id={result.run_id}  out={result.out_dir}")
    for m in metrics_to_print:
        s = query.summarize(result.rows, m)
        print(f"   {m:>20s}  mean={s['mean']:.4e}  std={s['std']:.4e}  n={s['n']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
