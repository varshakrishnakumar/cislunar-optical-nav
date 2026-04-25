#!/usr/bin/env python3
"""Single-config experiment entrypoint for the cisopt framework.

Usage:
    python run_experiment.py configs/examples/halo_l1_baseline.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    src = Path(__file__).resolve().parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a single cisopt experiment from a config file.")
    p.add_argument("config", type=Path, help="Path to .json / .yaml / .toml config")
    p.add_argument("--out", type=Path, default=None, help="Override output directory")
    p.add_argument("--quiet", action="store_true", help="Suppress per-metric printout")
    return p.parse_args()


def main() -> int:
    _ensure_src_on_path()
    from cisopt.config import load_config
    from cisopt.results import save_artifact
    from cisopt.runner import run_trial

    args = _parse_args()

    cfg = load_config(args.config)
    artifact = run_trial(cfg)

    out_dir = args.out if args.out is not None else Path(cfg.output.out_dir) / cfg.name
    save_path = save_artifact(artifact, out_dir)

    if not args.quiet:
        m = artifact.metrics
        print(f"== {cfg.name}  [hash {artifact.config_hash[:12]}]")
        print(f"  miss_uncorrected = {m.miss_uncorrected:.4e}")
        print(f"  miss_perfect     = {m.miss_perfect:.4e}")
        print(f"  miss_ekf         = {m.miss_ekf:.4e}")
        print(f"  |dv| perfect     = {m.dv_perfect_mag:.4e}")
        print(f"  |dv| EKF         = {m.dv_ekf_mag:.4e}")
        print(f"  inflation        = {m.dv_inflation_pct*100:+.2f}%")
        print(f"  NIS mean         = {m.nis_mean:.3f}    (target ~2 for 2-D bearing)")
        print(f"  NEES mean        = {m.nees_mean:.3f}    (target ~6 for 6-D state)")
        print(f"  valid_rate       = {m.valid_rate:.3f}")
        print(f"  artifact         = {save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
