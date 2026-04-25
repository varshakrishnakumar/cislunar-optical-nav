"""Bridge demo: equivalent MC study using the new cisopt framework.

This is the new Phase B path. Equivalent legacy script:
    python scripts/06_monte_carlo.py --study baseline --n-trials 50

The legacy script still owns publication-grade plotting (it feeds the
deck-builder) and stays the source of truth for figures until Phase D's
standardized viz API replaces those plots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    src = Path(__file__).resolve().parent.parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_PRESETS: dict[str, dict] = {
    "baseline": {"sigma_px": 1.0, "dropout_p": 0.0, "pointing": "estimate_tracking"},
    "dropout": {"sigma_px": 1.0, "dropout_p": 0.20, "pointing": "estimate_tracking"},
    "no_tracking": {"sigma_px": 1.0, "dropout_p": 0.0, "pointing": "fixed"},
    "high_noise": {"sigma_px": 5.0, "dropout_p": 0.0, "pointing": "estimate_tracking"},
}


def main() -> int:
    _ensure_src_on_path()
    from cisopt.config import load_config, patch_cfg
    from cisopt.sweeps import MCSweepCfg, query, run_mc

    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=tuple(_PRESETS), default="baseline")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/examples/halo_l1_baseline.yaml"))
    parser.add_argument("--out-root", type=Path, default=Path("results/cisopt/mc_demo"))
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    preset = _PRESETS[args.study]
    cfg = patch_cfg(base_cfg, {
        "name": f"{args.study}_mc",
        "sensor.params.sigma_px": preset["sigma_px"],
        "sensor.params.dropout_p": preset["dropout_p"],
        "sensor.params.pointing": preset["pointing"],
    })

    sweep = MCSweepCfg(base_cfg=cfg, n_trials=args.n_trials, n_workers=args.workers)
    result = run_mc(sweep, out_dir=args.out_root / args.study)

    print(f"== {args.study}: {result.n_trials} trials")
    for m in ("miss_ekf", "dv_inflation_pct", "nis_mean", "nees_mean", "valid_rate"):
        s = query.summarize(result.rows, m)
        print(f"   {m:>20s}  mean={s['mean']:.4e}  std={s['std']:.4e}")
    print(f"   parquet: {result.out_dir}/trials.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
