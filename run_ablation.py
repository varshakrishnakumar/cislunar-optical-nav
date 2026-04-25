#!/usr/bin/env python3
"""Run an ablation sweep over named axes on top of a base config.

Usage:
    python run_ablation.py configs/examples/halo_l1_ablation.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _ensure_src_on_path() -> None:
    src = Path(__file__).resolve().parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a cisopt ablation sweep.")
    p.add_argument("config", type=Path, help="Path to ablation config (.json/.yaml/.toml)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--on-error", choices=("warn", "raise", "skip"), default="warn")
    return p.parse_args()


def _load_ablation_doc(path: Path) -> dict:
    suf = path.suffix.lower()
    if suf == ".json":
        return json.loads(path.read_text())
    if suf in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(path.read_text())
    if suf == ".toml":
        import tomllib
        return tomllib.loads(path.read_text())
    raise ValueError(f"Unsupported ablation config extension: {suf}")


def main() -> int:
    _ensure_src_on_path()
    from cisopt.ablation import AblationCfg, run_ablation
    from cisopt.config import from_dict
    from cisopt.sweeps import query

    args = _parse_args()

    doc = _load_ablation_doc(args.config)
    if "base" not in doc or "axes" not in doc:
        raise ValueError(
            "Ablation config must have top-level 'base' (an experiment cfg) and 'axes' (dict)"
        )

    base_cfg = from_dict(doc["base"])
    abl = AblationCfg(
        base_cfg=base_cfg,
        axes={k: list(v) for k, v in doc["axes"].items()},
        n_trials_per_combo=int(doc.get("n_trials_per_combo", 20)),
        base_seed=int(doc.get("base_seed", 7)),
        n_workers=int(args.workers),
    )

    out_dir = args.out if args.out is not None else (
        Path(base_cfg.output.out_dir) / "ablation" / base_cfg.name
    )

    t0 = time.perf_counter()
    result = run_ablation(abl, out_dir=out_dir, on_trial_error=args.on_error)
    elapsed = time.perf_counter() - t0

    print(f"== {base_cfg.name}: {len(result.combos)} combos, "
          f"{result.n_trials} trials ({result.n_failed} failed) in {elapsed:.1f}s")
    print(f"   run_id={result.run_id}  out={result.out_dir}")
    metric = "miss_ekf"
    by_combo = query.summarize_by_combo(result.rows, metric)
    print(f"\n   {metric}  by combo (mean ± std):")
    for combo_id, s in sorted(by_combo.items()):
        print(f"     {combo_id:>30s}  {s['mean']:.4e} ± {s['std']:.4e}  (n={s['n']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
