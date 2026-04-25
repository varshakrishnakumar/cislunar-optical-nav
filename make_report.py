#!/usr/bin/env python3
"""Generate a paper-ready figure set from the cisopt artifact tree.

Walks an artifact root (from reproduce_paper.py) and writes figures into
the given plots directory using the new viz API. Picks up MC, ablation,
estimator zoo, coupling and observability artifacts automatically.

Usage:
    python make_report.py reports/onboard/data --plots-dir reports/onboard/figures
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
    p = argparse.ArgumentParser(description="Build cisopt figures from artifacts.")
    p.add_argument("data_dir", type=Path)
    p.add_argument("--plots-dir", type=Path, required=True)
    p.add_argument("--theme", choices=("paper", "dark"), default="paper")
    return p.parse_args()


def main() -> int:
    _ensure_src_on_path()
    from cisopt.viz import (
        apply_dark_theme,
        apply_paper_theme,
        build_report,
    )

    args = _parse_args()
    if args.theme == "dark":
        apply_dark_theme()
    else:
        apply_paper_theme()

    data_root = args.data_dir
    plots_root = args.plots_dir
    plots_root.mkdir(parents=True, exist_ok=True)

    def _maybe(path: Path) -> Path | None:
        return path if path.exists() else None

    # Single-trial observability sidecar (saved by reproduce_paper.py
    # and by run_coupling.py --with-observability).
    obs_npz = (
        _maybe(data_root / "halo_l1_baseline_single" / "observability.npz")
        or _maybe(data_root / "coupling" / "observability.npz")
        or _maybe(data_root / "halo_l1_coupling" / "observability.npz")
    )

    # Estimator zoo: support both the new layout (estimator_zoo/{cr3bp,spice}/<est>/)
    # and the legacy flat layout (estimator_zoo/<est>/).
    est_dir = data_root / "estimator_zoo"
    estimator_paths: dict[str, Path] = {}
    estimator_paths_spice: dict[str, Path] = {}
    if est_dir.exists():
        for fidelity, target in (("cr3bp", estimator_paths),
                                 ("spice", estimator_paths_spice)):
            sub = est_dir / fidelity
            if sub.exists():
                for est in sorted(sub.iterdir()):
                    tp = est / "trials.parquet"
                    if tp.exists():
                        target[est.name] = tp
        # Legacy flat layout fallback (no fidelity grouping).
        if not estimator_paths and not estimator_paths_spice:
            for est in sorted(est_dir.iterdir()):
                tp = est / "trials.parquet"
                if tp.exists():
                    estimator_paths[est.name] = tp

    paths = build_report(
        mc_parquet            = _maybe(data_root / "halo_l1_mc" / "trials.parquet"),
        ablation_parquet      = _maybe(data_root / "halo_l1_ablation" / "trials.parquet"),
        estimator_paths       = estimator_paths or None,
        estimator_paths_spice = estimator_paths_spice or None,
        coupling_parquet      = _maybe(data_root / "halo_l1_coupling" / "coupling_random.parquet"),
        observability_npz     = obs_npz,
        out_dir               = plots_root,
    )

    print(f"== wrote {len(paths)} figures to {plots_root}")
    for p in paths:
        print(f"   {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
