from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
DEFAULT_JPL_CACHE_DIR = REPO_ROOT / "data" / "cache" / "jpl_periodic_orbits"


def ensure_src_on_path() -> None:
    src = str(SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)


def repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def add_periodic_orbit_query_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--system", default="earth-moon")
    parser.add_argument("--family", default="halo")
    parser.add_argument("--libr", type=int, action="append", default=None, choices=[1, 2, 3])
    parser.add_argument("--branch", action="append", default=None, choices=["N", "S"])
    parser.add_argument("--period-min-days", type=float, default=5.0)
    parser.add_argument("--period-max-days", type=float, default=8.0)
    parser.add_argument("--target-period-days", type=float, default=6.56)
    parser.add_argument("--stability-max", type=float, default=None)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_JPL_CACHE_DIR)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--no-cache", action="store_true")


def selected_libration_points(args: argparse.Namespace, default: Sequence[int] = (2,)) -> list[int]:
    return list(args.libr) if args.libr is not None else list(default)


def selected_branches(args: argparse.Namespace, default: Sequence[str]) -> list[str]:
    return list(args.branch) if args.branch is not None else list(default)


def kernel_paths(kernel_args: Sequence[str | Path]) -> list[Path]:
    return [repo_path(kernel) for kernel in kernel_args]


def write_state_history_csv(path: str | Path, times_s, states) -> Path:
    outpath = repo_path(path)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"])
        for t_s, state in zip(times_s, states):
            writer.writerow([float(t_s), *[float(v) for v in state]])
    return outpath
