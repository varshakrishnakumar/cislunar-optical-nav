from __future__ import annotations

import csv
import importlib.util
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from _common import ensure_src_on_path, repo_path

ensure_src_on_path()

import numpy as np  # noqa: E402
from visualization.style import (  # noqa: E402
    AMBER,
    BG,
    BORDER,
    CYAN,
    EARTH,
    GREEN,
    MOON,
    ORANGE,
    PANEL,
    RED,
    TEXT,
    VIOLET,
    apply_dark_theme,
    plot_xy,
    plot_xy_with_err,
)


def load_midcourse_run_case() -> Callable[..., Any]:
    ensure_src_on_path()
    target = repo_path("scripts/06_midcourse_ekf_correction.py")
    if not target.exists():
        raise FileNotFoundError(f"Could not find midcourse EKF script at: {target}")
    spec = importlib.util.spec_from_file_location("midcourse06a", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {target}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run_case"):
        raise AttributeError(f"{target} does not define run_case(...)")
    return getattr(mod, "run_case")


def write_dict_rows_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
    outpath = repo_path(path)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    with outpath.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return outpath


def safe_mean(vals: Sequence[float]) -> float:
    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def safe_std(vals: Sequence[float]) -> float:
    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if arr.size else float("nan")


def maybe_import_sampler():
    ensure_src_on_path()
    try:
        import mc.sampler as sampler

        return sampler
    except Exception:
        return None


def default_dx0_est_err() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(6), np.zeros(6)


def sample_errors(
    sampler_mod,
    *,
    base_seed: int,
    trial_id: int,
    sigma_r_inj: float,
    sigma_v_inj: float,
    sigma_r_est: float,
    sigma_v_est: float,
    planar_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rng = sampler_mod.make_trial_rng(base_seed, trial_id)
    dx0 = sampler_mod.sample_injection_error(
        rng,
        sigma_r=sigma_r_inj,
        sigma_v=sigma_v_inj,
        planar_only=planar_only,
    )
    est = sampler_mod.sample_estimation_error(
        rng,
        sigma_r=sigma_r_est,
        sigma_v=sigma_v_est,
        planar_only=planar_only,
    )
    return np.array(dx0, dtype=float), np.array(est, dtype=float)
