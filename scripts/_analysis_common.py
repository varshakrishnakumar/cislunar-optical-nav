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
    plot_xy_band,
    plot_xy_with_err,
)


_VALID_TRUTH_MODES = ("cr3bp", "spice")


def load_midcourse_run_case(truth: str = "cr3bp") -> Callable[..., Any]:
    """Return the ``run_case`` callable to use for truth simulation.

    Parameters
    ----------
    truth : {"cr3bp", "spice"}
        ``cr3bp``  — pure rotating-frame three-body model (legacy default,
                     matches the published numbers).
        ``spice``  — high-fidelity Earth/Moon/Sun point-mass dynamics seeded
                     from SPICE (DE442s). The EKF still uses CR3BP, so the
                     gap between truth and filter dynamics is the model
                     mismatch the estimator has to absorb.

    The SPICE callable is wrapped so its signature matches ``run_case`` —
    callers pass ``q_acc=...`` as before; the wrapper translates to the
    SPICE function's ``q_acc_nd=...`` keyword. Outputs from the SPICE
    branch are in km / km·s; CR3BP outputs remain dimensionless. Plot
    label and threshold updates for SPICE outputs are deferred to Phase C.
    """
    truth = (truth or "cr3bp").lower()
    if truth not in _VALID_TRUTH_MODES:
        raise ValueError(
            f"Unknown truth mode: {truth!r} (expected one of {_VALID_TRUTH_MODES})"
        )

    ensure_src_on_path()
    target = repo_path("scripts/06_midcourse_ekf_correction.py")
    if not target.exists():
        raise FileNotFoundError(f"Could not find midcourse EKF script at: {target}")
    spec = importlib.util.spec_from_file_location("midcourse06a", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {target}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if truth == "cr3bp":
        if not hasattr(mod, "run_case"):
            raise AttributeError(f"{target} does not define run_case(...)")
        return getattr(mod, "run_case")

    # truth == "spice" — verify kernels exist before promising a callable.
    if not hasattr(mod, "run_case_spice"):
        raise AttributeError(f"{target} does not define run_case_spice(...)")
    default_kernels = getattr(mod, "_DEFAULT_KERNELS", ())
    missing = [str(k) for k in default_kernels if not Path(k).exists()]
    if missing:
        raise FileNotFoundError(
            "SPICE kernels not found:\n  " + "\n  ".join(missing) +
            "\nFetch them or pass --truth=cr3bp."
        )
    run_case_spice = getattr(mod, "run_case_spice")

    def _spice_run_case(*args: Any, q_acc: float = 1e-14, **kwargs: Any) -> Any:
        # run_case takes q_acc; run_case_spice takes q_acc_nd. Same units
        # (CR3BP-ND), different name — callers don't have to care.
        return run_case_spice(*args, q_acc_nd=q_acc, **kwargs)

    _spice_run_case.__name__ = "run_case_spice"
    _spice_run_case.__qualname__ = "run_case_spice"
    return _spice_run_case


def add_truth_arg(parser: "Any") -> None:
    """Standard ``--truth=cr3bp|spice`` flag for analysis drivers."""
    parser.add_argument(
        "--truth",
        choices=list(_VALID_TRUTH_MODES),
        default="cr3bp",
        help="Truth dynamics model. cr3bp = legacy rotating-frame (default, "
             "matches published numbers). spice = high-fidelity DE442s "
             "ephemeris truth; EKF stays CR3BP, so the gap is the model "
             "mismatch the filter must absorb. spice outputs go to a "
             "_spice-suffixed sibling directory so cr3bp artifacts stay "
             "untouched.",
    )


def apply_truth_suffix(plots_dir: str | Path, truth: str) -> Path:
    """Suffix a plots-dir with ``_spice`` when truth is SPICE.

    CR3BP runs intentionally write to the original (un-suffixed) location
    so existing artifacts and report figures are not invalidated. SPICE
    runs land in a sibling directory, e.g.
    ``results/mc/baseline_live`` → ``results/mc/baseline_live_spice``.
    """
    p = Path(plots_dir)
    if truth == "spice":
        return p.with_name(p.name + "_spice") if p.name else p / "_spice"
    return p


def tag_rows_with_truth(
    rows: Sequence[dict[str, Any]], truth: str
) -> list[dict[str, Any]]:
    """Add a ``truth_mode`` column to every row so mixed CSVs can be
    grouped and filtered downstream when generating comparison plots."""
    return [dict(row, truth_mode=truth) for row in rows]


def inject_truth_column_into_csv(path: str | Path, truth: str) -> Path:
    """Re-write a CSV with a ``truth_mode`` column appended to every row.

    Used to tag CSVs written by helpers we don't own (e.g.
    ``mc.save_results_csv``). Idempotent — re-running with the same
    ``truth`` is a no-op aside from rewriting the file.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return p
    with p.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        return p
    if "truth_mode" not in fieldnames:
        fieldnames.append("truth_mode")
    for row in rows:
        row["truth_mode"] = truth
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return p


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
