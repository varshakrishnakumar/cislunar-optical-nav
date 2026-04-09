from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


_BG     = "#080B14"
_PANEL  = "#0E1220"
_BORDER = "#1C2340"
_TEXT   = "#DCE0EC"
_DIM    = "#5A6080"
_CYAN   = "#22D3EE"
_AMBER  = "#F59E0B"
_GREEN  = "#10B981"
_RED    = "#F43F5E"
_VIOLET = "#8B5CF6"


def _apply_dark_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor":  _BG,
        "axes.facecolor":    _PANEL,
        "axes.edgecolor":    _BORDER,
        "axes.labelcolor":   _TEXT,
        "axes.titlecolor":   _TEXT,
        "text.color":        _TEXT,
        "xtick.color":       _TEXT,
        "ytick.color":       _TEXT,
        "grid.color":        _BORDER,
        "grid.alpha":        1.0,
        "grid.linestyle":    "--",
        "lines.linewidth":   2.0,
        "legend.facecolor":  _PANEL,
        "legend.edgecolor":  _BORDER,
        "legend.labelcolor": _TEXT,
        "savefig.facecolor": _BG,
        "savefig.edgecolor": _BG,
        "font.size":         11,
    })


def _repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_paths(repo_root: Path) -> None:
    src_dir = repo_root / "src"
    for p in [repo_root, src_dir]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _load_run_case(repo_root: Path) -> Callable[..., Dict[str, Any]]:
    target = repo_root / "scripts" / "06_midcourse_ekf_correction.py"
    if not target.exists():
        raise FileNotFoundError(f"Could not find 06A script at: {target}")
    spec = importlib.util.spec_from_file_location("midcourse06a", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {target}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run_case"):
        raise AttributeError("06A module does not define run_case(...)")
    return getattr(mod, "run_case")


def _plot_xy_with_err(
    x: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    outpath: Path,
    color: str = _CYAN,
    marker_color: str = _AMBER,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(_BORDER)

    ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                    color=color, alpha=0.18, label="±1σ band")
    ax.plot(x, y_mean, color=color, lw=2.0, zorder=3, label="mean")
    ax.errorbar(x, y_mean, yerr=y_std, fmt="none",
                ecolor=marker_color, capsize=4, elinewidth=1.0, zorder=4)
    ax.scatter(x, y_mean, s=50, color=marker_color, zorder=5,
               edgecolors=_BG, lw=0.5)

    ax.set_xlabel(xlabel, color=_TEXT)
    ax.set_ylabel(ylabel, color=_TEXT)
    ax.set_title(title, color=_TEXT)
    ax.legend(fontsize=9)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, facecolor=_BG)
    plt.close(fig)


def _safe_mean(vals: List[float]) -> float:
    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _safe_std(vals: List[float]) -> float:
    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if arr.size else float("nan")


def _maybe_import_sampler(repo_root: Path):
    _ensure_paths(repo_root)
    try:
        import mc.sampler as sampler
        return sampler
    except Exception:
        return None


def _default_dx0_est_err() -> Tuple[np.ndarray, np.ndarray]:
    return np.zeros(6), np.zeros(6)


def _sample_errors(
    sampler_mod,
    *,
    base_seed: int,
    trial_id: int,
    sigma_r_inj: float,
    sigma_v_inj: float,
    sigma_r_est: float,
    sigma_v_est: float,
    planar_only: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = sampler_mod.make_trial_rng(base_seed, trial_id)
    dx0 = sampler_mod.sample_injection_error(
        rng, sigma_r=sigma_r_inj, sigma_v=sigma_v_inj, planar_only=planar_only
    )
    est = sampler_mod.sample_estimation_error(
        rng, sigma_r=sigma_r_est, sigma_v=sigma_v_est, planar_only=planar_only
    )
    return np.array(dx0, dtype=float), np.array(est, dtype=float)


def _run_point(
    run_case: Callable[..., Dict[str, Any]],
    *,
    mu: float,
    t0: float,
    tf: float,
    tc: float,
    dt_meas: float,
    sigma_px: float,
    dropout_prob: float,
    fixed_camera_pointing: bool,
    n_mc: int,
    base_seed: int,
    vary_errors: bool,
    sampler_mod,
    sigma_r_inj: float,
    sigma_v_inj: float,
    sigma_r_est: float,
    sigma_v_est: float,
    planar_only: bool,
) -> Dict[str, float]:
    dv_delta_vals: List[float] = []
    miss_vals:     List[float] = []
    poserr_vals:   List[float] = []

    for j in range(n_mc):
        seed_j = base_seed + j
        if vary_errors and sampler_mod is not None:
            dx0, est_err = _sample_errors(
                sampler_mod,
                base_seed=base_seed, trial_id=j,
                sigma_r_inj=sigma_r_inj, sigma_v_inj=sigma_v_inj,
                sigma_r_est=sigma_r_est, sigma_v_est=sigma_v_est,
                planar_only=planar_only,
            )
        else:
            dx0, est_err = _default_dx0_est_err()

        out = run_case(
            mu, t0, tf, tc, dt_meas, sigma_px, dropout_prob, seed_j,
            dx0, est_err,
            fixed_camera_pointing=fixed_camera_pointing,
        )
        dv_delta_vals.append(float(out.get("dv_delta_mag", np.nan)))
        miss_vals.append(float(out.get("miss_ekf",         np.nan)))
        poserr_vals.append(float(out.get("pos_err_tc",     np.nan)))

    return {
        "dv_delta_mean":  _safe_mean(dv_delta_vals),
        "dv_delta_std":   _safe_std(dv_delta_vals),
        "miss_ekf_mean":  _safe_mean(miss_vals),
        "miss_ekf_std":   _safe_std(miss_vals),
        "pos_err_tc_mean":_safe_mean(poserr_vals),
        "pos_err_tc_std": _safe_std(poserr_vals),
    }


def main() -> None:
    _apply_dark_theme()

    p = argparse.ArgumentParser()
    p.add_argument("--plots-dir",    type=str,   default="results/plots")
    p.add_argument("--n-mc",         type=int,   default=10)
    p.add_argument("--base-seed",    type=int,   default=7)
    p.add_argument("--mu",           type=float, default=0.0121505856)
    p.add_argument("--t0",           type=float, default=0.0)
    p.add_argument("--tf",           type=float, default=6.0)
    p.add_argument("--tc",           type=float, default=2.0)
    p.add_argument("--dt-meas",      type=float, default=0.02)
    p.add_argument("--sigma-px",     type=float, default=1.5)
    p.add_argument("--dropout-prob", type=float, default=0.0)
    p.add_argument("--vary-errors",  action="store_true")
    p.add_argument("--planar-only",  action="store_true")
    p.add_argument("--sigma-r-inj",  type=float, default=1e-4)
    p.add_argument("--sigma-v-inj",  type=float, default=1e-4)
    p.add_argument("--sigma-r-est",  type=float, default=1e-4)
    p.add_argument("--sigma-v-est",  type=float, default=1e-4)

    args = p.parse_args()

    repo_root  = _repo_root_from_this_file()
    _ensure_paths(repo_root)
    run_case   = _load_run_case(repo_root)
    sampler_mod = _maybe_import_sampler(repo_root)

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    sigma_grid = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 5.0], dtype=float)
    tc_grid    = np.array([0.8, 1.2, 1.6, 2.0, 2.5, 3.0], dtype=float)

    rows: List[Dict[str, float]] = []

    _common = dict(
        mu=args.mu, t0=args.t0, tf=args.tf, dt_meas=args.dt_meas,
        n_mc=int(args.n_mc), vary_errors=bool(args.vary_errors),
        sampler_mod=sampler_mod,
        sigma_r_inj=float(args.sigma_r_inj), sigma_v_inj=float(args.sigma_v_inj),
        sigma_r_est=float(args.sigma_r_est), sigma_v_est=float(args.sigma_v_est),
        planar_only=bool(args.planar_only),
    )

    for sigma_px in sigma_grid:
        stats = _run_point(
            run_case,
            tc=args.tc, sigma_px=float(sigma_px),
            dropout_prob=float(args.dropout_prob),
            fixed_camera_pointing=False,
            base_seed=int(args.base_seed),
            **_common,
        )
        rows.append({"sweep": 0.0, "sigma_px": float(sigma_px),
                     "tc": float(args.tc), **stats})

    for tc in tc_grid:
        stats = _run_point(
            run_case,
            tc=float(tc), sigma_px=float(args.sigma_px),
            dropout_prob=float(args.dropout_prob),
            fixed_camera_pointing=False,
            base_seed=int(args.base_seed) + 1000,
            **_common,
        )
        rows.append({"sweep": 1.0, "sigma_px": float(args.sigma_px),
                     "tc": float(tc), **stats})

    for sigma_px in sigma_grid:
        stats = _run_point(
            run_case,
            tc=args.tc, sigma_px=float(sigma_px),
            dropout_prob=0.05,
            fixed_camera_pointing=True,
            base_seed=int(args.base_seed) + 2000,
            **_common,
        )
        rows.append({"sweep": 2.0, "sigma_px": float(sigma_px),
                     "tc": float(args.tc), **stats})

    csv_path = plots_dir / "06b_sensitivity_mc.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def _sel(sweep_id: float) -> List[Dict[str, float]]:
        return [r for r in rows if float(r["sweep"]) == sweep_id]

    r0 = _sel(0.0)
    x  = np.array([r["sigma_px"] for r in r0])
    _plot_xy_with_err(
        x, np.array([r["dv_delta_mean"] for r in r0]),
              np.array([r["dv_delta_std"]  for r in r0]),
        xlabel="σ_px  [px]",
        ylabel="‖Δdv‖ mean ± std  [ND]",
        title=f"Burn Error vs Pixel Noise  (mini-MC n={args.n_mc})",
        outpath=plots_dir / "06b_dv_error_vs_sigma.png",
        color=_CYAN, marker_color=_AMBER,
    )
    _plot_xy_with_err(
        x, np.array([r["miss_ekf_mean"]  for r in r0]),
              np.array([r["miss_ekf_std"]   for r in r0]),
        xlabel="σ_px  [px]",
        ylabel="Terminal miss mean ± std  [ND]",
        title=f"Terminal Miss vs Pixel Noise  (mini-MC n={args.n_mc})",
        outpath=plots_dir / "06b_miss_vs_sigma.png",
        color=_VIOLET, marker_color=_AMBER,
    )
    _plot_xy_with_err(
        x, np.array([r["pos_err_tc_mean"] for r in r0]),
              np.array([r["pos_err_tc_std"]  for r in r0]),
        xlabel="σ_px  [px]",
        ylabel="‖r̂(tc)−r(tc)‖ mean ± std  [ND]",
        title=f"Position Error at tc vs Pixel Noise  (mini-MC n={args.n_mc})",
        outpath=plots_dir / "06b_poserr_tc_vs_sigma.png",
        color=_GREEN, marker_color=_AMBER,
    )

    r1 = _sel(1.0)
    x  = np.array([r["tc"] for r in r1])
    _plot_xy_with_err(
        x, np.array([r["dv_delta_mean"] for r in r1]),
              np.array([r["dv_delta_std"]  for r in r1]),
        xlabel="Correction time  tc  [ND]",
        ylabel="‖Δdv‖ mean ± std  [ND]",
        title=f"Burn Error vs Correction Time  (mini-MC n={args.n_mc})",
        outpath=plots_dir / "06b_dv_error_vs_tc.png",
        color=_CYAN, marker_color=_AMBER,
    )
    _plot_xy_with_err(
        x, np.array([r["miss_ekf_mean"] for r in r1]),
              np.array([r["miss_ekf_std"]  for r in r1]),
        xlabel="Correction time  tc  [ND]",
        ylabel="Terminal miss mean ± std  [ND]",
        title=f"Terminal Miss vs Correction Time  (mini-MC n={args.n_mc})",
        outpath=plots_dir / "06b_miss_vs_tc.png",
        color=_VIOLET, marker_color=_AMBER,
    )
    _plot_xy_with_err(
        x, np.array([r["pos_err_tc_mean"] for r in r1]),
              np.array([r["pos_err_tc_std"]  for r in r1]),
        xlabel="Correction time  tc  [ND]",
        ylabel="‖r̂(tc)−r(tc)‖ mean ± std  [ND]",
        title=f"Position Error at tc vs Correction Time  (mini-MC n={args.n_mc})",
        outpath=plots_dir / "06b_poserr_tc_vs_tc.png",
        color=_GREEN, marker_color=_AMBER,
    )

    r2 = _sel(2.0)
    x  = np.array([r["sigma_px"] for r in r2])
    _plot_xy_with_err(
        x, np.array([r["dv_delta_mean"] for r in r2]),
              np.array([r["dv_delta_std"]  for r in r2]),
        xlabel="σ_px  [px]",
        ylabel="‖Δdv‖ mean ± std  [ND]",
        title=f"Burn Error vs σ_px  (dropout=0.05, fixed pointing, n={args.n_mc})",
        outpath=plots_dir / "06b_bonus_dv_error_vs_sigma.png",
        color=_RED, marker_color=_AMBER,
    )
    _plot_xy_with_err(
        x, np.array([r["miss_ekf_mean"] for r in r2]),
              np.array([r["miss_ekf_std"]  for r in r2]),
        xlabel="σ_px  [px]",
        ylabel="Terminal miss mean ± std  [ND]",
        title=f"Terminal Miss vs σ_px  (dropout=0.05, fixed pointing, n={args.n_mc})",
        outpath=plots_dir / "06b_bonus_miss_vs_sigma.png",
        color=_RED, marker_color=_AMBER,
    )
    _plot_xy_with_err(
        x, np.array([r["pos_err_tc_mean"] for r in r2]),
              np.array([r["pos_err_tc_std"]  for r in r2]),
        xlabel="σ_px  [px]",
        ylabel="‖r̂(tc)−r(tc)‖ mean ± std  [ND]",
        title=f"Position Error at tc vs σ_px  (dropout=0.05, fixed pointing, n={args.n_mc})",
        outpath=plots_dir / "06b_bonus_poserr_tc_vs_sigma.png",
        color=_RED, marker_color=_AMBER,
    )

    print("\n06B mini-MC sensitivity complete.")
    print(f"Wrote CSV:   {csv_path}")
    print(f"Wrote plots: {plots_dir}")


if __name__ == "__main__":
    main()
