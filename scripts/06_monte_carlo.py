from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import matplotlib.pyplot as plt
import numpy as np


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


def _plot_hist(vals: np.ndarray, *, xlabel: str, title: str, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    vals = vals[np.isfinite(vals)]
    plt.hist(vals, bins=30)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _plot_scatter(x: np.ndarray, y: np.ndarray, *, xlabel: str, ylabel: str, title: str, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    m = np.isfinite(x) & np.isfinite(y)
    plt.scatter(x[m], y[m], s=12)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _try_get_study_config(study: str, *, mu: float, t0: float, tf: float, tc: float, dt_meas: float):
    """
    Your study_config.py factories require keyword-only args:
      make_baseline_mc_config(*, mu, t0, tf, tc, dt_meas, ...)
    """
    import mc.study_config as sc  

    name = study.strip().lower()
    fn_map = {
        "baseline": getattr(sc, "make_baseline_mc_config", None),
        "dropout": getattr(sc, "make_dropout_mc_config", None),
        "no_tracking": getattr(sc, "make_no_tracking_mc_config", None),
        "notracking": getattr(sc, "make_no_tracking_mc_config", None),
        "high_noise": getattr(sc, "make_high_noise_mc_config", None),
        "highnoise": getattr(sc, "make_high_noise_mc_config", None),
    }
    fn = fn_map.get(name)
    if fn is None:
        raise ValueError(f"Unknown study '{study}'. Available: {list(fn_map.keys())}")

    return fn(mu=mu, t0=t0, tf=tf, tc=tc, dt_meas=dt_meas)


def _build_config(args):
    cfg = _try_get_study_config(args.study, mu=args.mu, t0=args.t0, tf=args.tf, tc=args.tc, dt_meas=args.dt_meas)
    if args.n_trials is not None:
        cfg.n_trials = int(args.n_trials)
    if args.sigma_px is not None:
        cfg.sigma_px = float(args.sigma_px)
    if args.dropout_prob is not None:
        cfg.dropout_prob = float(args.dropout_prob)
    if args.tracking_attitude is not None:
        cfg.tracking_attitude = bool(args.tracking_attitude)
    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--study", type=str, default="baseline", help="baseline | dropout | no_tracking | high_noise")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--plots-dir", type=str, default="results/plots")
    p.add_argument("--tol", type=float, default=1e-3, help="Miss tolerance for success rate.")
    p.add_argument("--no-plot-d", action="store_true", help="Disable optional Plot D (traceP vs miss).")

    # timeline / dynamics
    p.add_argument("--mu", type=float, default=0.0121505856)
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--tf", type=float, default=6.0)
    p.add_argument("--tc", type=float, default=2.0)
    p.add_argument("--dt-meas", type=float, default=0.02)

    p.add_argument("--sigma-px", type=float, default=None)
    p.add_argument("--dropout-prob", type=float, default=None)
    p.add_argument("--tracking-attitude", type=int, default=None, help="1/0 override")

    args = p.parse_args()

    repo_root = _repo_root_from_this_file()
    _ensure_paths(repo_root)

    import mc  
    from mc import run_monte_carlo, save_results_csv, summarize_results  

    run_case = _load_run_case(repo_root)
    config = _build_config(args)

    print(f"Running 06C Monte Carlo: study={config.study_name}, n_trials={config.n_trials}")
    print(f"  tc={config.tc}, sigma_px={config.sigma_px}, dropout_prob={config.dropout_prob}, tracking_attitude={config.tracking_attitude}")

    results = run_monte_carlo(config, run_case)

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = plots_dir / f"06c_{config.study_name}_results.csv"
    save_results_csv(results, csv_path)

    summary = summarize_results(results, tol=float(args.tol))

    dv_delta = np.array([r.dv_delta_mag for r in results], dtype=float)
    dv_infl = np.array([r.dv_inflation for r in results], dtype=float)  
    miss_ekf = np.array([r.miss_ekf for r in results], dtype=float)
    pos_err_tc = np.array([r.pos_err_tc for r in results], dtype=float)
    tracePpos_tc = np.array([r.tracePpos_tc for r in results], dtype=float)

    _plot_hist(
        dv_delta,
        xlabel="dv_delta_mag = ||dv_ekf - dv_perfect||",
        title=f"Burn error magnitude histogram ({config.study_name}, n={len(results)})",
        outpath=plots_dir / "06c_hist_dv_delta_mag.png",
    )
    try:
        import shutil
        shutil.copyfile(plots_dir / "06c_hist_dv_delta_mag.png", plots_dir / "06c_hist_dv_inflation.png")
    except Exception:
        pass

    _plot_hist(
        miss_ekf,
        xlabel="miss_ekf = ||r_ekf(tf) - r_target||",
        title=f"EKF terminal miss histogram ({config.study_name}, n={len(results)})",
        outpath=plots_dir / "06c_hist_miss_ekf.png",
    )
    _plot_scatter(
        pos_err_tc,
        dv_delta,
        xlabel="pos_err_tc = ||r_hat(tc) - r_true(tc)||",
        ylabel="dv_delta_mag = ||dv_ekf - dv_perfect||",
        title=f"pos_err(tc) vs burn error ({config.study_name})",
        outpath=plots_dir / "06c_scatter_poserr_vs_dvdelta.png",
    )
    if not args.no_plot_d:
        _plot_scatter(
            tracePpos_tc,
            miss_ekf,
            xlabel="tracePpos_tc = trace(P[:3,:3]) at tc",
            ylabel="miss_ekf",
            title=f"trace(P_pos) vs miss ({config.study_name})",
            outpath=plots_dir / "06c_scatter_traceP_vs_miss.png",
        )

    print("\n=== 06C Summary ===")
    print(f"trials: {summary.get('n')}")
    dv_delta_mean = float(np.nanmean(dv_delta))
    dv_delta_std = float(np.nanstd(dv_delta))
    print(f"mean dv_delta_mag:  {dv_delta_mean:.6g}  (std {dv_delta_std:.6g})")
    print(f"mean dv_inflation:  {summary.get('dv_inflation_mean'):.6g}  (std {summary.get('dv_inflation_std'):.6g})")
    print(f"p95 miss_ekf:       {summary.get('miss_ekf_p95'):.6g}")
    if "success_rate" in summary:
        print(f"success_rate (tol={args.tol:g}): {summary.get('success_rate'):.3f}")

    print("\nWrote:")
    print(f"  CSV:   {csv_path}")
    print(f"  PlotA: {plots_dir / '06c_hist_dv_delta_mag.png'}")
    print(f"        (alias) {plots_dir / '06c_hist_dv_inflation.png'}")
    print(f"  PlotB: {plots_dir / '06c_hist_miss_ekf.png'}")
    print(f"  PlotC: {plots_dir / '06c_scatter_poserr_vs_dvdelta.png'}")
    if not args.no_plot_d:
        print(f"  PlotD: {plots_dir / '06c_scatter_traceP_vs_miss.png'}")


if __name__ == "__main__":
    main()