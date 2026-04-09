from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import FancyBboxPatch


_BG     = "#080B14"
_PANEL  = "#0D1117"
_BORDER = "#1C2340"
_TEXT   = "#DCE0EC"
_DIM    = "#4A5270"
_CYAN   = "#22D3EE"
_AMBER  = "#F59E0B"
_GREEN  = "#10B981"
_RED    = "#F43F5E"
_VIOLET = "#A78BFA"
_PINK   = "#EC4899"

_CMAP_CYAN   = mcolors.LinearSegmentedColormap.from_list("cy",   [_PANEL, _CYAN])
_CMAP_VIOLET = mcolors.LinearSegmentedColormap.from_list("viol", [_PANEL, _VIOLET])
_CMAP_HOT    = mcolors.LinearSegmentedColormap.from_list("hot",  [_CYAN, _VIOLET, _PINK])


_STUDY_PRESETS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "study_name":   "baseline",
        "sigma_px":     1.0,
        "dropout_prob": 0.0,
        "camera_mode":  "estimate_tracking",
    },
    "dropout": {
        "study_name":   "dropout",
        "sigma_px":     1.0,
        "dropout_prob": 0.20,
        "camera_mode":  "estimate_tracking",
    },
    "no_tracking": {
        "study_name":   "no_tracking",
        "sigma_px":     1.0,
        "dropout_prob": 0.0,
        "camera_mode":  "fixed",
    },
    "high_noise": {
        "study_name":   "high_noise",
        "sigma_px":     5.0,
        "dropout_prob": 0.0,
        "camera_mode":  "estimate_tracking",
    },
}

_STUDY_ALIASES: Dict[str, str] = {
    "notracking":  "no_tracking",
    "highnoise":   "high_noise",
    "high-noise":  "high_noise",
    "no-tracking": "no_tracking",
}


def _resolve_study(name: str) -> str:
    key = name.strip().lower()
    key = _STUDY_ALIASES.get(key, key)
    if key not in _STUDY_PRESETS:
        available = list(_STUDY_PRESETS.keys())
        raise ValueError(f"Unknown study '{name}'. Available: {available}")
    return key



def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_paths(repo_root: Path) -> None:
    for p in [repo_root, repo_root / "src"]:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _load_run_case(repo_root: Path) -> Callable[..., Any]:
    target = repo_root / "scripts" / "06_midcourse_ekf_correction.py"
    if not target.exists():
        raise FileNotFoundError(f"Could not find EKF script at: {target}")
    spec = importlib.util.spec_from_file_location("midcourse06a", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {target}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run_case"):
        raise AttributeError("06_midcourse_ekf_correction.py does not define run_case()")
    return getattr(mod, "run_case")



def _build_config(args: argparse.Namespace) -> Any:
    from mc.types import MonteCarloConfig

    study_key = _resolve_study(args.study)
    preset    = dict(_STUDY_PRESETS[study_key])

    preset["n_trials"] = int(args.n_trials)
    preset["mu"]       = float(args.mu)
    preset["t0"]       = float(args.t0)
    preset["tf"]       = float(args.tf)
    preset["tc"]       = float(args.tc)
    preset["dt_meas"]  = float(args.dt_meas)

    if args.sigma_px     is not None:
        preset["sigma_px"]     = float(args.sigma_px)
    if args.dropout_prob is not None:
        preset["dropout_prob"] = float(args.dropout_prob)
    if args.camera_mode  is not None:
        preset["camera_mode"]  = str(args.camera_mode)

    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(MonteCarloConfig)}
    filtered = {k: v for k, v in preset.items() if k in valid_fields}

    return MonteCarloConfig(**filtered)



def _apply_dark_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor":  _BG,
        "axes.facecolor":    _PANEL,
        "axes.edgecolor":    _BORDER,
        "axes.labelcolor":   _TEXT,
        "axes.titlecolor":   _TEXT,
        "text.color":        _TEXT,
        "xtick.color":       _DIM,
        "ytick.color":       _DIM,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "grid.color":        _BORDER,
        "grid.alpha":        1.0,
        "grid.linestyle":    "-",
        "grid.linewidth":    0.4,
        "lines.linewidth":   2.0,
        "legend.facecolor":  _PANEL,
        "legend.edgecolor":  _BORDER,
        "legend.labelcolor": _TEXT,
        "legend.fontsize":   9,
        "savefig.facecolor": _BG,
        "savefig.edgecolor": _BG,
        "font.family":       "monospace",
        "font.size":         11,
    })


def _style_ax(ax: plt.Axes, *, minor_grid: bool = True) -> None:
    ax.set_facecolor(_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(_BORDER)
        sp.set_linewidth(0.8)
    ax.grid(True, which="major", color=_BORDER, linewidth=0.4)
    if minor_grid:
        ax.grid(True, which="minor", color=_BORDER, linewidth=0.2, alpha=0.5)
    ax.tick_params(colors=_DIM, which="both", length=3)


def _glow_vline(ax: plt.Axes, x: float, color: str, lw: float = 1.5,
                ls: str = "--", label: Optional[str] = None) -> None:
    for width, alpha in [(lw * 6, 0.08), (lw * 3, 0.15), (lw, 1.0)]:
        ax.axvline(x, color=color, lw=width, ls=ls,
                   alpha=alpha, label=label if width == lw else None)


def _stats_box(ax: plt.Axes, lines: List[str], loc: str = "upper right") -> None:
    text = "\n".join(lines)
    props = dict(boxstyle="round,pad=0.5", facecolor=_BG,
                 edgecolor=_BORDER, alpha=0.85)
    anchors = {
        "upper right": (0.97, 0.97), "upper left": (0.03, 0.97),
        "lower right": (0.97, 0.03), "lower left": (0.03, 0.03),
    }
    xy = anchors.get(loc, (0.97, 0.97))
    va = "top"    if "upper" in loc else "bottom"
    ha = "right"  if "right" in loc else "left"
    ax.text(*xy, text, transform=ax.transAxes, fontsize=8.5,
            va=va, ha=ha, color=_TEXT, bbox=props, family="monospace")



def _kde_curve(vals: np.ndarray, n_points: int = 300) -> tuple[np.ndarray, np.ndarray]:
    if vals.size < 2:
        return np.array([]), np.array([])
    std = float(np.std(vals))
    if std == 0:
        return np.array([]), np.array([])
    bw = 1.06 * std * vals.size ** (-0.2)
    x  = np.linspace(vals.min() - 2 * bw, vals.max() + 2 * bw, n_points)
    density = np.mean(
        np.exp(-0.5 * ((x[:, None] - vals[None, :]) / bw) ** 2), axis=1
    ) / (bw * np.sqrt(2 * np.pi))
    return x, density



def _plot_hist(
    vals: np.ndarray,
    *,
    xlabel: str,
    title: str,
    outpath: Path,
    color: str = _CYAN,
    tol: Optional[float] = None,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    vals = vals[np.isfinite(vals)]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(_BG)
    _style_ax(ax)

    if vals.size == 0:
        ax.set_title(f"{title}  [no data]", color=_RED)
        fig.tight_layout()
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        return

    n_bins = min(40, max(8, vals.size // 3))
    counts, edges, patches = ax.hist(
        vals, bins=n_bins, color=color, edgecolor=_BG,
        linewidth=0.4, alpha=0.0,
    )

    max_c = counts.max() or 1
    cmap  = _CMAP_CYAN if color == _CYAN else _CMAP_VIOLET
    for patch, count in zip(patches, counts):
        patch.set_facecolor(cmap(0.35 + 0.65 * count / max_c))
        patch.set_alpha(0.92)

    kx, ky = _kde_curve(vals)
    if kx.size:
        bin_width  = edges[1] - edges[0]
        ky_scaled  = ky * vals.size * bin_width
        for lw, al in [(8, 0.06), (4, 0.12), (1.8, 1.0)]:
            ax.plot(kx, ky_scaled, color=color, lw=lw, alpha=al, zorder=4)

    mu_v  = float(np.mean(vals))
    std_v = float(np.std(vals))
    _glow_vline(ax, mu_v, _AMBER, lw=1.6, ls="--")
    _glow_vline(ax, mu_v - std_v, _VIOLET, lw=1.2, ls=":")
    _glow_vline(ax, mu_v + std_v, _VIOLET, lw=1.2, ls=":")

    if tol is not None:
        _glow_vline(ax, tol, _GREEN, lw=1.2, ls="-.")

    p95 = float(np.percentile(vals, 95))
    lines = [
        f"n     = {vals.size}",
        f"μ     = {mu_v:.3e}",
        f"σ     = {std_v:.3e}",
        f"p95   = {p95:.3e}",
    ]
    if tol is not None:
        sr = float(np.mean(vals < tol))
        lines.append(f"P<tol = {sr:.1%}")
    _stats_box(ax, lines)

    ax.set_xlabel(xlabel, color=_TEXT, labelpad=6)
    ax.set_ylabel("count", color=_TEXT, labelpad=6)
    ax.set_title(title, color=_TEXT, pad=10, fontweight="bold")
    ax.set_xlim(left=max(0, edges[0]))

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, facecolor=_BG)
    plt.close(fig)



def _plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    outpath: Path,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(_BG)
    _style_ax(ax)

    if xm.size == 0:
        ax.set_title(f"{title}  [no data]", color=_RED)
        fig.tight_layout()
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        return

    h, xe, ye = np.histogram2d(xm, ym, bins=min(30, xm.size // 2 or 5))
    xi = np.searchsorted(xe[:-1], xm) - 1
    yi = np.searchsorted(ye[:-1], ym) - 1
    xi = np.clip(xi, 0, h.shape[0] - 1)
    yi = np.clip(yi, 0, h.shape[1] - 1)
    density = h[xi, yi]

    order = density.argsort()
    sc = ax.scatter(
        xm[order], ym[order],
        c=density[order], cmap=_CMAP_HOT,
        s=22, alpha=0.85, edgecolors="none", zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.035)
    cbar.ax.yaxis.set_tick_params(color=_DIM)
    cbar.outline.set_edgecolor(_BORDER)
    cbar.set_label("local density", color=_DIM, fontsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_DIM, fontsize=7)

    if xm.size > 2:
        coeffs = np.polyfit(xm, ym, 1)
        xfit   = np.linspace(xm.min(), xm.max(), 200)
        yfit   = np.polyval(coeffs, xfit)
        for lw, al in [(10, 0.05), (5, 0.10), (1.8, 1.0)]:
            ax.plot(xfit, yfit, color=_AMBER, lw=lw, alpha=al,
                    ls="--", zorder=4,
                    label=f"slope = {coeffs[0]:.2e}" if lw == 1.8 else None)

        r = float(np.corrcoef(xm, ym)[0, 1])
        _stats_box(ax, [
            f"n = {xm.size}",
            f"slope = {coeffs[0]:.2e}",
            f"r = {r:.3f}",
        ])
        ax.legend(fontsize=8)

    ax.set_xlabel(xlabel, color=_TEXT, labelpad=6)
    ax.set_ylabel(ylabel, color=_TEXT, labelpad=6)
    ax.set_title(title, color=_TEXT, pad=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, facecolor=_BG)
    plt.close(fig)



def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monte Carlo study for EKF midcourse-correction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--study", default="baseline",
                   help="baseline | dropout | no_tracking | high_noise")
    p.add_argument("--n-trials",     type=int,   default=20)
    p.add_argument("--plots-dir",    type=str,   default="results/plots")
    p.add_argument("--tol",          type=float, default=1e-3,
                   help="Miss tolerance for success-rate annotation.")
    p.add_argument("--no-plot-d",    action="store_true",
                   help="Skip Plot D (tr(P_pos) vs miss).")

    p.add_argument("--mu",           type=float, default=0.0121505856)
    p.add_argument("--t0",           type=float, default=0.0)
    p.add_argument("--tf",           type=float, default=6.0)
    p.add_argument("--tc",           type=float, default=2.0)
    p.add_argument("--dt-meas",      type=float, default=0.02)

    p.add_argument("--sigma-px",     type=float, default=None)
    p.add_argument("--dropout-prob", type=float, default=None)
    p.add_argument("--camera-mode",  type=str,   default=None,
                   help="estimate_tracking | fixed")

    return p.parse_args()


def main() -> None:
    _apply_dark_theme()
    args = p = _parse_args()

    repo_root = _repo_root()
    _ensure_paths(repo_root)

    from mc import run_monte_carlo, save_results_csv, summarize_results

    run_case = _load_run_case(repo_root)
    config   = _build_config(args)

    cam = getattr(config, "camera_mode",
                  getattr(config, "tracking_attitude", "unknown"))
    print(
        f"\n▸ 06C Monte Carlo — study={config.study_name}  "
        f"n_trials={config.n_trials}\n"
        f"  mu={config.mu}  t0={config.t0}  tf={config.tf}  tc={config.tc}\n"
        f"  sigma_px={config.sigma_px}  dropout_prob={config.dropout_prob}  "
        f"camera={cam}\n"
    )

    results = run_monte_carlo(config, run_case)

    if not results:
        print("ERROR: no trials completed — check run_case() and config.", file=sys.stderr)
        sys.exit(1)

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = plots_dir / f"06c_{config.study_name}_results.csv"
    save_results_csv(results, csv_path)

    def _arr(attr: str) -> np.ndarray:
        out = []
        for r in results:
            try:
                v = getattr(r, attr)
                out.append(float(v))
            except (AttributeError, TypeError, ValueError):
                out.append(float("nan"))
        return np.array(out, dtype=float)

    dv_delta     = _arr("dv_delta_mag")
    dv_infl      = _arr("dv_inflation")
    miss_ekf     = _arr("miss_ekf")
    pos_err_tc   = _arr("pos_err_tc")
    tracePpos_tc = _arr("tracePpos_tc")

    summary = summarize_results(results, tol=float(args.tol))
    n       = len(results)

    _plot_hist(
        dv_delta,
        xlabel="‖Δv_ekf − Δv_perfect‖  [ND]",
        title=f"Burn Error Magnitude  ·  {config.study_name}  (n={n})",
        outpath=plots_dir / "06c_hist_dv_delta_mag.png",
        color=_CYAN,
        tol=None,
    )

    _plot_hist(
        dv_infl,
        xlabel="Δv inflation  [ND]",
        title=f"ΔV Inflation  ·  {config.study_name}  (n={n})",
        outpath=plots_dir / "06c_hist_dv_inflation.png",
        color=_CYAN,
        tol=None,
    )

    _plot_hist(
        miss_ekf,
        xlabel="‖r_ekf(tf) − r_target‖  [ND]",
        title=f"EKF Terminal Miss  ·  {config.study_name}  (n={n})",
        outpath=plots_dir / "06c_hist_miss_ekf.png",
        color=_VIOLET,
        tol=float(args.tol),
    )

    _plot_scatter(
        pos_err_tc, dv_delta,
        xlabel="‖r̂(tc) − r(tc)‖  [ND]",
        ylabel="‖Δv_ekf − Δv_perfect‖  [ND]",
        title=f"State Error at tc  vs  Burn Error  ·  {config.study_name}",
        outpath=plots_dir / "06c_scatter_poserr_vs_dvdelta.png",
    )

    if not args.no_plot_d:
        _plot_scatter(
            tracePpos_tc, miss_ekf,
            xlabel="tr(P_pos) at tc  [ND²]",
            ylabel="‖r_ekf(tf) − r_target‖  [ND]",
            title=f"Position Covariance  vs  Terminal Miss  ·  {config.study_name}",
            outpath=plots_dir / "06c_scatter_traceP_vs_miss.png",
        )

    print("=== 06C Summary " + "=" * 46)
    print(f"  trials            : {summary.get('n', n)}")
    print(f"  mean dv_delta_mag : {float(np.nanmean(dv_delta)):.4e}"
          f"  (σ {float(np.nanstd(dv_delta)):.4e})")
    print(f"  mean dv_inflation : {summary.get('dv_inflation_mean', float('nan')):.4e}"
          f"  (σ {summary.get('dv_inflation_std', float('nan')):.4e})")
    print(f"  p95  miss_ekf     : {summary.get('miss_ekf_p95', float('nan')):.4e}")
    if "success_rate" in summary:
        print(f"  success_rate      : {summary['success_rate']:.3f}"
              f"  (tol={args.tol:g})")
    print("=" * 62)

    print("\nOutputs:")
    print(f"  CSV  : {csv_path}")
    plot_names = [
        "06c_hist_dv_delta_mag.png",
        "06c_hist_dv_inflation.png",
        "06c_hist_miss_ekf.png",
        "06c_scatter_poserr_vs_dvdelta.png",
    ]
    if not args.no_plot_d:
        plot_names.append("06c_scatter_traceP_vs_miss.png")
    for name in plot_names:
        print(f"  Plot : {plots_dir / name}")


if __name__ == "__main__":
    main()
