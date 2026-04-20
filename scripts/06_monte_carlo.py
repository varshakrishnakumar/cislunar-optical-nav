from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from _analysis_common import load_midcourse_run_case

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from _common import ensure_src_on_path, repo_path


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
_CMAP_GREEN  = mcolors.LinearSegmentedColormap.from_list("green", [_PANEL, _GREEN])
_CMAP_AMBER  = mcolors.LinearSegmentedColormap.from_list("amber", [_PANEL, _AMBER])
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
    preset["q_acc"] = float(args.q_acc)

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
    xmin_zero: bool = True,
    xlim: Optional[tuple[float, float]] = None,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    vals = vals[np.isfinite(vals)]
    if xlim is not None:
        lo, hi = xlim
        vals = vals[(vals >= lo) & (vals <= hi)]

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
    cmap = {
        _CYAN: _CMAP_CYAN,
        _VIOLET: _CMAP_VIOLET,
        _GREEN: _CMAP_GREEN,
        _AMBER: _CMAP_AMBER,
    }.get(color, _CMAP_VIOLET)
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
    med_v = float(np.median(vals))
    _glow_vline(ax, med_v, _AMBER, lw=1.8, ls="-")
    _glow_vline(ax, mu_v, _VIOLET, lw=1.2, ls="--")

    if tol is not None:
        _glow_vline(ax, tol, _GREEN, lw=1.2, ls="-.")

    p05 = float(np.percentile(vals, 5))
    p95 = float(np.percentile(vals, 95))
    lines = [
        f"n      = {vals.size}",
        f"median = {med_v:.3e}",
        f"p05    = {p05:.3e}",
        f"p95    = {p95:.3e}",
        f"μ      = {mu_v:.3e}",
        f"σ      = {std_v:.3e}",
    ]
    if tol is not None:
        sr = float(np.mean(vals < tol))
        lines.append(f"P<tol = {sr:.1%}")
    # Place the stats box in the corner with less data mass to avoid overlap.
    mid = float(vals.min() + vals.max()) / 2.0
    stats_loc = "upper left" if mu_v > mid else "upper right"
    _stats_box(ax, lines, loc=stats_loc)

    ax.set_xlabel(xlabel, color=_TEXT, labelpad=6)
    ax.set_ylabel("count", color=_TEXT, labelpad=6)
    ax.set_title(title, color=_TEXT, pad=10, fontweight="bold")
    if xlim is not None:
        ax.set_xlim(*xlim)
    elif xmin_zero:
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

    if xm.size > 2 and np.ptp(xm) > 0.0 and np.ptp(ym) > 0.0:
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


def _plot_representative_orbit(
    out: Dict[str, Any],
    *,
    mu: float,
    trial_id: int,
    outpath: Path,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    dbg = out["debug"]
    xs_nom = np.asarray(dbg["xs_nom"], dtype=float)
    xs_true = np.asarray(dbg["xs_true"], dtype=float)
    xs_unc = np.asarray(dbg["xs_unc_tf"], dtype=float)
    xs_perf = np.asarray(dbg["xs_perf_tf"], dtype=float)
    xs_ekf = np.asarray(dbg["xs_ekf_tf"], dtype=float)
    r_target = np.asarray(dbg["r_target"], dtype=float)
    k_tc = int(dbg["k_tc"])

    earth = np.array([-float(mu), 0.0])
    moon = np.array([1.0 - float(mu), 0.0])

    fig, ax = plt.subplots(figsize=(13, 5.2), layout="constrained")
    fig.patch.set_facecolor(_BG)
    _style_ax(ax)

    ax.scatter([earth[0]], [earth[1]], s=110, color="#3B82F6", edgecolors=_BG, zorder=5, label="Earth")
    ax.scatter([moon[0]], [moon[1]], s=75, color="#C8CDD8", edgecolors=_BG, zorder=5, label="Moon")
    ax.plot(xs_nom[:, 0], xs_nom[:, 1], color=_CYAN, lw=1.4, alpha=0.72, label="nominal target arc")
    ax.plot(xs_true[:, 0], xs_true[:, 1], color=_AMBER, lw=1.3, ls="--", alpha=0.85, label="truth before burn")
    ax.plot(xs_unc[:, 0], xs_unc[:, 1], color=_AMBER, lw=1.1, ls=":", alpha=0.65, label="uncorrected after tc")
    ax.plot(xs_perf[:, 0], xs_perf[:, 1], color=_GREEN, lw=1.7, label="perfect-info burn")
    ax.plot(xs_ekf[:, 0], xs_ekf[:, 1], color=_VIOLET, lw=1.7, ls=(0, (5, 3)), label="IEKF burn")
    ax.scatter([r_target[0]], [r_target[1]], s=120, color=_AMBER, marker="*", edgecolors=_BG, zorder=8, label="target")
    if 0 <= k_tc < len(xs_true):
        ax.scatter([xs_true[k_tc, 0]], [xs_true[k_tc, 1]], s=85, color=_RED, marker="D", edgecolors=_BG, zorder=7, label="correction time")

    _stats_box(
        ax,
        [
            f"trial = {trial_id}",
            f"miss_IEKF = {float(out['miss_ekf']):.3e}",
            f"valid rate = {float(out['valid_rate']):.2f}",
            f"mean NIS = {float(out['nis_mean']):.2f}",
        ],
        loc="upper left",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.1, 1.15)
    ax.set_ylim(-0.2, 0.2)
    ax.set_xlabel("x [dimensionless CR3BP length]", color=_TEXT, labelpad=6)
    ax.set_ylabel("y [dimensionless CR3BP length]", color=_TEXT, labelpad=8)
    ax.set_title("Representative Monte Carlo Trial Orbit and Correction", color=_TEXT, pad=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.savefig(outpath, dpi=220, facecolor=_BG)
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
    p.add_argument("--representative-trial-id", type=int, default=0,
                   help="Replay this trial and save an orbit-level report plot; set negative to skip.")

    p.add_argument("--mu",           type=float, default=0.0121505856)
    p.add_argument("--t0",           type=float, default=0.0)
    p.add_argument("--tf",           type=float, default=6.0)
    p.add_argument("--tc",           type=float, default=2.0)
    p.add_argument("--dt-meas",      type=float, default=0.02)

    p.add_argument("--sigma-px",     type=float, default=None)
    p.add_argument("--dropout-prob", type=float, default=None)
    p.add_argument("--camera-mode",  type=str,   default=None,
                   help="estimate_tracking | fixed")
    p.add_argument("--q-acc",        type=float, default=1e-14,
                   help="EKF process-noise density (ND CR3BP units).")

    return p.parse_args()


def main() -> None:
    _apply_dark_theme()
    args = _parse_args()

    ensure_src_on_path()

    from mc import run_monte_carlo, save_results_csv, summarize_results
    from mc.sampler import make_trial_rng, sample_estimation_error, sample_injection_error

    run_case = load_midcourse_run_case()
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

    plots_dir = repo_path(args.plots_dir)
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
    dv_bias      = _arr("dv_mag_bias")
    dv_infl_pct  = _arr("dv_inflation_pct") * 100.0
    miss_ekf     = _arr("miss_ekf")
    pos_err_tc   = _arr("pos_err_tc")
    tracePpos_tc = _arr("tracePpos_tc")
    nis_mean     = _arr("nis_mean")
    valid_rate   = _arr("valid_rate")

    summary = summarize_results(results, tol=float(args.tol))
    n       = len(results)

    def _summary_stat(metric: str, stat: str) -> float:
        block = summary.get(metric, {})
        if not isinstance(block, dict):
            return float("nan")
        return float(block.get(stat, float("nan")))

    _plot_hist(
        dv_delta,
        xlabel="‖Δv_ekf − Δv_perfect‖  [dimensionless CR3BP velocity]",
        title=f"Burn Error Magnitude  ·  {config.study_name}  (n={n})",
        outpath=plots_dir / "06c_hist_dv_delta_mag.png",
        color=_CYAN,
        tol=None,
    )

    _plot_hist(
        dv_bias,
        xlabel="|Δv_EKF| − |Δv_perfect|  [dimensionless CR3BP velocity]",
        title=f"Burn-Magnitude Bias  ·  {config.study_name}  (n={n})",
        outpath=plots_dir / "06c_hist_dv_mag_bias.png",
        color=_CYAN,
        tol=None,
        xmin_zero=False,
    )

    # Clip to [-100, 500] %: above that is ratio-blowup (tiny |Δv_perfect|).
    _plot_hist(
        dv_infl_pct,
        xlabel="(|Δv_EKF| / |Δv_perfect| − 1) × 100  [%]",
        title=f"Relative ΔV Inflation  ·  {config.study_name}  (n={n}, clipped)",
        outpath=plots_dir / "06c_hist_dv_inflation_pct.png",
        color=_AMBER,
        tol=None,
        xmin_zero=False,
        xlim=(-100.0, 500.0),
    )

    _plot_hist(
        miss_ekf,
        xlabel="‖r_ekf(tf) − r_target‖  [dimensionless CR3BP length]",
        title=f"EKF Terminal Miss  ·  {config.study_name}  (n={n})",
        outpath=plots_dir / "06c_hist_miss_ekf.png",
        color=_VIOLET,
        tol=float(args.tol),
    )

    _plot_hist(
        valid_rate,
        xlabel="valid bearing-update rate  [fraction of scheduled updates]",
        title=f"Measurement Availability  ·  {config.study_name}  (n={n})",
        outpath=plots_dir / "06c_hist_valid_measurement_rate.png",
        color=_GREEN,
        tol=None,
    )

    _plot_scatter(
        pos_err_tc, dv_delta,
        xlabel="‖r̂(tc) − r(tc)‖  [dimensionless CR3BP length]",
        ylabel="‖Δv_ekf − Δv_perfect‖  [dimensionless CR3BP velocity]",
        title=f"State Error at tc  vs  Burn Error  ·  {config.study_name}",
        outpath=plots_dir / "06c_scatter_poserr_vs_dvdelta.png",
    )

    _plot_scatter(
        valid_rate, miss_ekf,
        xlabel="valid bearing-update rate  [fraction]",
        ylabel="‖r_ekf(tf) − r_target‖  [dimensionless CR3BP length]",
        title=f"Measurement Availability  vs  Terminal Miss  ·  {config.study_name}",
        outpath=plots_dir / "06c_scatter_validrate_vs_miss.png",
    )

    _plot_scatter(
        nis_mean, miss_ekf,
        xlabel="mean NIS  [expected ≈ 2 for a consistent 2-D bearing residual]",
        ylabel="‖r_ekf(tf) − r_target‖  [dimensionless CR3BP length]",
        title=f"IEKF Innovation Consistency  vs  Terminal Miss  ·  {config.study_name}",
        outpath=plots_dir / "06c_scatter_nis_vs_miss.png",
    )

    if not args.no_plot_d:
        _plot_scatter(
            tracePpos_tc, miss_ekf,
            xlabel="tr(P_pos) at tc  [dimensionless CR3BP length²]",
            ylabel="‖r_ekf(tf) − r_target‖  [dimensionless CR3BP length]",
            title=f"Position Covariance  vs  Terminal Miss  ·  {config.study_name}",
            outpath=plots_dir / "06c_scatter_traceP_vs_miss.png",
        )

    representative_plot: Path | None = None
    if int(args.representative_trial_id) >= 0:
        trial_id = int(args.representative_trial_id)
        rng = make_trial_rng(config.base_seed, trial_id)
        seed = int(rng.integers(0, 2**31 - 1))
        dx0 = sample_injection_error(
            rng,
            sigma_r=float(config.sigma_r_inj),
            sigma_v=float(config.sigma_v_inj),
            planar_only=bool(config.planar_only),
        )
        est_err = sample_estimation_error(
            rng,
            sigma_r=float(config.sigma_r_est),
            sigma_v=float(config.sigma_v_est),
            planar_only=bool(config.planar_only),
        )
        rep_out = run_case(
            mu=config.mu,
            t0=config.t0,
            tf=config.tf,
            tc=config.tc,
            dt_meas=config.dt_meas,
            sigma_px=config.sigma_px,
            dropout_prob=config.dropout_prob,
            seed=seed,
            dx0=dx0,
            est_err=est_err,
            camera_mode=config.camera_mode,
            q_acc=float(getattr(config, "q_acc", 1e-14)),
        )
        representative_plot = plots_dir / "06c_representative_trial_orbit.png"
        _plot_representative_orbit(
            rep_out,
            mu=float(config.mu),
            trial_id=trial_id,
            outpath=representative_plot,
        )

    print("=== 06C Summary " + "=" * 46)
    print(f"  trials            : {summary.get('n', n)}")
    print(f"  |Δdv|   median={_summary_stat('dv_delta_mag', 'median'):.3e}  "
          f"p95={_summary_stat('dv_delta_mag', 'p95'):.3e}")
    print(f"  bias    median={_summary_stat('dv_mag_bias',  'median'):+.3e}  "
          f"p95={_summary_stat('dv_mag_bias',  'p95'):+.3e}")
    infl_med = _summary_stat('dv_inflation_pct', 'median') * 100.0
    infl_p95 = _summary_stat('dv_inflation_pct', 'p95')    * 100.0
    print(f"  infl %  median={infl_med:.1f}%  p95={infl_p95:.1f}%")
    print(f"  miss    median={_summary_stat('miss_ekf', 'median'):.3e}  "
          f"p95={_summary_stat('miss_ekf', 'p95'):.3e}")
    if "success_rate" in summary:
        print(f"  success_rate      : {summary['success_rate']:.3f}"
              f"  (tol={args.tol:g})")
    print("=" * 62)

    print("\nOutputs:")
    print(f"  CSV  : {csv_path}")
    plot_names = [
        "06c_hist_dv_delta_mag.png",
        "06c_hist_dv_mag_bias.png",
        "06c_hist_dv_inflation_pct.png",
        "06c_hist_miss_ekf.png",
        "06c_hist_valid_measurement_rate.png",
        "06c_scatter_poserr_vs_dvdelta.png",
        "06c_scatter_validrate_vs_miss.png",
        "06c_scatter_nis_vs_miss.png",
    ]
    if not args.no_plot_d:
        plot_names.append("06c_scatter_traceP_vs_miss.png")
    if representative_plot is not None:
        plot_names.append(representative_plot.name)
    for name in plot_names:
        print(f"  Plot : {plots_dir / name}")


if __name__ == "__main__":
    main()
