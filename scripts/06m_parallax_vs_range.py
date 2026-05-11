"""Phase 3 — parallax vs range-error.

Bearing-only measurements *cannot* directly observe range; range becomes
observable only as the LOS-to-Moon vector sweeps an angular arc relative
to the trajectory. This script makes that physics visible:

  x-axis: accumulated angular change of LOS-to-Moon between t0 and tc
  y-axis: range error |‖r̂(tc)−r_moon‖ − ‖r(tc)−r_moon‖|

Multiple tc values are swept (0.5 → 5.0 ND) so each MC trial contributes
a series of (parallax, range_error) points across observation arc
lengths. The expected story: more parallax → smaller range error.

Usage
-----
python scripts/06m_parallax_vs_range.py --n-seeds 30 \
    --tc-list 0.25 0.5 1.0 1.5 2.0 3.0 4.0 5.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _analysis_common import (  # noqa: E402
    add_truth_arg,
    apply_dark_theme,
    apply_truth_suffix,
    load_midcourse_run_case,
    AMBER, BG, BORDER, CYAN, GREEN, PANEL, RED, TEXT, VIOLET,
)
from _common import ensure_src_on_path, repo_path

ensure_src_on_path()
from utils.units import RunUnits  # noqa: E402

from _paper_constants import KM_PER_LU as _KM_PER_LU


def _run_seed_grid(
    *, truth: str, tc_list: list[float], n_seeds: int,
    base_seed: int, config_kwargs: dict, n_workers: int,
) -> dict:
    from _parallel_seeds import run_seeds_parallel

    parallax_net: list[float] = []
    parallax_cum: list[float] = []
    range_err: list[float] = []
    pos_err:   list[float] = []
    miss:      list[float] = []
    tc_arr:    list[float] = []
    for tc in tc_list:
        rows = run_seeds_parallel(
            truth=truth, n_seeds=int(n_seeds), base_seed=int(base_seed),
            n_workers=int(n_workers),
            kwargs_extra={**config_kwargs, "tc": float(tc)},
            extract_fields=[
                ("parallax_net_rad",        "parallax_net_rad"),
                ("parallax_cumulative_rad", "parallax_cumulative_rad"),
                ("range_err_tc",            "range_err_tc"),
                ("pos_err_tc",              "pos_err_tc"),
                ("miss_ekf",                "miss_ekf"),
            ],
            extra_row_fields={"tc": float(tc)},
        )
        for r in rows:
            parallax_net.append(r["parallax_net_rad"])
            parallax_cum.append(r["parallax_cumulative_rad"])
            range_err.append(r["range_err_tc"])
            pos_err.append(r["pos_err_tc"])
            miss.append(r["miss_ekf"])
            tc_arr.append(r["tc"])
    return dict(
        parallax_net=np.asarray(parallax_net),
        parallax_cum=np.asarray(parallax_cum),
        range_err=np.asarray(range_err),
        pos_err=np.asarray(pos_err),
        miss=np.asarray(miss),
        tc=np.asarray(tc_arr),
    )


def _plot_parallax(
    data: dict, units: RunUnits, *, truth: str, n_seeds: int,
    outpath: Path,
) -> None:
    apply_dark_theme()
    # Cumulative parallax is the correct measure under multi-rev /
    # oscillatory LOS geometry; net parallax under-counts it. We plot
    # cumulative (primary) and report net in the summary file too.
    parallax_deg_cum = np.rad2deg(data["parallax_cum"])
    if units.truth == "cr3bp":
        range_err_km = data["range_err"] * _KM_PER_LU
        pos_err_km   = data["pos_err"]   * _KM_PER_LU
        miss_km      = data["miss"]      * _KM_PER_LU
    else:
        range_err_km = data["range_err"]
        pos_err_km   = data["pos_err"]
        miss_km      = data["miss"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    fig.patch.set_facecolor(BG)

    cmap = plt.get_cmap("plasma")
    tc_norm = (data["tc"] - data["tc"].min()) / max(np.ptp(data["tc"]), 1e-9)

    for ax, y, ylabel, title in [
        (axes[0], range_err_km, "range error  |Δrange|  [km]",
            "Parallax → Range Error"),
        (axes[1], pos_err_km, "pos_err at tc  [km]", "Parallax → Pos Error"),
        (axes[2], miss_km, "miss_ekf  [km]", "Parallax → Terminal Miss"),
    ]:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.grid(True, color=BORDER, lw=0.3)
        sc = ax.scatter(parallax_deg_cum, y, c=tc_norm, cmap=cmap, s=22,
                        alpha=0.85, edgecolors="none")
        # log-fit
        mask = (parallax_deg_cum > 0) & (y > 0) \
            & np.isfinite(parallax_deg_cum) & np.isfinite(y)
        if mask.sum() > 5:
            slope, intercept = np.polyfit(np.log10(parallax_deg_cum[mask]),
                                           np.log10(y[mask]), 1)
            xfit = np.linspace(parallax_deg_cum[mask].min(),
                               parallax_deg_cum[mask].max(), 100)
            yfit = 10 ** (slope * np.log10(xfit) + intercept)
            ax.plot(xfit, yfit, color=AMBER, ls="--", lw=1.3,
                    label=f"power fit  slope = {slope:.2f}")
            ax.legend(fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("cumulative LOS angular change  [deg]", color=TEXT)
        ax.set_ylabel(ylabel, color=TEXT)
        ax.set_title(title, color=TEXT, fontweight="bold")

    cb = fig.colorbar(sc, ax=axes[-1], pad=0.02, fraction=0.04)
    cb.set_label("tc (normalized)", color=TEXT)
    cb.outline.set_edgecolor(BORDER)

    fig.suptitle(
        f"Parallax vs Range / Position / Miss  ·  truth={truth}  "
        f"n_seeds={n_seeds} per tc",
        color=TEXT, fontsize=12,
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, facecolor=BG)
    plt.close(fig)


def _write_summary(data: dict, units: RunUnits, out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    parallax_deg_net = np.rad2deg(data["parallax_net"])
    parallax_deg_cum = np.rad2deg(data["parallax_cum"])
    range_err_km = data["range_err"] * (_KM_PER_LU if units.truth == "cr3bp" else 1.0)
    miss_km      = data["miss"]      * (_KM_PER_LU if units.truth == "cr3bp" else 1.0)
    lines = [
        "Parallax vs range error — bin medians",
        "Net parallax = endpoint-to-endpoint LOS angle (under-counts on "
        "multi-rev arcs).",
        "Cumulative parallax = sum of step-to-step LOS angle deltas "
        "(correct under any geometry).",
        "=" * 84,
        f"{'tc':>6}  {'plx_net [deg]':>15}  {'plx_cum [deg]':>15}  "
        f"{'range_err_med [km]':>20}  {'miss_med [km]':>15}",
    ]
    for tc in sorted(np.unique(data["tc"])):
        m = data["tc"] == tc
        lines.append(
            f"{tc:6g}  {np.nanmedian(parallax_deg_net[m]):15.3f}  "
            f"{np.nanmedian(parallax_deg_cum[m]):15.3f}  "
            f"{np.nanmedian(range_err_km[m]):20.3f}  "
            f"{np.nanmedian(miss_km[m]):15.3f}"
        )

    # Power-law fit summary against cumulative parallax (the better
    # geometric quantity); also report net for legacy comparison.
    for label, deg_arr in (("cumulative", parallax_deg_cum),
                            ("net",        parallax_deg_net)):
        mask = (deg_arr > 0) & (range_err_km > 0) \
            & np.isfinite(deg_arr) & np.isfinite(range_err_km)
        if mask.sum() > 5:
            slope, intercept = np.polyfit(np.log10(deg_arr[mask]),
                                           np.log10(range_err_km[mask]), 1)
            lines.append("")
            lines.append(
                f"power-law fit ({label} parallax):  "
                f"range_err ∝ parallax^{slope:.3f}  "
                f"(intercept = 10^{intercept:.2f})"
            )
    out_txt.write_text("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parallax (LOS sweep) vs range-error analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--tc-list", type=float, nargs="+",
                   default=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    p.add_argument("--mu",   type=float, default=0.0121505856)
    p.add_argument("--t0",   type=float, default=0.0)
    p.add_argument("--tf",   type=float, default=6.0)
    p.add_argument("--dt-meas",  type=float, default=0.02)
    p.add_argument("--sigma-px", type=float, default=1.0)
    p.add_argument("--q-acc", type=float, default=1e-14)
    p.add_argument("--out", type=str, default="results/mc/parallax_vs_range")
    p.add_argument("--base-seed", type=int, default=7)
    p.add_argument("--n-workers", type=int, default=-1,
                   help="Process-pool size; -1 = cpu_count(); 1 = serial.")
    add_truth_arg(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    apply_dark_theme()
    units = RunUnits.for_truth(args.truth)

    config_kwargs = dict(
        mu=args.mu, t0=args.t0, tf=args.tf,
        dt_meas=args.dt_meas, sigma_px=args.sigma_px, dropout_prob=0.0,
        camera_mode="estimate_tracking", q_acc=args.q_acc,
    )

    print(f"▸ tc grid: {args.tc_list}  n_seeds/tc: {args.n_seeds}  "
          f"truth: {args.truth}  workers: {args.n_workers}")
    data = _run_seed_grid(
        truth=str(args.truth), tc_list=list(args.tc_list),
        n_seeds=int(args.n_seeds), base_seed=int(args.base_seed),
        config_kwargs=config_kwargs, n_workers=int(args.n_workers),
    )

    out_dir = apply_truth_suffix(repo_path(args.out), args.truth)
    _plot_parallax(data, units, truth=args.truth,
                   n_seeds=int(args.n_seeds),
                   outpath=out_dir / "06m_parallax_vs_range.png")
    _write_summary(data, units, out_dir / "06m_parallax_vs_range.txt")
    print("\nWrote:")
    print(f"  {out_dir / '06m_parallax_vs_range.png'}")
    print(f"  {out_dir / '06m_parallax_vs_range.txt'}")


if __name__ == "__main__":
    main()
