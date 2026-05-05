"""Render slide-13 sensitivity figures (FIG 13A, FIG 13B) from a CSV.

Produces two square, presentation-grade figures sized for the ~5.6 × 5.8 in
panel placeholders on slide 13:

  - 13a_miss_vs_sigma.png  ← FIG 13A · MISS VS Σ_PIX
  - 13b_miss_vs_tc.png     ← FIG 13B · MISS VS T_C

Both are dark-themed and use the same in-panel-header / suptitle pattern as
slides 9 and 11.  Highlights the calibrated baseline (σ=1, tc=2) with a
violet vertical marker so the slide tells "this is where we live, here is
how it degrades around it."

Usage:
    python scripts/_render_slide13_figures.py [--csv PATH] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _common import ensure_src_on_path, repo_path

ensure_src_on_path()
from visualization.style import (  # noqa: E402
    AMBER, BG, BORDER, CYAN, GREEN, PANEL, RED, TEXT, VIOLET,
    apply_dark_theme,
)


def _read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        clean = {}
        for k, v in r.items():
            if v == "" or v is None:
                clean[k] = float("nan")
            else:
                try:
                    clean[k] = float(v)
                except ValueError:
                    clean[k] = v  # keep strings (sweep, camera_mode)
        out.append(clean)
    return out


def _filter(rows, sweep: str) -> list[dict]:
    return [r for r in rows if r.get("sweep") == sweep]


def _make_panel_figure(
    rows: list[dict],
    *,
    x_key: str,
    xlabel: str,
    title: str,
    subtitle: str,
    baseline_x: float,
    baseline_label: str,
    out_path: Path,
    n_seeds: int,
    q_acc: float,
) -> None:
    rows = sorted(rows, key=lambda r: r[x_key])
    x   = np.array([r[x_key]              for r in rows], dtype=float)
    med = np.array([r["miss_ekf_median"]  for r in rows], dtype=float)
    lo  = np.array([r["miss_ekf_p05"]     for r in rows], dtype=float)
    hi  = np.array([r["miss_ekf_p95"]     for r in rows], dtype=float)

    # ── Figure 7×7 (matches the ~5.6×5.8 in panel) ───────────────────────
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=BG)
    fig.subplots_adjust(left=0.155, right=0.96, top=0.86, bottom=0.16)

    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, alpha=0.45, ls="--", lw=0.7, which="both")

    # P05–P95 band
    ax.fill_between(x, lo, hi, color=VIOLET, alpha=0.22,
                    label="P05 – P95 band", zorder=2)

    # Median line + markers
    ax.plot(x, med, "-o", color=AMBER, lw=2.4, ms=8,
            markeredgecolor=TEXT, markeredgewidth=0.7,
            label="median miss", zorder=4)

    # Baseline marker (calibrated operating point)
    ax.axvline(baseline_x, color=CYAN, lw=1.6, ls=":", alpha=0.85, zorder=3)

    # Use log-y if range spans more than ~1.5 decades — sensitivity bands
    # are heavy-tailed.  Otherwise linear.
    finite = np.concatenate([med[np.isfinite(med)], hi[np.isfinite(hi)]])
    if finite.size and finite.max() / max(finite.min(), 1e-12) > 30:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel, fontweight="bold", fontsize=12)
    ax.set_ylabel("Terminal miss   [dimensionless CR3BP length]",
                  fontweight="bold", fontsize=11)

    # Baseline annotation — anchored to the bottom so it can't collide
    # with the upper-right legend on either subplot.
    ax.text(baseline_x, 0.025, f"  {baseline_label}",
            transform=ax.get_xaxis_transform(),
            color=CYAN, fontsize=10, fontweight="bold",
            ha="left", va="bottom")

    # In-panel header (left-aligned, deck style)
    ax.text(0.005, 1.075, title, transform=ax.transAxes,
            color=TEXT, fontweight="bold", fontsize=14,
            ha="left", va="bottom")
    ax.text(0.005, 1.025, subtitle, transform=ax.transAxes,
            color=TEXT, fontsize=10.5, ha="left", va="bottom", alpha=0.85)

    # Footer line — provenance
    ax.text(0.50, -0.135,
            f"n_seeds = {n_seeds} per point   ·   q_acc = {q_acc:.0e}",
            transform=ax.transAxes,
            color=TEXT, fontsize=9.5, ha="center", va="top", alpha=0.65)

    leg = ax.legend(loc="upper right", facecolor=PANEL, edgecolor=BORDER,
                    labelcolor=TEXT, framealpha=0.95, fontsize=10,
                    handlelength=1.8, borderaxespad=0.6)
    leg.get_frame().set_linewidth(0.6)

    fig.savefig(out_path, dpi=210, facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv",
                   default="results/mc/sensitivity_n80/06b_sensitivity.csv",
                   help="Sensitivity CSV (falls back to existing tuned data if missing).")
    p.add_argument("--out-dir",
                   default="results/mc/sensitivity_n80",
                   help="Where to write 13a_*.png and 13b_*.png.")
    args = p.parse_args()

    apply_dark_theme()

    csv_path = repo_path(args.csv)
    if not csv_path.exists():
        fallback = repo_path("results/mc/sensitivity_tuned/06b_sensitivity.csv")
        print(f"⚠ {csv_path} not found — falling back to {fallback}")
        csv_path = fallback

    rows = _read_rows(csv_path)
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")

    n_seeds = int(rows[0].get("n_seeds", 0))
    # We don't store q_acc in the CSV; the n80 directory is generated with
    # --q-acc 1e-9, the legacy tuned dir was 1e-14.  Default to 1e-9 if the
    # path indicates the n80 run.
    q_acc = 1e-9 if "n80" in str(csv_path) else 1e-14

    sigma_rows = _filter(rows, "sigma")
    tc_rows    = _filter(rows, "tc")

    out_dir = repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if sigma_rows:
        _make_panel_figure(
            sigma_rows,
            x_key="sigma_px",
            xlabel="σ_pix   [pixels]",
            title="Terminal miss vs pixel noise",
            subtitle="Median + P05–P95 band   ·   tc = 2.0   ·   active pointing",
            baseline_x=1.0,
            baseline_label="baseline σ = 1 px",
            out_path=out_dir / "13a_miss_vs_sigma.png",
            n_seeds=n_seeds, q_acc=q_acc,
        )
    else:
        print("⚠ no σ-sweep rows found")

    if tc_rows:
        _make_panel_figure(
            tc_rows,
            x_key="tc",
            xlabel="Correction time t_c   [dimensionless CR3BP time]",
            title="Terminal miss vs correction time",
            subtitle="Median + P05–P95 band   ·   σ_pix = 1.5   ·   active pointing",
            baseline_x=2.0,
            baseline_label="baseline tc = 2.0",
            out_path=out_dir / "13b_miss_vs_tc.png",
            n_seeds=n_seeds, q_acc=q_acc,
        )
    else:
        print("⚠ no tc-sweep rows found")


if __name__ == "__main__":
    main()
