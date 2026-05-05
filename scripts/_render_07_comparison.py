"""Re-render the slide-9 active-vs-fixed comparison from cached .npz.

Same numbers as 07_active_tracking.py's `make_comparison_plots`, but with
the slide-deck design language: in-panel headers, side stats panel, suptitle,
and a tighter layout that reads cleanly in a slide thumbnail.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

from _common import ensure_src_on_path, repo_path

ensure_src_on_path()
from visualization.style import (  # noqa: E402
    AMBER, BG, BORDER, CYAN, GREEN, PANEL, RED, TEXT, VIOLET,
    apply_dark_theme,
)


def _load(case: str) -> dict:
    p = repo_path(f"results/active_tracking/07_active_tracking_{case}.npz")
    return dict(np.load(p, allow_pickle=True))


def _norms(rec: dict) -> tuple[np.ndarray, np.ndarray]:
    xhat = np.asarray(rec["xhat_hist"], dtype=float)
    xtru = np.asarray(rec["x_true_hist"], dtype=float)
    pos = np.linalg.norm(xhat[:, :3] - xtru[:, :3], axis=1)
    vel = np.linalg.norm(xhat[:, 3:6] - xtru[:, 3:6], axis=1)
    return pos, vel


def main() -> None:
    apply_dark_theme()

    A = _load("active")
    F = _load("fixed")

    ta = np.asarray(A["t_hist"], dtype=float)
    tf = np.asarray(F["t_hist"], dtype=float)
    pa, va = _norms(A)
    pf, vf = _norms(F)

    vis_a = np.asarray(A["visible_hist"], dtype=bool).astype(int)
    vis_f = np.asarray(F["visible_hist"], dtype=bool).astype(int)
    upd_a = np.asarray(A["update_used_hist"], dtype=bool).astype(int)
    upd_f = np.asarray(F["update_used_hist"], dtype=bool).astype(int)

    nis_a = np.asarray(A["nis_hist"], dtype=float)
    nis_f = np.asarray(F["nis_hist"], dtype=float)

    nis_lo, nis_hi = float(chi2.ppf(0.025, df=2)), float(chi2.ppf(0.975, df=2))

    vis_frac_a = float(A["visibility_fraction"])
    vis_frac_f = float(F["visibility_fraction"])
    rms_a = float(A["rms_position_error"])
    rms_f = float(F["rms_position_error"])
    nis_in_band_a = float(np.mean((nis_a[np.isfinite(nis_a)] >= nis_lo)
                                  & (nis_a[np.isfinite(nis_a)] <= nis_hi)))
    nis_in_band_f = float(np.mean((nis_f[np.isfinite(nis_f)] >= nis_lo)
                                  & (nis_f[np.isfinite(nis_f)] <= nis_hi)))

    # ─── Figure ──────────────────────────────────────────────────────────
    # Wide 14×9 with a 3-row grid: visibility (compact), pos/vel error
    # (semilog dominant), NIS (compact).  Right gutter holds a stats card.
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        height_ratios=[0.55, 1.6, 0.85], width_ratios=[3.5, 1.0],
        left=0.06, right=0.985, top=0.90, bottom=0.085,
        hspace=0.22, wspace=0.04,
    )

    axV = fig.add_subplot(gs[0, 0])
    axE = fig.add_subplot(gs[1, 0], sharex=axV)
    axN = fig.add_subplot(gs[2, 0], sharex=axV)
    axS = fig.add_subplot(gs[:, 1])
    axS.axis("off")

    for ax in (axV, axE, axN):
        ax.set_facecolor(PANEL)
        ax.grid(True, color=BORDER, alpha=0.45, ls="--", lw=0.7)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

    # ── Visibility panel ────────────────────────────────────────────────
    axV.step(ta, vis_a + 0.02, where="post", color=CYAN,  lw=1.6,
             label=f"active ({vis_frac_a*100:.1f}%)")
    axV.step(tf, vis_f - 0.02, where="post", color=AMBER, lw=1.6,
             label=f"fixed ({vis_frac_f*100:.1f}%)", alpha=0.95)
    axV.set_yticks([0, 1])
    axV.set_yticklabels(["off", "on"])
    axV.set_ylim(-0.15, 1.18)
    axV.set_ylabel("Moon visible", fontweight="bold", fontsize=10)
    axV.tick_params(labelbottom=False)
    axV.text(0.005, 1.10,
             "Visibility — fixed camera loses the Moon at t ≈ 1.1; active stays on it",
             transform=axV.transAxes, color=TEXT, fontweight="bold",
             fontsize=11.5, ha="left", va="bottom")
    axV.legend(loc="lower right", fontsize=9, ncol=2, framealpha=0.9,
               handlelength=1.6, columnspacing=1.0)

    # ── Position-error panel (semilog) ──────────────────────────────────
    axE.semilogy(ta, pa + 1e-12, color=CYAN,  lw=1.9, label="active ‖pos err‖")
    axE.semilogy(tf, pf + 1e-12, color=AMBER, lw=1.9, ls="-",
                 alpha=0.95, label="fixed ‖pos err‖")
    axE.semilogy(ta, va + 1e-12, color=CYAN,  lw=1.0, ls=":", alpha=0.65,
                 label="active ‖vel err‖")
    axE.semilogy(tf, vf + 1e-12, color=AMBER, lw=1.0, ls=":", alpha=0.65,
                 label="fixed ‖vel err‖")
    axE.set_ylabel("Error norm  [dimensionless CR3BP units]",
                   fontweight="bold", fontsize=10)
    axE.tick_params(labelbottom=False)
    axE.text(0.005, 1.03,
             f"Error norms — fixed climbs to {pf[-1]:.1e}; active holds {pa[-1]:.1e}",
             transform=axE.transAxes, color=TEXT, fontweight="bold",
             fontsize=11.5, ha="left", va="bottom")
    axE.legend(loc="lower left", fontsize=9, ncol=2, framealpha=0.9,
               handlelength=1.8, columnspacing=1.2)

    # ── NIS panel ────────────────────────────────────────────────────────
    axN.axhspan(nis_lo, nis_hi, color=GREEN, alpha=0.12, zorder=0)
    axN.axhline(2.0, color=TEXT, lw=0.8, ls="--", alpha=0.5, zorder=1)
    nis_ok_a = np.isfinite(nis_a); nis_ok_f = np.isfinite(nis_f)
    axN.scatter(ta[nis_ok_a], nis_a[nis_ok_a], s=11, c=CYAN, alpha=0.85,
                label="active NIS", zorder=3, edgecolors="none")
    axN.scatter(tf[nis_ok_f], nis_f[nis_ok_f], s=11, c=AMBER, alpha=0.85,
                label="fixed NIS",  zorder=3, edgecolors="none")
    axN.set_ylabel("NIS", fontweight="bold", fontsize=10)
    axN.set_xlabel("t   [dimensionless CR3BP time]",
                   fontweight="bold", fontsize=10)
    axN.set_ylim(0, 12)
    axN.text(0.005, 1.06,
             f"NIS — active stays inside χ²(2) 95% gate [{nis_lo:.2f}, {nis_hi:.2f}]",
             transform=axN.transAxes, color=TEXT, fontweight="bold",
             fontsize=11.5, ha="left", va="bottom")
    axN.text(0.985, nis_hi, "  χ²(2) gate",
             transform=axN.get_yaxis_transform(), color=GREEN, fontsize=9,
             ha="right", va="bottom", fontweight="bold", alpha=0.9)
    axN.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.9,
               handlelength=1.5, columnspacing=1.0)

    # ── Stats card (right gutter) ────────────────────────────────────────
    def _row(y, label, val_a, val_f, *, unit=""):
        axS.text(0.04, y, label, color=TEXT, fontsize=9.5,
                 fontweight="bold", transform=axS.transAxes,
                 ha="left", va="top", alpha=0.75)
        axS.text(0.04, y - 0.034, f"{val_a}", color=CYAN, fontsize=14,
                 fontweight="bold", transform=axS.transAxes,
                 ha="left", va="top")
        axS.text(0.96, y - 0.034, f"{val_f}", color=AMBER, fontsize=14,
                 fontweight="bold", transform=axS.transAxes,
                 ha="right", va="top")
        if unit:
            axS.text(0.50, y - 0.034, unit, color=TEXT, fontsize=9,
                     transform=axS.transAxes,
                     ha="center", va="top", alpha=0.65)

    # header
    axS.text(0.04, 0.985, "ACTIVE", color=CYAN, fontsize=10,
             fontweight="bold", transform=axS.transAxes, ha="left", va="top")
    axS.text(0.96, 0.985, "FIXED", color=AMBER, fontsize=10,
             fontweight="bold", transform=axS.transAxes, ha="right", va="top")
    axS.plot([0.02, 0.98], [0.965, 0.965],
             transform=axS.transAxes, color=BORDER, lw=0.8)

    _row(0.92,  "VISIBILITY",       f"{vis_frac_a*100:.1f}%", f"{vis_frac_f*100:.1f}%")
    _row(0.78,  "RMS POS ERR",      f"{rms_a:.1e}",           f"{rms_f:.1e}", unit="ND")
    _row(0.64,  "FINAL POS ERR",    f"{pa[-1]:.1e}",          f"{pf[-1]:.1e}", unit="ND")
    _row(0.50,  "NIS IN-GATE",      f"{nis_in_band_a*100:.0f}%", f"{nis_in_band_f*100:.0f}%")

    # bottom line: headline
    axS.plot([0.02, 0.98], [0.34, 0.34],
             transform=axS.transAxes, color=BORDER, lw=0.8)
    axS.text(0.50, 0.30, "RMS RATIO", color=TEXT, fontsize=9.5,
             fontweight="bold", transform=axS.transAxes,
             ha="center", va="top", alpha=0.75)
    ratio = rms_f / max(rms_a, 1e-15)
    axS.text(0.50, 0.225, f"{ratio:,.0f}×", color=GREEN, fontsize=28,
             fontweight="bold", transform=axS.transAxes,
             ha="center", va="top")
    axS.text(0.50, 0.115, "fixed ÷ active", color=TEXT, fontsize=9,
             transform=axS.transAxes,
             ha="center", va="top", alpha=0.75)

    # ── Suptitle ─────────────────────────────────────────────────────────
    fig.suptitle(
        f"Active-Pointing vs Fixed-Camera  ·  6.56 d arc  ·  {len(ta)} steps  ·  σ_px = 1 px",
        color=TEXT, fontweight="bold", fontsize=15, y=0.965,
    )

    out = repo_path("results/active_tracking/07_active_tracking_fixed_vs_active_comparison.png")
    fig.savefig(out, dpi=210, facecolor=BG)
    plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
