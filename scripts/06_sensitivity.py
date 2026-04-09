from __future__ import annotations

from pathlib import Path
import importlib.util
import csv
import numpy as np
import matplotlib.pyplot as plt


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


def _load_run_case():
    here  = Path(__file__).resolve()
    cand  = here.parent / "06_midcourse_ekf_correction.py"
    if not cand.exists():
        raise FileNotFoundError(f"Expected 06A script at: {cand}")
    spec = importlib.util.spec_from_file_location("midcourse06a", cand)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {cand}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run_case"):
        raise AttributeError(f"{cand} does not define run_case(...)")
    return mod.run_case


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_xy(
    x: np.ndarray,
    y: np.ndarray,
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

    ax.plot(x, y, color=color, lw=2.0, zorder=3)
    ax.scatter(x, y, s=50, color=marker_color, zorder=4, edgecolors=_BG, lw=0.5)
    ax.set_xlabel(xlabel, color=_TEXT)
    ax.set_ylabel(ylabel, color=_TEXT)
    ax.set_title(title, color=_TEXT)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, facecolor=_BG)
    plt.close(fig)


def main() -> None:
    _apply_dark_theme()
    run_case = _load_run_case()

    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    mu       = 0.0121505856
    t0       = 0.0
    tf       = 6.0
    dt_meas  = 0.02
    seed     = 7

    dx0     = np.array([2e-4, -1e-4, 0.0, 0.0,  2e-3,   0.0], dtype=float)
    est_err = np.array([3e-4,  2e-4, 0.0, 0.0, -1.5e-3, 0.0], dtype=float)

    rows: list[dict] = []

    tc_fixed     = 2.0
    dropout_prob = 0.0
    sigma_grid   = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    for sigma_px in sigma_grid:
        out = run_case(
            mu, t0, tf, tc_fixed, dt_meas,
            float(sigma_px), float(dropout_prob), int(seed),
            dx0, est_err,
            camera_mode="estimate_tracking",
        )
        rows.append({
            "sweep": "sigma", "bonus": 0,
            "sigma_px":              float(sigma_px),
            "tc":                    float(out["tc"]),
            "dropout_prob":          float(dropout_prob),
            "camera_mode":           out["camera_mode"],
            "dv_perfect":            float(out["dv_perfect_mag"]),
            "dv_ekf":                float(out["dv_ekf_mag"]),
            "dv_delta":              float(out["dv_delta_mag"]),
            "dv_inflation":          float(out["dv_inflation"]),
            "dv_inflation_pct":      float(out["dv_inflation_pct"]),
            "miss_unc":              float(out["miss_uncorrected"]),
            "miss_perf":             float(out["miss_perfect"]),
            "miss_ekf":              float(out["miss_ekf"]),
            "pos_err_tc":            float(out["pos_err_tc"]),
            "tracePpos_tc":          float(out["tracePpos_tc"]),
            "nis_mean":              float(out["nis_mean"]),
            "valid_rate":            float(out["valid_rate"]),
        })

    _sel = lambda s, b: [r for r in rows if r["sweep"] == s and r["bonus"] == b]

    s0 = _sel("sigma", 0)
    sigma_arr = np.array([r["sigma_px"]    for r in s0], dtype=float)
    dv_infl   = np.array([r["dv_inflation"] for r in s0], dtype=float)
    miss_ekf  = np.array([r["miss_ekf"]    for r in s0], dtype=float)
    poserr_tc = np.array([r["pos_err_tc"]  for r in s0], dtype=float)

    _plot_xy(sigma_arr, dv_infl,
             xlabel="σ_px  [px]",
             ylabel="ΔV inflation  (|dv_ekf| − |dv_perfect|)  [ND]",
             title="ΔV Inflation vs Pixel Noise  (tc fixed)",
             outpath=plots_dir / "06b_dv_inflation_vs_sigma.png",
             color=_CYAN, marker_color=_AMBER)
    _plot_xy(sigma_arr, miss_ekf,
             xlabel="σ_px  [px]",
             ylabel="Terminal miss  [ND]",
             title="Terminal Miss vs Pixel Noise  (tc fixed)",
             outpath=plots_dir / "06b_miss_vs_sigma.png",
             color=_VIOLET, marker_color=_AMBER)
    _plot_xy(sigma_arr, poserr_tc,
             xlabel="σ_px  [px]",
             ylabel="‖r̂(tc) − r(tc)‖  [ND]",
             title="Position Error at tc vs Pixel Noise  (tc fixed)",
             outpath=plots_dir / "06b_poserr_tc_vs_sigma.png",
             color=_GREEN, marker_color=_AMBER)

    sigma_fixed = 1.5
    tc_grid     = [0.8, 1.2, 1.6, 2.0, 2.5, 3.0]

    for tc in tc_grid:
        out = run_case(
            mu, t0, tf, float(tc), dt_meas,
            float(sigma_fixed), float(dropout_prob), int(seed),
            dx0, est_err,
            camera_mode="estimate_tracking",
        )
        rows.append({
            "sweep": "tc", "bonus": 0,
            "sigma_px":              float(sigma_fixed),
            "tc":                    float(out["tc"]),
            "dropout_prob":          float(dropout_prob),
            "camera_mode":         out["camera_mode"],
            "dv_perfect":            float(out["dv_perfect_mag"]),
            "dv_ekf":                float(out["dv_ekf_mag"]),
            "dv_delta":              float(out["dv_delta_mag"]),
            "dv_inflation":          float(out["dv_inflation"]),
            "dv_inflation_pct":      float(out["dv_inflation_pct"]),
            "miss_unc":              float(out["miss_uncorrected"]),
            "miss_perf":             float(out["miss_perfect"]),
            "miss_ekf":              float(out["miss_ekf"]),
            "pos_err_tc":            float(out["pos_err_tc"]),
            "tracePpos_tc":          float(out["tracePpos_tc"]),
            "nis_mean":              float(out["nis_mean"]),
            "valid_rate":            float(out["valid_rate"]),
        })

    t0_sel = _sel("tc", 0)
    tc_arr       = np.array([r["tc"]           for r in t0_sel], dtype=float)
    dv_infl_tc   = np.array([r["dv_inflation"]  for r in t0_sel], dtype=float)
    miss_ekf_tc  = np.array([r["miss_ekf"]      for r in t0_sel], dtype=float)
    poserr_tc_tc = np.array([r["pos_err_tc"]    for r in t0_sel], dtype=float)

    _plot_xy(tc_arr, dv_infl_tc,
             xlabel="Correction time  tc  [ND]",
             ylabel="ΔV inflation  [ND]",
             title="ΔV Inflation vs Correction Time  (σ_px fixed)",
             outpath=plots_dir / "06b_dv_inflation_vs_tc.png",
             color=_CYAN, marker_color=_AMBER)
    _plot_xy(tc_arr, miss_ekf_tc,
             xlabel="Correction time  tc  [ND]",
             ylabel="Terminal miss  [ND]",
             title="Terminal Miss vs Correction Time  (σ_px fixed)",
             outpath=plots_dir / "06b_miss_vs_tc.png",
             color=_VIOLET, marker_color=_AMBER)
    _plot_xy(tc_arr, poserr_tc_tc,
             xlabel="Correction time  tc  [ND]",
             ylabel="‖r̂(tc) − r(tc)‖  [ND]",
             title="Position Error at tc vs Correction Time  (σ_px fixed)",
             outpath=plots_dir / "06b_poserr_tc_vs_tc.png",
             color=_GREEN, marker_color=_AMBER)

    dropout_prob_bonus = 0.05
    tc_bonus           = 2.0

    for sigma_px in sigma_grid:
        out = run_case(
            mu, t0, tf, float(tc_bonus), dt_meas,
            float(sigma_px), float(dropout_prob_bonus), int(seed),
            dx0, est_err,
            camera_mode="fixed",
        )
        rows.append({
            "sweep": "sigma", "bonus": 1,
            "sigma_px":              float(sigma_px),
            "tc":                    float(out["tc"]),
            "dropout_prob":          float(dropout_prob_bonus),
            "camera_mode":         out["camera_mode"],
            "dv_perfect":            float(out["dv_perfect_mag"]),
            "dv_ekf":                float(out["dv_ekf_mag"]),
            "dv_delta":              float(out["dv_delta_mag"]),
            "dv_inflation":          float(out["dv_inflation"]),
            "dv_inflation_pct":      float(out["dv_inflation_pct"]),
            "miss_unc":              float(out["miss_uncorrected"]),
            "miss_perf":             float(out["miss_perfect"]),
            "miss_ekf":              float(out["miss_ekf"]),
            "pos_err_tc":            float(out["pos_err_tc"]),
            "tracePpos_tc":          float(out["tracePpos_tc"]),
            "nis_mean":              float(out["nis_mean"]),
            "valid_rate":            float(out["valid_rate"]),
        })

    s1 = _sel("sigma", 1)
    sigma_bonus   = np.array([r["sigma_px"]    for r in s1], dtype=float)
    dv_infl_bonus = np.array([r["dv_inflation"] for r in s1], dtype=float)
    miss_bonus    = np.array([r["miss_ekf"]     for r in s1], dtype=float)
    poserr_bonus  = np.array([r["pos_err_tc"]   for r in s1], dtype=float)

    _plot_xy(sigma_bonus, dv_infl_bonus,
             xlabel="σ_px  [px]",
             ylabel="ΔV inflation  [ND]",
             title="ΔV Inflation vs σ_px  (dropout=0.05, fixed pointing)",
             outpath=plots_dir / "06b_bonus_dv_inflation_vs_sigma.png",
             color=_RED, marker_color=_AMBER)
    _plot_xy(sigma_bonus, miss_bonus,
             xlabel="σ_px  [px]",
             ylabel="Terminal miss  [ND]",
             title="Terminal Miss vs σ_px  (dropout=0.05, fixed pointing)",
             outpath=plots_dir / "06b_bonus_miss_vs_sigma.png",
             color=_RED, marker_color=_AMBER)
    _plot_xy(sigma_bonus, poserr_bonus,
             xlabel="σ_px  [px]",
             ylabel="‖r̂(tc) − r(tc)‖  [ND]",
             title="Position Error at tc vs σ_px  (dropout=0.05, fixed pointing)",
             outpath=plots_dir / "06b_bonus_poserr_tc_vs_sigma.png",
             color=_RED, marker_color=_AMBER)

    csv_path = plots_dir / "06b_sensitivity.csv"
    _write_csv(csv_path, rows)

    print("06B sensitivity complete.")
    print(f"Wrote CSV: {csv_path}")
    print("Wrote plots:")
    for fname in [
        "06b_dv_inflation_vs_sigma.png", "06b_miss_vs_sigma.png",
        "06b_poserr_tc_vs_sigma.png",
        "06b_dv_inflation_vs_tc.png", "06b_miss_vs_tc.png",
        "06b_poserr_tc_vs_tc.png",
        "06b_bonus_dv_inflation_vs_sigma.png", "06b_bonus_miss_vs_sigma.png",
        "06b_bonus_poserr_tc_vs_sigma.png",
    ]:
        print(f"  - {plots_dir / fname}")


if __name__ == "__main__":
    main()
