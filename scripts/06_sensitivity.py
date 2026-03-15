from __future__ import annotations

from pathlib import Path
import importlib.util
import csv
import numpy as np
import matplotlib.pyplot as plt


def _load_run_case():
    """
    Load run_case() from scripts/06_midcourse_ekf_correction.py.

    We use importlib because filenames starting with digits aren't importable
    as regular Python modules.
    """
    here = Path(__file__).resolve()
    cand = here.parent / "06_midcourse_ekf_correction.py"
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
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main() -> None:
    run_case = _load_run_case()

    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    mu = 0.0121505856
    t0 = 0.0
    tf = 6.0
    dt_meas = 0.02
    seed = 7

    # fixed errors (so trends are attributable to sigma/tc)
    dx0 = np.array([2e-4, -1e-4, 0.0, 0.0, 2e-3, 0.0], dtype=float)
    est_err = np.array([3e-4, 2e-4, 0.0, 0.0, -1.5e-3, 0.0], dtype=float)

    rows: list[dict] = []

    tc_fixed = 2.0
    dropout_prob = 0.0
    sigma_grid = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    for sigma_px in sigma_grid:
        out = run_case(
            mu, t0, tf, tc_fixed, dt_meas,
            float(sigma_px), float(dropout_prob), int(seed),
            dx0, est_err,
            fixed_camera_pointing=False,
        )

        rows.append({
            "sweep": "sigma",
            "bonus": 0,
            "sigma_px": float(sigma_px),
            "tc": float(out["tc"]),  # tc_eff returned by 06A
            "dropout_prob": float(dropout_prob),
            "fixed_camera_pointing": int(bool(out.get("fixed_camera_pointing", False))),
            "dv_perfect": float(out["dv_perfect_mag"]),
            "dv_ekf": float(out["dv_ekf_mag"]),
            "dv_delta": float(out["dv_delta_mag"]),
            "dv_inflation": float(out["dv_inflation"]),
            "dv_inflation_pct": float(out["dv_inflation_pct"]),
            "miss_unc": float(out["miss_uncorrected"]),
            "miss_perf": float(out["miss_perfect"]),
            "miss_ekf": float(out["miss_ekf"]),
            "pos_err_tc": float(out["pos_err_tc"]),
            "tracePpos_tc": float(out["tracePpos_tc"]),
            "nis_mean": float(out["nis_mean"]),
            "valid_rate": float(out["valid_rate"]),
        })

    sigma_arr = np.array([r["sigma_px"] for r in rows if r["sweep"] == "sigma" and r["bonus"] == 0], dtype=float)
    dv_infl = np.array([r["dv_inflation"] for r in rows if r["sweep"] == "sigma" and r["bonus"] == 0], dtype=float)
    miss_ekf = np.array([r["miss_ekf"] for r in rows if r["sweep"] == "sigma" and r["bonus"] == 0], dtype=float)
    poserr_tc = np.array([r["pos_err_tc"] for r in rows if r["sweep"] == "sigma" and r["bonus"] == 0], dtype=float)

    _plot_xy(
        sigma_arr, dv_infl,
        xlabel="sigma_px [px]",
        ylabel="ΔV inflation (|dv_ekf| - |dv_perfect|)",
        title="06B: ΔV inflation vs sigma_px (tc fixed)",
        outpath=plots_dir / "06b_dv_inflation_vs_sigma.png",
    )
    _plot_xy(
        sigma_arr, miss_ekf,
        xlabel="sigma_px [px]",
        ylabel="miss_ekf at tf",
        title="06B: terminal miss vs sigma_px (tc fixed)",
        outpath=plots_dir / "06b_miss_vs_sigma.png",
    )
    _plot_xy(
        sigma_arr, poserr_tc,
        xlabel="sigma_px [px]",
        ylabel="||r_hat(tc) - r_true(tc)||",
        title="06B: position error at tc vs sigma_px (tc fixed)",
        outpath=plots_dir / "06b_poserr_tc_vs_sigma.png",
    )

    sigma_fixed = 1.5
    tc_grid = [0.8, 1.2, 1.6, 2.0, 2.5, 3.0]

    for tc in tc_grid:
        out = run_case(
            mu, t0, tf, float(tc), dt_meas,
            float(sigma_fixed), float(dropout_prob), int(seed),
            dx0, est_err,
            fixed_camera_pointing=False,
        )

        rows.append({
            "sweep": "tc",
            "bonus": 0,
            "sigma_px": float(sigma_fixed),
            "tc": float(out["tc"]),
            "dropout_prob": float(dropout_prob),
            "fixed_camera_pointing": int(bool(out.get("fixed_camera_pointing", False))),
            "dv_perfect": float(out["dv_perfect_mag"]),
            "dv_ekf": float(out["dv_ekf_mag"]),
            "dv_delta": float(out["dv_delta_mag"]),
            "dv_inflation": float(out["dv_inflation"]),
            "dv_inflation_pct": float(out["dv_inflation_pct"]),
            "miss_unc": float(out["miss_uncorrected"]),
            "miss_perf": float(out["miss_perfect"]),
            "miss_ekf": float(out["miss_ekf"]),
            "pos_err_tc": float(out["pos_err_tc"]),
            "tracePpos_tc": float(out["tracePpos_tc"]),
            "nis_mean": float(out["nis_mean"]),
            "valid_rate": float(out["valid_rate"]),
        })

    tc_arr = np.array([r["tc"] for r in rows if r["sweep"] == "tc" and r["bonus"] == 0], dtype=float)
    dv_infl_tc = np.array([r["dv_inflation"] for r in rows if r["sweep"] == "tc" and r["bonus"] == 0], dtype=float)
    miss_ekf_tc = np.array([r["miss_ekf"] for r in rows if r["sweep"] == "tc" and r["bonus"] == 0], dtype=float)
    poserr_tc_tc = np.array([r["pos_err_tc"] for r in rows if r["sweep"] == "tc" and r["bonus"] == 0], dtype=float)

    _plot_xy(
        tc_arr, dv_infl_tc,
        xlabel="tc [time units]",
        ylabel="ΔV inflation (|dv_ekf| - |dv_perfect|)",
        title="06B: ΔV inflation vs tc (sigma fixed)",
        outpath=plots_dir / "06b_dv_inflation_vs_tc.png",
    )
    _plot_xy(
        tc_arr, miss_ekf_tc,
        xlabel="tc [time units]",
        ylabel="miss_ekf at tf",
        title="06B: terminal miss vs tc (sigma fixed)",
        outpath=plots_dir / "06b_miss_vs_tc.png",
    )
    _plot_xy(
        tc_arr, poserr_tc_tc,
        xlabel="tc [time units]",
        ylabel="||r_hat(tc) - r_true(tc)||",
        title="06B: position error at tc vs tc (sigma fixed)",
        outpath=plots_dir / "06b_poserr_tc_vs_tc.png",
    )

    dropout_prob_bonus = 0.05
    tc_bonus = 2.0

    for sigma_px in sigma_grid:
        out = run_case(
            mu, t0, tf, float(tc_bonus), dt_meas,
            float(sigma_px), float(dropout_prob_bonus), int(seed),
            dx0, est_err,
            fixed_camera_pointing=True,
        )

        rows.append({
            "sweep": "sigma",
            "bonus": 1,
            "sigma_px": float(sigma_px),
            "tc": float(out["tc"]),
            "dropout_prob": float(dropout_prob_bonus),
            "fixed_camera_pointing": int(bool(out.get("fixed_camera_pointing", True))),
            "dv_perfect": float(out["dv_perfect_mag"]),
            "dv_ekf": float(out["dv_ekf_mag"]),
            "dv_delta": float(out["dv_delta_mag"]),
            "dv_inflation": float(out["dv_inflation"]),
            "dv_inflation_pct": float(out["dv_inflation_pct"]),
            "miss_unc": float(out["miss_uncorrected"]),
            "miss_perf": float(out["miss_perfect"]),
            "miss_ekf": float(out["miss_ekf"]),
            "pos_err_tc": float(out["pos_err_tc"]),
            "tracePpos_tc": float(out["tracePpos_tc"]),
            "nis_mean": float(out["nis_mean"]),
            "valid_rate": float(out["valid_rate"]),
        })

    sigma_bonus = np.array([r["sigma_px"] for r in rows if r["sweep"] == "sigma" and r["bonus"] == 1], dtype=float)
    dv_infl_bonus = np.array([r["dv_inflation"] for r in rows if r["sweep"] == "sigma" and r["bonus"] == 1], dtype=float)
    miss_bonus = np.array([r["miss_ekf"] for r in rows if r["sweep"] == "sigma" and r["bonus"] == 1], dtype=float)
    poserr_bonus = np.array([r["pos_err_tc"] for r in rows if r["sweep"] == "sigma" and r["bonus"] == 1], dtype=float)

    _plot_xy(
        sigma_bonus, dv_infl_bonus,
        xlabel="sigma_px [px]",
        ylabel="ΔV inflation (|dv_ekf| - |dv_perfect|)",
        title="06B BONUS: ΔV inflation vs sigma_px (dropout=0.05, fixed pointing)",
        outpath=plots_dir / "06b_bonus_dv_inflation_vs_sigma.png",
    )
    _plot_xy(
        sigma_bonus, miss_bonus,
        xlabel="sigma_px [px]",
        ylabel="miss_ekf at tf",
        title="06B BONUS: terminal miss vs sigma_px (dropout=0.05, fixed pointing)",
        outpath=plots_dir / "06b_bonus_miss_vs_sigma.png",
    )
    _plot_xy(
        sigma_bonus, poserr_bonus,
        xlabel="sigma_px [px]",
        ylabel="||r_hat(tc) - r_true(tc)||",
        title="06B BONUS: pos err at tc vs sigma_px (dropout=0.05, fixed pointing)",
        outpath=plots_dir / "06b_bonus_poserr_tc_vs_sigma.png",
    )

    csv_path = plots_dir / "06b_sensitivity.csv"
    _write_csv(csv_path, rows)

    print("06B complete.")
    print(f"Wrote CSV: {csv_path}")
    print("Wrote plots:")
    for p in [
        "06b_dv_inflation_vs_sigma.png",
        "06b_miss_vs_sigma.png",
        "06b_poserr_tc_vs_sigma.png",
        "06b_dv_inflation_vs_tc.png",
        "06b_miss_vs_tc.png",
        "06b_poserr_tc_vs_tc.png",
        "06b_bonus_dv_inflation_vs_sigma.png",
        "06b_bonus_miss_vs_sigma.png",
        "06b_bonus_poserr_tc_vs_sigma.png",
    ]:
        print(f"  - {plots_dir / p}")


if __name__ == "__main__":
    main()
