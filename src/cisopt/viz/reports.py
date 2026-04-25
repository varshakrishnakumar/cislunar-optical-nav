"""Build a paper-ready figure set from a Parquet trial table.

Each ``build_*`` function takes a Parquet path (the ``trials.parquet`` written
by the MC or ablation runner) and writes its figures into the supplied output
directory. They're composable -- ``build_report`` runs the full set.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from .distributions import plot_box, plot_cdf, plot_hist, plot_scatter
from .style import PALETTE, style_axis


def _load_rows(path: str | Path) -> list[dict]:
    return pq.read_table(Path(path)).to_pylist()


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_mc_distributions(
    parquet_path: str | Path, out_dir: str | Path, *, prefix: str = "",
) -> list[Path]:
    """4-panel histogram + 1 CDF for the canonical MC metrics."""
    rows = _load_rows(parquet_path)
    out = Path(out_dir)
    paths: list[Path] = []

    metrics = [
        ("miss_ekf",          "miss_ekf",          False),
        ("miss_perfect",      "miss_perfect",      False),
        ("dv_ekf_mag",        "dv_ekf_mag",        False),
        ("dv_inflation_pct",  "dv_inflation_pct",  False),
        ("nis_mean",          "nis_mean",          False),
        ("nees_mean",         "nees_mean",         False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (m, label, log_x) in zip(axes.flat, metrics):
        plot_hist(rows, m, ax=ax, log_x=log_x, color=PALETTE["cyan"])
        ax.set_title(m)
    fig.suptitle("MC metric distributions", y=1.00)
    fig.tight_layout()
    p = out / f"{prefix}mc_metric_hists.png"
    _save(fig, p); paths.append(p)

    # CDF panel for the most informative metric
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_cdf(rows, "miss_ekf", ax=ax, log_x=True)
    ax.set_title("miss_ekf empirical CDF (log x)")
    p = out / f"{prefix}mc_miss_cdf.png"
    _save(fig, p); paths.append(p)

    return paths


def build_ablation_panel(
    parquet_path: str | Path, out_dir: str | Path, *, prefix: str = "",
) -> list[Path]:
    rows = _load_rows(parquet_path)
    out = Path(out_dir)
    paths: list[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    plot_box(rows, "miss_ekf",         "combo_id", ax=axes[0, 0], log_y=True)
    axes[0, 0].set_title("miss_ekf by combo (log)")
    plot_box(rows, "dv_inflation_pct", "combo_id", ax=axes[0, 1])
    axes[0, 1].set_title("dv_inflation_pct by combo")
    plot_box(rows, "nees_mean",        "combo_id", ax=axes[1, 0])
    axes[1, 0].set_title("nees_mean by combo  (target ≈ 6)")
    axes[1, 0].axhline(6.0, color=PALETTE["green"], ls="--", lw=0.9, alpha=0.7)
    # valid_rate is geometry-determined and has zero within-combo variance,
    # so its boxes degenerate to a line. pos_err_tc actually varies and tells
    # the more interesting story of filter position-error spread.
    plot_box(rows, "pos_err_tc",       "combo_id", ax=axes[1, 1], log_y=True)
    axes[1, 1].set_title("pos_err_tc by combo (log)")
    fig.suptitle("Ablation: miss / Δv / NEES / pos error by combo", y=1.00)
    fig.tight_layout()
    p = out / f"{prefix}ablation_box.png"
    _save(fig, p); paths.append(p)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    plot_cdf(rows, "miss_ekf", by="combo_id", ax=ax,
             log_x=True, legend_outside=True)
    ax.set_title("miss_ekf CDFs by combo (log x)")
    fig.tight_layout()
    p = out / f"{prefix}ablation_miss_cdf.png"
    _save(fig, p); paths.append(p)

    return paths


def build_estimator_zoo_panel(
    parquet_paths: dict[str, str | Path], out_dir: str | Path, *, prefix: str = "",
) -> list[Path]:
    """``parquet_paths``: mapping estimator name -> trials.parquet."""
    out = Path(out_dir)
    paths: list[Path] = []

    rows_combined: list[dict] = []
    for est_name, p in parquet_paths.items():
        for r in _load_rows(p):
            rr = dict(r)
            rr["estimator"] = est_name
            rows_combined.append(rr)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    plot_box(rows_combined, "miss_ekf",   by="estimator", ax=axes[0], log_y=True, rotate_labels=0)
    axes[0].set_title("miss_ekf")
    plot_box(rows_combined, "dv_ekf_mag", by="estimator", ax=axes[1], rotate_labels=0)
    axes[1].set_title("|dv|_ekf")
    plot_box(rows_combined, "nees_mean",  by="estimator", ax=axes[2], rotate_labels=0)
    axes[2].set_title("NEES (target ≈ 6)")
    axes[2].axhline(6.0, color=PALETTE["green"], ls="--", lw=0.9, alpha=0.7)
    fig.suptitle("Estimator zoo: EKF / IEKF / UKF on the same scenario", y=1.02)
    fig.tight_layout()
    p = out / f"{prefix}estimator_zoo.png"
    _save(fig, p); paths.append(p)

    return paths


def build_estimator_deep_dive(
    parquet_paths: dict[str, str | Path],
    out_dir: str | Path,
    *,
    prefix: str = "",
    label: str = "",
) -> list[Path]:
    """Per-estimator NIS/NEES histograms with χ² reference lines + CDFs of
    miss_ekf and dv_inflation_pct.

    ``parquet_paths``: mapping estimator name -> trials.parquet.
    ``label`` is appended to the figure title (e.g. "(CR3BP)" / "(SPICE)").
    """
    from scipy.stats import chi2

    out = Path(out_dir)
    paths: list[Path] = []

    rows_by_est: dict[str, list[dict]] = {
        name: _load_rows(p) for name, p in parquet_paths.items()
    }
    rows_combined = []
    for name, rs in rows_by_est.items():
        for r in rs:
            rr = dict(r); rr["estimator"] = name
            rows_combined.append(rr)

    n_est = len(rows_by_est)
    fig, axes = plt.subplots(2, n_est, figsize=(4.5 * n_est, 7), squeeze=False)

    # χ² reference shading for NIS (df=2) and NEES (df=6).
    nis_lo, nis_hi = chi2.ppf(0.025, 2), chi2.ppf(0.975, 2)
    nees_lo, nees_hi = chi2.ppf(0.025, 6), chi2.ppf(0.975, 6)

    for col, (est_name, rs) in enumerate(sorted(rows_by_est.items())):
        ax_nis = axes[0, col]
        ax_nees = axes[1, col]

        plot_hist(rs, "nis_mean",  ax=ax_nis,  color=PALETTE["cyan"],   show_stats=True)
        ax_nis.axvspan(nis_lo, nis_hi, color=PALETTE["green"], alpha=0.10,
                       label=f"χ²(2) 95% [{nis_lo:.1f}, {nis_hi:.1f}]")
        ax_nis.axvline(2.0, color=PALETTE["green"], ls=":", lw=0.9)
        ax_nis.set_title(f"{est_name.upper()}: NIS distribution")
        ax_nis.legend(fontsize=7, loc="upper right")

        plot_hist(rs, "nees_mean", ax=ax_nees, color=PALETTE["violet"], show_stats=True)
        ax_nees.axvspan(nees_lo, nees_hi, color=PALETTE["green"], alpha=0.10,
                        label=f"χ²(6) 95% [{nees_lo:.1f}, {nees_hi:.1f}]")
        ax_nees.axvline(6.0, color=PALETTE["green"], ls=":", lw=0.9)
        ax_nees.set_title(f"{est_name.upper()}: NEES distribution")
        ax_nees.legend(fontsize=7, loc="upper right")

    suffix = f" {label}" if label else ""
    fig.suptitle(f"Estimator zoo deep dive{suffix} — NIS / NEES distributions", y=1.00)
    fig.tight_layout()
    p = out / f"{prefix}estimator_deep_dive_dists.png"
    _save(fig, p); paths.append(p)

    # CDFs of miss_ekf and dv_inflation_pct on shared axes.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_cdf(rows_combined, "miss_ekf",         by="estimator", ax=axes[0], log_x=True)
    axes[0].set_title("miss_ekf CDF by estimator")
    plot_cdf(rows_combined, "dv_inflation_pct", by="estimator", ax=axes[1])
    axes[1].set_title("dv_inflation_pct CDF by estimator")
    fig.suptitle(f"Estimator zoo deep dive{suffix} — outcome CDFs", y=1.02)
    fig.tight_layout()
    p = out / f"{prefix}estimator_deep_dive_cdfs.png"
    _save(fig, p); paths.append(p)

    return paths


def build_coupling_panel(
    parquet_path: str | Path, out_dir: str | Path, *, prefix: str = "",
) -> list[Path]:
    rows = _load_rows(parquet_path)
    out = Path(out_dir)
    paths: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_scatter(rows, "sigma_r", "miss_offset", by="sigma_v", ax=axes[0],
                 log_x=True, log_y=True)
    axes[0].set_title("σ_r,est vs miss_offset (grouped by σ_v)")
    plot_scatter(rows, "sigma_r", "dv_inflation_pct", by="sigma_v", ax=axes[1],
                 log_x=True, log_y=True)
    axes[1].set_title("σ_r,est vs Δv inflation (log-log)")
    fig.suptitle("Navigation→burn coupling map", y=1.02)
    fig.tight_layout()
    p = out / f"{prefix}coupling.png"
    _save(fig, p); paths.append(p)

    return paths


def build_observability_panel(
    npz_path: str | Path, out_dir: str | Path, *, prefix: str = "",
) -> list[Path]:
    out = Path(out_dir)
    paths: list[Path] = []

    data = np.load(Path(npz_path))
    eigvals = data["eigvals"]
    weak = data["weak_directions"]
    W = data["W"]

    full_eig = np.linalg.eigvalsh(0.5 * (W + W.T))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    state_labels = ["x", "y", "z", "vx", "vy", "vz"]
    eigvecs_full = np.linalg.eigh(0.5 * (W + W.T))[1]

    axes[0].bar(range(len(full_eig)), full_eig, color=PALETTE["cyan"],
                edgecolor=PALETTE["border"], lw=0.6)
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(full_eig)))
    axes[0].set_xticklabels([f"λ_{i+1}" for i in range(len(full_eig))])
    axes[0].set_title("Final Gramian eigenvalues (log)")
    axes[0].set_ylabel("eigenvalue")
    style_axis(axes[0])

    weights = np.abs(eigvecs_full)
    im = axes[1].imshow(weights, aspect="auto", cmap="magma_r", origin="lower")
    axes[1].set_xticks(range(weights.shape[1]))
    axes[1].set_xticklabels(
        [f"λ={full_eig[i]:.1e}" for i in range(weights.shape[1])],
        rotation=30, ha="right",
    )
    axes[1].set_yticks(range(weights.shape[0]))
    axes[1].set_yticklabels(state_labels)
    axes[1].set_title("Eigenvector magnitudes (cols = modes)")
    fig.colorbar(im, ax=axes[1], fraction=0.04, pad=0.02)

    fig.suptitle("Observability Gramian summary", y=1.02)
    fig.tight_layout()
    p = out / f"{prefix}observability.png"
    _save(fig, p); paths.append(p)

    return paths


def build_report(
    *,
    mc_parquet: str | Path | None = None,
    ablation_parquet: str | Path | None = None,
    estimator_paths: dict[str, str | Path] | None = None,
    estimator_paths_spice: dict[str, str | Path] | None = None,
    coupling_parquet: str | Path | None = None,
    observability_npz: str | Path | None = None,
    out_dir: str | Path,
    prefix: str = "",
) -> list[Path]:
    """Run every requested panel into ``out_dir``. Skips missing inputs."""
    paths: list[Path] = []
    if mc_parquet is not None:
        paths += build_mc_distributions(mc_parquet, out_dir, prefix=prefix)
    if ablation_parquet is not None:
        paths += build_ablation_panel(ablation_parquet, out_dir, prefix=prefix)
    if estimator_paths is not None:
        paths += build_estimator_zoo_panel(estimator_paths, out_dir, prefix=prefix)
        paths += build_estimator_deep_dive(estimator_paths, out_dir,
                                           prefix=prefix + "cr3bp_",
                                           label="(CR3BP halo-L1)")
    if estimator_paths_spice is not None:
        paths += build_estimator_deep_dive(estimator_paths_spice, out_dir,
                                           prefix=prefix + "spice_",
                                           label="(SPICE halo-L1)")
    if coupling_parquet is not None:
        paths += build_coupling_panel(coupling_parquet, out_dir, prefix=prefix)
    if observability_npz is not None:
        paths += build_observability_panel(observability_npz, out_dir, prefix=prefix)
    return paths
