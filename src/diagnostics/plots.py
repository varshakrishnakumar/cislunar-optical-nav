from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from diagnostics.types import HypothesisResult, RunResult, UpdateRecord


Array = np.ndarray


_BG = "#0b1020"
_AX_BG = "#111827"
_GRID = "#94a3b8"
_TEXT = "#e5e7eb"
_MUTED = "#94a3b8"

_CYAN = "#22d3ee"
_SKY = "#38bdf8"
_BLUE = "#60a5fa"
_INDIGO = "#818cf8"
_VIOLET = "#a78bfa"
_MAGENTA = "#e879f9"
_PINK = "#f472b6"
_RED = "#fb7185"
_ORANGE = "#fb923c"
_AMBER = "#fbbf24"
_LIME = "#a3e635"
_GREEN = "#34d399"
_TEAL = "#2dd4bf"
_WHITE = "#f8fafc"

_TRAJ_TRUE = _CYAN
_TRAJ_EST = _AMBER
_TRAJ_MINUS = _MAGENTA
_VALID = _GREEN
_REJECT = _RED
_ACCEPT = _CYAN
_ERR = _AMBER
_SIGMA = _VIOLET
_NIS = _CYAN
_NEES_MINUS = _PINK
_NEES_PLUS = _LIME
_LOS = _SKY
_PIXEL = _ORANGE
_BODY = _WHITE


def apply_dark_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": _BG,
            "axes.facecolor": _AX_BG,
            "axes.edgecolor": _MUTED,
            "axes.labelcolor": _TEXT,
            "axes.titlecolor": _WHITE,
            "xtick.color": _TEXT,
            "ytick.color": _TEXT,
            "text.color": _TEXT,
            "legend.facecolor": "#0f172a",
            "legend.edgecolor": _MUTED,
            "savefig.facecolor": _BG,
            "savefig.edgecolor": _BG,
            "grid.color": _GRID,
            "grid.alpha": 0.30,
            "grid.linestyle": "--",
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelweight": "medium",
        }
    )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _finite_mask(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return np.isfinite(x)
    return np.all(np.isfinite(x), axis=1)


def _safe_bounds(x: Array, pad_frac: float = 0.08) -> tuple[float, float]:
    vals = np.asarray(x, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if np.isclose(lo, hi):
        d = 1.0 if lo == 0.0 else 0.1 * abs(lo)
        return lo - d, hi + d
    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad


def _scatter_update_markers(t: Array, y: Array, updates: Sequence[UpdateRecord], *, ax, label: str = "updates") -> None:
    used = np.array([u.update_used for u in updates], dtype=bool)
    if used.shape[0] != len(t):
        return
    if not np.any(used):
        return
    ax.scatter(
        t[used],
        y[used],
        s=28,
        color=_WHITE,
        edgecolors=_BG,
        linewidths=0.8,
        zorder=6,
        label=label,
    )


def _style_axes(ax, *, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_xy_trajectory(result: RunResult, outpath: str | Path) -> Path:
    apply_dark_theme()
    outpath = _as_path(outpath)

    x_true = result.trace.x_true_hist
    x_minus = result.trace.xhat_minus_hist
    x_plus = result.trace.xhat_plus_hist

    fig, ax = plt.subplots(figsize=(8.5, 7.0))

    ax.plot(x_true[:, 0], x_true[:, 1], color=_TRAJ_TRUE, linewidth=2.4, label="truth")
    ax.plot(x_plus[:, 0], x_plus[:, 1], color=_TRAJ_EST, linewidth=2.2, label="estimate (+)")
    if np.any(np.isfinite(x_minus)):
        ax.plot(x_minus[:, 0], x_minus[:, 1], color=_TRAJ_MINUS, linewidth=1.4, alpha=0.8, label="estimate (-)")

    ax.scatter(x_true[0, 0], x_true[0, 1], s=80, color=_GREEN, edgecolors=_BG, linewidths=0.8, label="start", zorder=5)
    ax.scatter(x_true[-1, 0], x_true[-1, 1], s=90, color=_RED, edgecolors=_BG, linewidths=0.8, label="end", zorder=5)

    body_x = 1.0 - float(result.config["mu"])
    ax.scatter([body_x], [0.0], s=140, color=_BODY, edgecolors=_BG, linewidths=1.0, label="observed body", zorder=6)

    _style_axes(ax, title="CR3BP XY Trajectory", xlabel="x", ylabel="y")
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath


def plot_3sigma_consistency(
    result: RunResult,
    outpath: str | Path,
    *,
    idxs: tuple[int, int, int],
    labels: tuple[str, str, str],
    title: str,
) -> Path:
    apply_dark_theme()
    outpath = _as_path(outpath)

    t = result.trace.t_meas
    err = result.trace.err_plus_hist
    P = result.trace.P_plus_hist

    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    colors = [_CYAN, _AMBER, _MAGENTA]
    for idx, label, color in zip(idxs, labels, colors):
        sigma3 = 3.0 * np.sqrt(np.maximum(P[:, idx, idx], 0.0))
        ax.plot(t, err[:, idx], color=color, linewidth=2.0, label=f"{label} error")
        ax.plot(t, sigma3, color=color, linestyle="--", linewidth=1.3, alpha=0.9, label=f"+3σ {label}")
        ax.plot(t, -sigma3, color=color, linestyle="--", linewidth=1.3, alpha=0.9, label=f"-3σ {label}")

    _style_axes(ax, title=title, xlabel="t", ylabel="state error")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath


def plot_nis_nees(result: RunResult, outdir: str | Path) -> dict[str, Path]:
    apply_dark_theme()
    outdir = _as_path(outdir)
    _ensure_dir(outdir)

    t = result.trace.t_meas
    nis = np.array([u.nis for u in result.trace.updates], dtype=float)
    nees_minus = result.trace.nees_minus_hist
    nees_plus = result.trace.nees_plus_hist

    nis_path = outdir / "nis.png"
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.plot(t, nis, color=_NIS, linewidth=2.0, label="NIS")
    _scatter_update_markers(t, np.nan_to_num(nis, nan=np.nan), result.trace.updates, ax=ax)
    _style_axes(ax, title="Normalized Innovation Squared (NIS)", xlabel="t", ylabel="NIS")
    ax.legend()
    fig.tight_layout()
    fig.savefig(nis_path, dpi=220)
    plt.close(fig)

    nees_path = outdir / "nees.png"
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(t, nees_minus, color=_NEES_MINUS, linewidth=1.9, label="NEES (-)")
    ax.plot(t, nees_plus, color=_NEES_PLUS, linewidth=1.9, label="NEES (+)")
    _style_axes(ax, title="Normalized Estimation Error Squared (NEES)", xlabel="t", ylabel="NEES")
    ax.legend()
    fig.tight_layout()
    fig.savefig(nees_path, dpi=220)
    plt.close(fig)

    return {"nis": nis_path, "nees": nees_path}


def plot_error_norms(result: RunResult, outpath: str | Path) -> Path:
    apply_dark_theme()
    outpath = _as_path(outpath)

    t = result.trace.t_meas
    err = result.trace.err_plus_hist
    pos_norm = np.linalg.norm(err[:, :3], axis=1)
    vel_norm = np.linalg.norm(err[:, 3:6], axis=1)

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.plot(t, pos_norm, color=_AMBER, linewidth=2.2, label="||position error||")
    ax.plot(t, vel_norm, color=_CYAN, linewidth=2.0, label="||velocity error||")
    _scatter_update_markers(t, pos_norm, result.trace.updates, ax=ax)

    _style_axes(ax, title="Error Norms Through Time", xlabel="t", ylabel="norm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath


def plot_los_angle(result: RunResult, outpath: str | Path) -> Path:
    apply_dark_theme()
    outpath = _as_path(outpath)

    t = result.trace.t_meas
    los_angle = result.trace.los_angle_hist

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.plot(t, los_angle, color=_LOS, linewidth=2.2)
    _scatter_update_markers(t, np.nan_to_num(los_angle, nan=np.nan), result.trace.updates, ax=ax)
    _style_axes(ax, title="LOS Mismatch Angle", xlabel="t", ylabel="angle [rad]")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath


def plot_visibility_and_gating(result: RunResult, outpath: str | Path) -> Path:
    apply_dark_theme()
    outpath = _as_path(outpath)

    t = result.trace.t_meas
    valid = np.array([u.valid_measurement for u in result.trace.updates], dtype=int)
    used = np.array([u.update_used for u in result.trace.updates], dtype=int)
    accepted = np.array([0 if u.gate is None else int(u.gate.accepted) for u in result.trace.updates], dtype=int)

    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    ax.step(t, valid + 0.10, where="post", color=_GREEN, linewidth=2.0, label="valid measurement")
    ax.step(t, accepted + 0.00, where="post", color=_CYAN, linewidth=2.0, label="gate accepted")
    ax.step(t, used - 0.10, where="post", color=_AMBER, linewidth=2.0, label="update used")

    ax.set_yticks([0, 1])
    ax.set_ylim(-0.5, 1.5)
    _style_axes(ax, title="Measurement Validity, Gating, and Update Use", xlabel="t", ylabel="0 / 1")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath


def plot_innovations(result: RunResult, outpath: str | Path) -> Path:
    apply_dark_theme()
    outpath = _as_path(outpath)

    t = result.trace.t_meas
    innov = np.full((len(result.trace.updates), 2), np.nan, dtype=float)
    for i, u in enumerate(result.trace.updates):
        if u.innovation is not None:
            innov[i] = u.innovation

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.plot(t, innov[:, 0], color=_CYAN, linewidth=1.9, label="innovation[0]")
    ax.plot(t, innov[:, 1], color=_PINK, linewidth=1.9, label="innovation[1]")
    _style_axes(ax, title="Innovation Components", xlabel="t", ylabel="innovation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath



def _extract_pixel_traces(result: RunResult) -> tuple[Array, Array, Array]:
    n = len(result.trace.updates)
    meas = np.full((n, 2), np.nan, dtype=float)
    pred = np.full((n, 2), np.nan, dtype=float)
    used = np.zeros(n, dtype=bool)

    for i, u in enumerate(result.trace.updates):
        if u.pixel_uv is not None:
            meas[i] = np.asarray(u.pixel_uv, dtype=float)
        if getattr(u, "pixel_uv_pred", None) is not None:
            pred[i] = np.asarray(u.pixel_uv_pred, dtype=float)
        used[i] = bool(u.update_used)

    return meas, pred, used

def plot_predicted_vs_measured_pixels(result: RunResult, outpath: str | Path) -> Path:
    apply_dark_theme()
    outpath = _as_path(outpath)

    t = result.trace.t_meas
    meas, pred, used = _extract_pixel_traces(result)

    fig, ax = plt.subplots(figsize=(11.0, 5.4))

    ax.plot(t, meas[:, 0], color=_CYAN, linewidth=2.0, label="u_meas")
    ax.plot(t, pred[:, 0], color=_CYAN, linewidth=1.6, linestyle="--", alpha=0.95, label="u_pred")
    ax.plot(t, meas[:, 1], color=_AMBER, linewidth=2.0, label="v_meas")
    ax.plot(t, pred[:, 1], color=_AMBER, linewidth=1.6, linestyle="--", alpha=0.95, label="v_pred")

    if np.any(used):
        ax.scatter(t[used], meas[used, 0], s=20, color=_WHITE, edgecolors=_BG, linewidths=0.5, zorder=6)
        ax.scatter(t[used], meas[used, 1], s=20, color=_WHITE, edgecolors=_BG, linewidths=0.5, zorder=6)

    _style_axes(ax, title="Measured vs Predicted Pixel Coordinates", xlabel="t", ylabel="pixel coordinate")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath

def plot_image_plane_measured_vs_predicted(result: RunResult, outpath: str | Path) -> Path:
    apply_dark_theme()
    outpath = _as_path(outpath)

    meas, pred, used = _extract_pixel_traces(result)
    meas_ok = np.all(np.isfinite(meas), axis=1)
    pred_ok = np.all(np.isfinite(pred), axis=1)

    fig, ax = plt.subplots(figsize=(8.5, 6.8))

    if np.any(meas_ok):
        ax.scatter(
            meas[meas_ok, 0],
            meas[meas_ok, 1],
            s=34,
            color=_CYAN,
            alpha=0.75,
            label="measured pixels",
        )

    if np.any(pred_ok):
        ax.scatter(
            pred[pred_ok, 0],
            pred[pred_ok, 1],
            s=34,
            color=_AMBER,
            alpha=0.75,
            label="predicted pixels",
        )

    both = meas_ok & pred_ok
    for i in np.where(both)[0]:
        ax.plot(
            [meas[i, 0], pred[i, 0]],
            [meas[i, 1], pred[i, 1]],
            color=_MUTED,
            alpha=0.20,
            linewidth=0.8,
        )

    used_both = both & used
    if np.any(used_both):
        ax.scatter(
            meas[used_both, 0],
            meas[used_both, 1],
            s=52,
            color=_WHITE,
            edgecolors=_BG,
            linewidths=0.6,
            label="accepted measurements",
            zorder=6,
        )

    ax.set_xlim(-10, 650)
    ax.set_ylim(490, -10)
    _style_axes(ax, title="Image Plane: Measured vs Predicted", xlabel="u [px]", ylabel="v [px]")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath

def plot_image_plane_detections(result: RunResult, outpath: str | Path) -> Path:
    apply_dark_theme()
    outpath = _as_path(outpath)

    pixel_uv = np.full((len(result.trace.updates), 2), np.nan, dtype=float)
    valid = np.zeros(len(result.trace.updates), dtype=bool)
    used = np.zeros(len(result.trace.updates), dtype=bool)

    for i, u in enumerate(result.trace.updates):
        if u.pixel_uv is not None:
            pixel_uv[i] = u.pixel_uv
        valid[i] = u.valid_measurement
        used[i] = u.update_used

    fig, ax = plt.subplots(figsize=(8.4, 6.6))

    mask_valid = np.all(np.isfinite(pixel_uv), axis=1) & valid
    mask_used = mask_valid & used
    mask_rejected = mask_valid & (~used)

    if np.any(mask_valid):
        ax.scatter(
            pixel_uv[mask_valid, 0],
            pixel_uv[mask_valid, 1],
            s=34,
            color=_MUTED,
            alpha=0.30,
            label="all valid detections",
        )
    if np.any(mask_used):
        ax.scatter(
            pixel_uv[mask_used, 0],
            pixel_uv[mask_used, 1],
            s=48,
            color=_ACCEPT,
            edgecolors=_BG,
            linewidths=0.7,
            label="accepted / used",
        )
    if np.any(mask_rejected):
        ax.scatter(
            pixel_uv[mask_rejected, 0],
            pixel_uv[mask_rejected, 1],
            s=46,
            color=_REJECT,
            edgecolors=_BG,
            linewidths=0.7,
            label="rejected",
        )

    ax.axvline(0.0, color=_GRID, alpha=0.18, linewidth=1.0)
    ax.axhline(0.0, color=_GRID, alpha=0.18, linewidth=1.0)
    ax.axvline(640.0, color=_GRID, alpha=0.18, linewidth=1.0)
    ax.axhline(480.0, color=_GRID, alpha=0.18, linewidth=1.0)

    ax.set_xlim(-10, 650)
    ax.set_ylim(490, -10)
    _style_axes(ax, title="Camera Image-Plane Detections", xlabel="u [px]", ylabel="v [px]")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath


def plot_detection_timeline(result: RunResult, outpath: str | Path) -> Path:
    apply_dark_theme()
    outpath = _as_path(outpath)

    t = result.trace.t_meas
    u_hist = np.full(len(result.trace.updates), np.nan, dtype=float)
    v_hist = np.full(len(result.trace.updates), np.nan, dtype=float)
    accepted = np.zeros(len(result.trace.updates), dtype=bool)

    for i, u in enumerate(result.trace.updates):
        if u.pixel_uv is not None:
            u_hist[i] = float(u.pixel_uv[0])
            v_hist[i] = float(u.pixel_uv[1])
        accepted[i] = u.update_used

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.plot(t, u_hist, color=_CYAN, linewidth=2.0, label="u_px")
    ax.plot(t, v_hist, color=_AMBER, linewidth=2.0, label="v_px")

    if np.any(accepted):
        ax.scatter(t[accepted], u_hist[accepted], s=24, color=_WHITE, edgecolors=_BG, linewidths=0.6, zorder=6)
        ax.scatter(t[accepted], v_hist[accepted], s=24, color=_WHITE, edgecolors=_BG, linewidths=0.6, zorder=6)

    _style_axes(ax, title="Detection Coordinates Through Time", xlabel="t", ylabel="pixel coordinate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath


def plot_camera_strip(result: RunResult, outpath: str | Path, *, max_frames: int = 10) -> Path:
    apply_dark_theme()
    outpath = _as_path(outpath)

    pixel_uv = []
    times = []
    statuses = []

    for u in result.trace.updates:
        if u.pixel_uv is not None and np.all(np.isfinite(u.pixel_uv)):
            pixel_uv.append(np.asarray(u.pixel_uv, dtype=float))
            times.append(float(u.t))
            statuses.append(bool(u.update_used))

    if len(pixel_uv) == 0:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.text(0.5, 0.5, "No valid detections available", ha="center", va="center", fontsize=16, color=_TEXT)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(outpath, dpi=220)
        plt.close(fig)
        return outpath

    idxs = np.linspace(0, len(pixel_uv) - 1, min(max_frames, len(pixel_uv))).astype(int)

    fig, axes = plt.subplots(1, len(idxs), figsize=(2.8 * len(idxs), 3.2))
    if len(idxs) == 1:
        axes = [axes]

    for ax, idx in zip(axes, idxs):
        uv = pixel_uv[idx]
        used = statuses[idx]
        t = times[idx]

        ax.set_facecolor(_AX_BG)
        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)
        ax.grid(True, linewidth=0.7, alpha=0.18)
        ax.scatter(
            [uv[0]],
            [uv[1]],
            s=180,
            color=_ACCEPT if used else _REJECT,
            edgecolors=_WHITE,
            linewidths=1.0,
            zorder=6,
        )
        for scale, alpha in [(520, 0.06), (320, 0.10), (220, 0.16)]:
            ax.scatter([uv[0]], [uv[1]], s=scale, color=_ACCEPT if used else _REJECT, alpha=alpha, edgecolors="none")

        ax.set_title(f"t={t:.2f}", fontsize=10, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Camera Detection Strip", fontsize=15, color=_WHITE, y=1.03, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return outpath

def plot_hypothesis_summary(
    hypotheses: Sequence[HypothesisResult],
    outpath: str | Path,
) -> Path:
    apply_dark_theme()
    outpath = _as_path(outpath)

    labels = [h.name for h in hypotheses]
    scores = np.array([1 if h.passed else 0 for h in hypotheses], dtype=float)
    colors = [
        _GREEN if h.passed and h.severity == "info"
        else _AMBER if h.passed and h.severity == "warning"
        else _RED if not h.passed
        else _MUTED
        for h in hypotheses
    ]

    fig_h = max(4.5, 0.36 * len(labels))
    fig, ax = plt.subplots(figsize=(10.5, fig_h))

    y = np.arange(len(labels))
    ax.barh(y, scores, color=colors, edgecolor=_BG, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["fail", "pass"])
    _style_axes(ax, title="Hypothesis Pass / Fail Summary", xlabel="outcome", ylabel="diagnostic")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath


def plot_run_dashboard(
    result: RunResult,
    outpath: str | Path,
) -> Path:
    apply_dark_theme()
    outpath = _as_path(outpath)

    t = result.trace.t_meas
    pos_norm = np.linalg.norm(result.trace.err_plus_hist[:, :3], axis=1)
    los_angle = result.trace.los_angle_hist
    nis = np.array([u.nis for u in result.trace.updates], dtype=float)
    valid = np.array([u.valid_measurement for u in result.trace.updates], dtype=int)
    used = np.array([u.update_used for u in result.trace.updates], dtype=int)

    fig = plt.figure(figsize=(13.5, 9.5))
    gs = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    x_true = result.trace.x_true_hist
    x_plus = result.trace.xhat_plus_hist
    ax1.plot(x_true[:, 0], x_true[:, 1], color=_TRAJ_TRUE, linewidth=2.2, label="truth")
    ax1.plot(x_plus[:, 0], x_plus[:, 1], color=_TRAJ_EST, linewidth=2.0, label="estimate")
    ax1.scatter([1.0 - float(result.config["mu"])], [0.0], s=110, color=_WHITE, edgecolors=_BG, linewidths=0.8)
    _style_axes(ax1, title="Trajectory", xlabel="x", ylabel="y")
    ax1.legend()
    ax1.set_aspect("equal", adjustable="box")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, pos_norm, color=_ERR, linewidth=2.2, label="position error")
    ax2.plot(t, los_angle, color=_LOS, linewidth=2.0, label="LOS angle")
    _style_axes(ax2, title="Tracking Error", xlabel="t", ylabel="metric")
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, nis, color=_NIS, linewidth=2.0, label="NIS")
    ax3.step(t, valid, where="post", color=_GREEN, linewidth=1.8, label="valid")
    ax3.step(t, used, where="post", color=_AMBER, linewidth=1.8, label="used")
    _style_axes(ax3, title="Measurement / Update Activity", xlabel="t", ylabel="value")
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    meas, pred, used = _extract_pixel_traces(result)
    meas_ok = np.all(np.isfinite(meas), axis=1)
    pred_ok = np.all(np.isfinite(pred), axis=1)

    if np.any(meas_ok):
        ax4.scatter(meas[meas_ok, 0], meas[meas_ok, 1], s=26, color=_CYAN, alpha=0.70, label="measured")
    if np.any(pred_ok):
        ax4.scatter(pred[pred_ok, 0], pred[pred_ok, 1], s=26, color=_AMBER, alpha=0.70, label="predicted")

    both = meas_ok & pred_ok
    for i in np.where(both)[0]:
        ax4.plot(
            [meas[i, 0], pred[i, 0]],
            [meas[i, 1], pred[i, 1]],
            color=_MUTED,
            alpha=0.18,
            linewidth=0.7,
        )

    used_both = both & used
    if np.any(used_both):
        ax4.scatter(
            meas[used_both, 0],
            meas[used_both, 1],
            s=40,
            color=_WHITE,
            edgecolors=_BG,
            linewidths=0.5,
            zorder=6,
            label="accepted",
        )

    ax4.set_xlim(-10, 650)
    ax4.set_ylim(490, -10)
    _style_axes(ax4, title="Measured vs Predicted Image Plane", xlabel="u [px]", ylabel="v [px]")
    ax4.legend(loc="best", fontsize=8)

    title = (
        f"EKF Diagnostics Dashboard  |  mode={result.summary.camera_mode}  |  "
        f"final_pos_err={result.summary.final_pos_err:.3e}  |  "
        f"update_rate={result.summary.update_rate:.2f}"
    )
    fig.suptitle(title, fontsize=16, color=_WHITE, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=240)
    plt.close(fig)
    return outpath





def save_all_plots(
    result: RunResult,
    outdir: str | Path,
    *,
    hypotheses: Optional[Sequence[HypothesisResult]] = None,
) -> dict[str, Path]:
    apply_dark_theme()
    outdir = _as_path(outdir)
    _ensure_dir(outdir)

    outputs: dict[str, Path] = {}

    outputs["trajectory_xy"] = plot_xy_trajectory(result, outdir / "trajectory_xy.png")
    outputs["pos_3sigma"] = plot_3sigma_consistency(
        result,
        outdir / "pos_3sigma.png",
        idxs=(0, 1, 2),
        labels=("x", "y", "z"),
        title="Position 3σ Consistency",
    )
    outputs["vel_3sigma"] = plot_3sigma_consistency(
        result,
        outdir / "vel_3sigma.png",
        idxs=(3, 4, 5),
        labels=("vx", "vy", "vz"),
        title="Velocity 3σ Consistency",
    )

    outputs.update({f"core_{k}": v for k, v in plot_nis_nees(result, outdir).items()})
    outputs["error_norms"] = plot_error_norms(result, outdir / "error_norms.png")
    outputs["los_angle"] = plot_los_angle(result, outdir / "los_angle.png")
    outputs["visibility_gating"] = plot_visibility_and_gating(result, outdir / "visibility_gating.png")
    outputs["innovations"] = plot_innovations(result, outdir / "innovations.png")

    outputs["image_plane_detections"] = plot_image_plane_detections(result, outdir / "image_plane_detections.png")
    outputs["detection_timeline"] = plot_detection_timeline(result, outdir / "detection_timeline.png")
    outputs["camera_strip"] = plot_camera_strip(result, outdir / "camera_strip.png")
    outputs["dashboard"] = plot_run_dashboard(result, outdir / "dashboard.png")
    outputs["pixel_meas_vs_pred_timeline"] = plot_predicted_vs_measured_pixels(
        result, outdir / "pixel_meas_vs_pred_timeline.png"
    )
    outputs["image_plane_meas_vs_pred"] = plot_image_plane_measured_vs_predicted(
        result, outdir / "image_plane_meas_vs_pred.png"
    )

    if hypotheses is not None:
        outputs["hypotheses"] = plot_hypothesis_summary(hypotheses, outdir / "hypotheses.png")

    return outputs
