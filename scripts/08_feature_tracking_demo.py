from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from _common import ensure_src_on_path, repo_path

ensure_src_on_path()

from dynamics.cr3bp import CR3BP
from dynamics.integrators import propagate
from cv.pointing import camera_dcm_from_boresight
from visualization.style import (
    AMBER as _AMBER,
    BG as _BG,
    BORDER as _BORDER,
    CYAN as _CYAN,
    DIM as _DIM,
    GREEN as _GREEN,
    ORANGE as _ORANGE,
    PANEL as _PANEL,
    RED as _RED,
    TEXT as _TEXT,
    VIOLET as _VIOLET,
    apply_dark_theme,
    plt,
)
from matplotlib.lines import Line2D  # noqa: E402


_PINK   = "#F472B6"

_R_MOON_ND   = 0.00452
_EM_DISTANCE_KM = 384400.0
_TEXTURE_URL = (
    "https://s3-us-west-2.amazonaws.com/s.cdpn.io/17271/lroc_color_poles_1k.jpg"
)

_TRAIL_LEN = 24
_TRACK_PALETTE = [_GREEN, _CYAN, _AMBER, _ORANGE, _VIOLET, _PINK, _RED]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a synthetic Moon-imaging sequence with real KLT feature "
            "tracking and known-pose triangulation (SLAM mapping)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--plots-dir", type=Path, default=Path("results/demos"))
    parser.add_argument("--videos-dir", type=Path, default=Path("results/videos"))
    parser.add_argument("--metrics-csv", type=Path, default=Path("results/vision/08_feature_metrics.csv"))
    parser.add_argument("--texture-path", type=Path, default=Path("results/seeds/moon_texture.jpg"))
    parser.add_argument("--download-texture", action="store_true", help="Download the optional Moon texture if missing.")
    parser.add_argument("--duration", type=float, default=14.0, help="CR3BP propagation duration in dimensionless time.")
    parser.add_argument("--fps", type=int, default=24, help="Output video playback frame rate.")
    parser.add_argument("--slam-panel", action="store_true", default=True,
                        help="Render a right-side 3D SLAM-map panel beside the image plane (default on).")
    parser.add_argument("--no-slam-panel", dest="slam_panel", action="store_false",
                        help="Disable the right-side SLAM-map panel.")
    parser.add_argument("--highlight", action="store_true",
                        help="Also emit a short highlight clip (middle 40%% of the arc).")
    parser.add_argument("--slowmo", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=420, help="Cap rendered frames for practical runtime.")
    parser.add_argument("--strip-count", type=int, default=8)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-features", type=int, default=120)
    parser.add_argument("--min-features", type=int, default=60)
    parser.add_argument("--feature-quality", type=float, default=0.008)
    parser.add_argument("--feature-min-dist", type=int, default=6)
    parser.add_argument("--fb-threshold", type=float, default=1.0)
    parser.add_argument("--klt-win", type=int, default=15)
    parser.add_argument("--klt-levels", type=int, default=3)
    parser.add_argument("--triangulate-min-obs", type=int, default=6)
    parser.add_argument("--triangulate-min-baseline-px", type=float, default=4.0)
    parser.add_argument("--triangulate-max-rms-px", type=float, default=2.0,
                        help="Reject triangulated landmarks with reproj RMS above this threshold.")
    return parser.parse_args()


def _procedural_texture(size: int = 1024, seed: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    x = (x / size - 0.5) * 2.0
    y = (y / size - 0.5) * 2.0
    r = np.sqrt(x * x + y * y)
    broad = 0.52 + 0.12 * np.sin(12.0 * x + 5.0 * y) + 0.08 * np.cos(19.0 * y)
    craters = np.zeros_like(broad)
    for _ in range(90):
        cx, cy = rng.uniform(-1.0, 1.0, size=2)
        rad = rng.uniform(0.012, 0.06)
        amp = rng.uniform(-0.18, 0.10)
        craters += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * rad * rad))
    noise = rng.normal(0.0, 0.035, size=(size, size)).astype(np.float32)
    tex = np.clip(broad + craters + noise - 0.10 * r, 0.05, 0.95)
    return np.dstack([tex * 1.05, tex, tex * 0.92]).astype(np.float32)


def _load_texture(cache_path: Path, *, allow_download: bool = False) -> np.ndarray:
    if not cache_path.exists() and allow_download:
        print(f"Downloading Moon texture → {cache_path} ...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_TEXTURE_URL, str(cache_path))
        print("  Done.")
    if not cache_path.exists():
        print("Moon texture not found; using procedural deterministic texture.")
        return _procedural_texture()

    img = cv2.imread(str(cache_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Could not load texture {cache_path}; using procedural deterministic texture.")
        return _procedural_texture()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0



def _render_moon(
    *,
    cam_w: int, cam_h: int,
    u_c: float, v_c: float, r_px: float,
    R_cam: np.ndarray,
    r_body: np.ndarray, r_sc: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    tex: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    frame = np.zeros((cam_h, cam_w, 3), dtype=np.float32)
    s = rng.random((cam_h, cam_w)) < 0.0006
    frame[s] = rng.uniform(0.4, 1.0, (s.sum(), 3))

    if r_px < 0.8:
        ui, vi = int(round(u_c)), int(round(v_c))
        if 0 <= ui < cam_w and 0 <= vi < cam_h:
            frame[vi, ui] = [1.0, 1.0, 1.0]
        frame += rng.normal(0.0, 0.008, (cam_h, cam_w, 3)).astype(np.float32)
        return cv2.cvtColor(
            (np.clip(frame, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        )

    th, tw = tex.shape[:2]
    pad = max(4, int(r_px * 0.05))
    u0 = max(0, int(u_c - r_px) - pad); u1 = min(cam_w, int(u_c + r_px) + pad + 1)
    v0 = max(0, int(v_c - r_px) - pad); v1 = min(cam_h, int(v_c + r_px) + pad + 1)

    bore      = r_body - r_sc
    moon_cam  = R_cam @ bore
    r_sph     = _R_MOON_ND
    sun_cam   = R_cam @ np.array([-1.0, 0.0, 0.0])
    sun_cam  /= np.linalg.norm(sun_cam)

    uu, vv = np.meshgrid(np.arange(u0, u1, dtype=np.float32),
                          np.arange(v0, v1, dtype=np.float32))
    rdx = (uu - cx) / fx; rdy = (vv - cy) / fy
    rdz = np.ones_like(rdx)
    rlen = np.sqrt(rdx**2 + rdy**2 + rdz**2)
    rdx /= rlen; rdy /= rlen; rdz /= rlen

    mc = moon_cam.astype(np.float32)
    b  = -2.0 * (rdx * mc[0] + rdy * mc[1] + rdz * mc[2])
    c_ = float(mc[0]**2 + mc[1]**2 + mc[2]**2 - r_sph**2)
    disc = b**2 - 4.0 * c_
    hit  = disc >= 0.0
    if not hit.any():
        frame += rng.normal(0.0, 0.008, (cam_h, cam_w, 3)).astype(np.float32)
        return cv2.cvtColor(
            (np.clip(frame, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        )

    sq  = np.where(hit, np.sqrt(np.maximum(disc, 0.0)), 0.0)
    t   = np.where(hit, (-b - sq) / 2.0, 0.0)
    t   = np.where((t < 0) & hit, (-b + sq) / 2.0, t)
    hit = hit & (t > 0)

    hx = np.where(hit, rdx * t, 0.0)
    hy = np.where(hit, rdy * t, 0.0)
    hz = np.where(hit, rdz * t, 0.0)

    nx = np.where(hit, (hx - mc[0]) / r_sph, 0.0)
    ny = np.where(hit, (hy - mc[1]) / r_sph, 0.0)
    nz = np.where(hit, (hz - mc[2]) / r_sph, 0.0)

    diff    = np.clip(nx * sun_cam[0] + ny * sun_cam[1] + nz * sun_cam[2], 0, 1)
    shading = 0.12 + 0.88 * diff

    # Texture lookup must use WORLD-frame surface normals, otherwise the
    # texture slides under the camera as R_cam rotates (boresight tracking),
    # producing features that aren't anchored to fixed 3D surface points.
    R_world_from_cam = R_cam.T.astype(np.float32)
    nwx = R_world_from_cam[0, 0] * nx + R_world_from_cam[0, 1] * ny + R_world_from_cam[0, 2] * nz
    nwy = R_world_from_cam[1, 0] * nx + R_world_from_cam[1, 1] * ny + R_world_from_cam[1, 2] * nz
    nwz = R_world_from_cam[2, 0] * nx + R_world_from_cam[2, 1] * ny + R_world_from_cam[2, 2] * nz

    lon = np.where(hit, np.arctan2(nwy, nwx), 0.0)
    lat = np.where(hit, np.arcsin(np.clip(nwz, -1, 1)), 0.0)
    tu  = np.clip(((lon / (2 * np.pi) + 0.5) * (tw - 1)).astype(int), 0, tw - 1)
    tv  = np.clip(((0.5 - lat / np.pi) * (th - 1)).astype(int), 0, th - 1)

    color = np.where(hit[:, :, None], tex[tv, tu] * shading[:, :, None], 0.0)
    frame[v0:v1, u0:u1] = color
    frame += rng.normal(0.0, 0.012, (cam_h, cam_w, 3)).astype(np.float32)
    return cv2.cvtColor(
        (np.clip(frame, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
    )



def _detect(frame_bgr: np.ndarray) -> dict:
    gray = cv2.GaussianBlur(
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0
    )
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"found": False, "mask": None}
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 3:
        return {"found": False, "mask": None}
    M = cv2.moments(best)
    if abs(M["m00"]) < 1e-9:
        return {"found": False, "mask": None}
    (ecx, ecy), er = cv2.minEnclosingCircle(best)
    x, y, w, h = cv2.boundingRect(best)

    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [best], -1, 255, thickness=cv2.FILLED)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)

    return {
        "found":    True,
        "centroid": (M["m10"] / M["m00"], M["m01"] / M["m00"]),
        "contour":  best,
        "bbox":     (x, y, w, h),
        "circle":   (float(ecx), float(ecy), float(er)),
        "mask":     mask,
    }


@dataclass
class _Track:
    tid: int
    uv: tuple[float, float]
    history: deque  # deque of (frame_idx, u, v)
    age: int = 0
    alive: bool = True
    color: str = _GREEN


class KLTTracker:
    def __init__(
        self,
        *,
        max_features: int,
        min_features: int,
        quality: float,
        min_distance: int,
        fb_threshold: float,
        win_size: int,
        pyr_levels: int,
    ) -> None:
        self.max_features = int(max_features)
        self.min_features = int(min_features)
        self.quality = float(quality)
        self.min_distance = int(min_distance)
        self.fb_threshold = float(fb_threshold)
        self.win_size = (int(win_size), int(win_size))
        self.pyr_levels = int(pyr_levels)
        self.criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01,
        )
        self.next_id = 0
        self.tracks: dict[int, _Track] = {}
        self.prev_gray: Optional[np.ndarray] = None
        self.history_all: dict[int, list[tuple[int, float, float]]] = {}
        self._palette_i = 0

    def _alive_ids(self) -> list[int]:
        return [i for i, t in self.tracks.items() if t.alive]

    def _seed_new(
        self,
        gray: np.ndarray,
        mask: Optional[np.ndarray],
        frame_idx: int,
    ) -> None:
        alive = self._alive_ids()
        n_need = self.max_features - len(alive)
        if n_need <= 0:
            return
        exclusion = np.full_like(gray, 255, dtype=np.uint8)
        if mask is not None:
            exclusion = cv2.bitwise_and(exclusion, mask)
        for tid in alive:
            u, v = self.tracks[tid].uv
            cv2.circle(exclusion, (int(u), int(v)), self.min_distance, 0, -1)
        if int(np.count_nonzero(exclusion)) < 16:
            return
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=n_need,
            qualityLevel=self.quality,
            minDistance=self.min_distance,
            mask=exclusion,
            blockSize=5,
            useHarrisDetector=False,
        )
        if pts is None:
            return
        for p in pts.reshape(-1, 2):
            u, v = float(p[0]), float(p[1])
            tid = self.next_id
            self.next_id += 1
            color = _TRACK_PALETTE[self._palette_i % len(_TRACK_PALETTE)]
            self._palette_i += 1
            track = _Track(
                tid=tid,
                uv=(u, v),
                history=deque([(frame_idx, u, v)], maxlen=_TRAIL_LEN),
                age=0,
                alive=True,
                color=color,
            )
            self.tracks[tid] = track
            self.history_all[tid] = [(frame_idx, u, v)]

    def update(
        self,
        frame_bgr: np.ndarray,
        mask: Optional[np.ndarray],
        frame_idx: int,
    ) -> list[_Track]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        alive = self._alive_ids()
        if self.prev_gray is not None and alive:
            prev_pts = np.array(
                [self.tracks[i].uv for i in alive], dtype=np.float32
            ).reshape(-1, 1, 2)

            next_pts, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, prev_pts, None,
                winSize=self.win_size, maxLevel=self.pyr_levels,
                criteria=self.criteria,
            )
            back_pts, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
                gray, self.prev_gray, next_pts, None,
                winSize=self.win_size, maxLevel=self.pyr_levels,
                criteria=self.criteria,
            )
            fb_err = np.linalg.norm(
                (prev_pts - back_pts).reshape(-1, 2), axis=1,
            )

            h, w = gray.shape
            for k, tid in enumerate(alive):
                ok_flow = bool(st_fwd[k]) and bool(st_bwd[k])
                ok_fb   = fb_err[k] < self.fb_threshold
                if not (ok_flow and ok_fb):
                    self.tracks[tid].alive = False
                    continue
                u, v = float(next_pts[k, 0, 0]), float(next_pts[k, 0, 1])
                if not (1.0 <= u < w - 1.0 and 1.0 <= v < h - 1.0):
                    self.tracks[tid].alive = False
                    continue
                if mask is not None:
                    ui, vi = int(round(u)), int(round(v))
                    if mask[vi, ui] == 0:
                        self.tracks[tid].alive = False
                        continue
                tk = self.tracks[tid]
                tk.uv = (u, v)
                tk.age += 1
                tk.history.append((frame_idx, u, v))
                self.history_all[tid].append((frame_idx, u, v))

        n_alive = len(self._alive_ids())
        if (
            self.prev_gray is None
            or n_alive < self.min_features
        ):
            self._seed_new(gray, mask, frame_idx)

        self.prev_gray = gray
        return [t for t in self.tracks.values() if t.alive]


def _projection_matrix(
    R_cam: np.ndarray, r_sc: np.ndarray, K: np.ndarray,
) -> np.ndarray:
    Rt = np.zeros((3, 4), dtype=float)
    Rt[:, :3] = R_cam
    Rt[:,  3] = -R_cam @ r_sc
    return K @ Rt


def _triangulate_nview(
    Ps: list[np.ndarray],
    uvs: list[tuple[float, float]],
) -> np.ndarray:
    n = len(Ps)
    A = np.zeros((2 * n, 4), dtype=float)
    for i, (P, (u, v)) in enumerate(zip(Ps, uvs)):
        A[2 * i + 0] = u * P[2] - P[0]
        A[2 * i + 1] = v * P[2] - P[1]
    try:
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.array([np.nan, np.nan, np.nan])
    X_h = Vt[-1]
    if abs(X_h[3]) < 1e-15:
        return np.array([np.nan, np.nan, np.nan])
    return X_h[:3] / X_h[3]


def _reproject(
    P: np.ndarray, X_world: np.ndarray,
) -> tuple[float, float] | None:
    X_h = np.array([X_world[0], X_world[1], X_world[2], 1.0], dtype=float)
    uvw = P @ X_h
    if uvw[2] <= 0:
        return None
    return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])


def _triangulate_tracks(
    tracker: KLTTracker,
    *,
    K: np.ndarray,
    pose_R: list[np.ndarray],
    pose_r: list[np.ndarray],
    min_obs: int,
    min_baseline_px: float,
) -> dict[int, dict]:
    results: dict[int, dict] = {}
    proj_cache: dict[int, np.ndarray] = {}

    def P_at(idx: int) -> np.ndarray:
        if idx not in proj_cache:
            proj_cache[idx] = _projection_matrix(pose_R[idx], pose_r[idx], K)
        return proj_cache[idx]

    for tid, obs in tracker.history_all.items():
        if len(obs) < min_obs:
            continue
        uv = np.array([(u, v) for (_, u, v) in obs], dtype=float)
        baseline = float(np.linalg.norm(uv.max(axis=0) - uv.min(axis=0)))
        if baseline < min_baseline_px:
            continue

        Ps  = [P_at(fi) for (fi, _u, _v) in obs]
        uvs = [(u, v)    for (_fi, u, v) in obs]
        X   = _triangulate_nview(Ps, uvs)
        if not np.all(np.isfinite(X)):
            continue

        residuals = []
        for P, (u, v) in zip(Ps, uvs):
            proj = _reproject(P, X)
            if proj is None:
                continue
            residuals.append(np.hypot(proj[0] - u, proj[1] - v))
        if not residuals:
            continue
        rms = float(np.sqrt(np.mean(np.square(residuals))))

        results[tid] = {
            "X_world": X,
            "n_obs":   len(obs),
            "baseline_px": baseline,
            "rms_reproj_px": rms,
        }
    return results



def _detection_metrics(
    *,
    t_k: float,
    det: dict,
    u_true: float,
    v_true: float,
    r_px_true: float,
    range_nd_true: float,
    fx: float,
    n_tracks_alive: int,
    mean_track_age: float,
    mean_flow_px: float,
    max_flow_px: float,
) -> dict[str, float | int]:
    if not det["found"]:
        return {
            "t_dimensionless": float(t_k),
            "detected": 0,
            "centroid_error_px": float("nan"),
            "radius_error_px": float("nan"),
            "angular_radius_mrad": float("nan"),
            "range_true_km": float(range_nd_true * _EM_DISTANCE_KM),
            "range_proxy_km": float("nan"),
            "range_proxy_error_km": float("nan"),
            "range_proxy_dimensionless": float("nan"),
            "range_proxy_rel_error": float("nan"),
            "tracks_alive": int(n_tracks_alive),
            "mean_track_age_frames": float(mean_track_age),
            "mean_track_flow_px": float(mean_flow_px),
            "max_track_flow_px": float(max_flow_px),
        }

    cx_det, cy_det = det["centroid"]
    _, _, r_px_det = det["circle"]
    alpha = float(r_px_det / max(1e-12, fx))
    range_proxy = float(_R_MOON_ND / max(alpha, 1e-12))
    range_true_km = float(range_nd_true * _EM_DISTANCE_KM)
    range_proxy_km = float(range_proxy * _EM_DISTANCE_KM)
    centroid_error = float(np.hypot(cx_det - u_true, cy_det - v_true))
    radius_error = float(r_px_det - r_px_true)
    rel_range_error = float((range_proxy - range_nd_true) / max(range_nd_true, 1e-12))
    return {
        "t_dimensionless": float(t_k),
        "detected": 1,
        "centroid_error_px": centroid_error,
        "radius_error_px": radius_error,
        "angular_radius_mrad": float(alpha * 1e3),
        "range_true_km": range_true_km,
        "range_proxy_km": range_proxy_km,
        "range_proxy_error_km": float(range_proxy_km - range_true_km),
        "range_proxy_dimensionless": range_proxy,
        "range_proxy_rel_error": rel_range_error,
        "tracks_alive": int(n_tracks_alive),
        "mean_track_age_frames": float(mean_track_age),
        "mean_track_flow_px": float(mean_flow_px),
        "max_track_flow_px": float(max_flow_px),
    }


def _write_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No metrics to write.")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _median_finite(vals: np.ndarray) -> float:
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float("nan")


def _plot_metrics(
    rows: list[dict],
    outpath: Path,
    *,
    triangulation: dict[int, dict],
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    t = np.array([r["t_dimensionless"] for r in rows], dtype=float)
    centroid = np.array([r["centroid_error_px"] for r in rows], dtype=float)
    range_rel = np.array([r["range_proxy_rel_error"] for r in rows], dtype=float) * 100.0
    tracks_alive = np.array([r["tracks_alive"] for r in rows], dtype=float)
    flow = np.array([r["mean_track_flow_px"] for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor(_BG)
    for ax in axes.flat:
        ax.set_facecolor(_PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(_BORDER)
        ax.grid(True, color=_BORDER, alpha=0.7)
        ax.tick_params(colors=_TEXT)

    axes[0, 0].plot(t, centroid, color=_CYAN)
    axes[0, 0].set_title("Moon Centroid Error", color=_TEXT)
    axes[0, 0].set_ylabel("error [px]", color=_TEXT)

    axes[0, 1].plot(t, range_rel, color=_VIOLET)
    axes[0, 1].axhline(0.0, color=_BORDER, lw=1.0)
    axes[0, 1].set_title("Angular-Radius Range Proxy Error", color=_TEXT)
    axes[0, 1].set_ylabel("relative error [%]", color=_TEXT)

    axes[1, 0].plot(t, tracks_alive, color=_GREEN)
    axes[1, 0].set_title("Active KLT Tracks", color=_TEXT)
    axes[1, 0].set_ylabel("count", color=_TEXT)
    axes[1, 0].set_xlabel("time [dimensionless CR3BP units]", color=_TEXT)

    axes[1, 1].plot(t, flow, color=_AMBER)
    axes[1, 1].set_title("Mean Track Optical Flow", color=_TEXT)
    axes[1, 1].set_ylabel("flow [px/frame]", color=_TEXT)
    axes[1, 1].set_xlabel("time [dimensionless CR3BP units]", color=_TEXT)

    rms_vals = np.array(
        [r["rms_reproj_px"] for r in triangulation.values()], dtype=float,
    ) if triangulation else np.array([])

    finite_centroid = centroid[np.isfinite(centroid)]
    finite_range = range_rel[np.isfinite(range_rel)]
    tri_str = (
        f"  |  triangulated landmarks = {rms_vals.size}"
        f", median reproj RMS = {_median_finite(rms_vals):.2f} px"
        if rms_vals.size else "  |  triangulated landmarks = 0"
    )
    summary = (
        f"median centroid error = {_median_finite(finite_centroid):.2f} px  "
        f"|  median range-proxy error = {_median_finite(finite_range):+.2f}%"
        f"{tri_str}"
    )
    fig.suptitle(
        "08 Vision Metrics: Moon centroid, angular size, and KLT feature tracks\n"
        + summary,
        color=_TEXT,
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(outpath, dpi=220, facecolor=_BG)
    plt.close(fig)


def _compute_reveal_frames(
    tracker: "KLTTracker",
    triangulation: dict[int, dict],
    min_obs: int,
) -> dict[int, int]:
    """For each triangulated landmark, the frame index at which its
    min_obs-th observation was captured. Used to "reveal" landmarks
    during the video as KLT tracks mature."""
    reveal: dict[int, int] = {}
    for tid in triangulation:
        obs = tracker.history_all.get(int(tid), [])
        if len(obs) >= int(min_obs):
            reveal[int(tid)] = int(obs[int(min_obs) - 1][0])
    return reveal


def _render_slam_panel_frame(
    *,
    out_path: Path,
    width_px: int,
    height_px: int,
    t_k: float,
    r_sc_k: np.ndarray,
    r_moon: np.ndarray,
    slam_pts_k: np.ndarray,
    slam_rms_k: np.ndarray,
    tex: np.ndarray,
    n_total: int,
    final_rms_median: float,
) -> None:
    """Render the SLAM-map panel for one frame. Small textured Moon at
    origin, accumulated triangulated landmarks scattered in world frame
    around it (coloured by reproj RMS), spacecraft position as a cyan
    dot, axes in lunar radii."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi),
                      dpi=dpi, facecolor=_BG)
    ax = fig.add_axes([0.04, 0.07, 0.92, 0.82], projection="3d")
    ax.set_facecolor(_PANEL)
    pane_rgba = (0.020, 0.031, 0.063, 1.0)
    grid_rgba = (0.10, 0.13, 0.25, 0.22)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(pane_rgba)
        axis.label.set_color(_TEXT)
        axis._axinfo["grid"]["color"] = grid_rgba
        axis._axinfo["grid"]["linewidth"] = 0.35
    ax.tick_params(colors=_DIM, labelsize=6)

    # Textured Moon at origin (in lunar-radius units)
    n_sph = 48
    u = np.linspace(0.0, 2.0 * np.pi, n_sph)
    v = np.linspace(0.0, np.pi, n_sph)
    cu, su = np.cos(u), np.sin(u)
    cv, sv = np.cos(v), np.sin(v)
    xs = np.outer(cu, sv)
    ys = np.outer(su, sv)
    zs = np.outer(np.ones_like(u), cv)
    if tex.dtype.kind in ("u", "i"):
        tex_f = tex.astype(np.float32) / 255.0
    else:
        tex_f = tex
    H, W = tex_f.shape[:2]
    uu_f = 0.5 * (u[:-1] + u[1:])
    vv_f = 0.5 * (v[:-1] + v[1:])
    Uc, Vc = np.meshgrid(uu_f, vv_f, indexing="ij")
    lon = (Uc + np.pi) % (2.0 * np.pi)
    px_i = np.clip((lon / (2.0 * np.pi) * W).astype(int), 0, W - 1)
    py_i = np.clip((Vc /        np.pi * H).astype(int), 0, H - 1)
    face_rgba = np.empty((n_sph - 1, n_sph - 1, 4), dtype=float)
    face_rgba[..., :3] = tex_f[py_i, px_i, :3]
    face_rgba[..., 3]  = 1.0
    ax.plot_surface(xs, ys, zs, facecolors=face_rgba,
                    shade=False, linewidth=0, antialiased=False,
                    rstride=1, cstride=1, zorder=3)

    # Accumulated landmarks (world frame shifted so Moon = origin; in R_moon units)
    if slam_pts_k.size > 0:
        P = (slam_pts_k - r_moon[None, :]) / _R_MOON_ND
        sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2],
                        c=slam_rms_k, cmap="plasma",
                        vmin=0.0, vmax=max(2.0, float(final_rms_median) * 1.5),
                        s=24, depthshade=False, zorder=6,
                        edgecolor=_TEXT, linewidth=0.3)

    # (Earlier versions drew the spacecraft + its LOS ray here, but the
    # spacecraft's Moon-relative range swings well outside the ±3.5 R_moon
    # plot box during most of the arc — producing a "ball on a string"
    # artifact that fought the map-building story. The SLAM panel now
    # shows only the map and the Moon.)

    lim = 3.5
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_xlabel("x  [R☾]", color=_TEXT, fontsize=7, labelpad=-2)
    ax.set_ylabel("y  [R☾]", color=_TEXT, fontsize=7, labelpad=-2)
    ax.set_zlabel("z  [R☾]", color=_TEXT, fontsize=7, labelpad=-2)

    # Slow camera rotation for depth perception
    ax.view_init(elev=22.0, azim=-55.0 + 55.0 * min(1.0, t_k / 14.0))

    n_shown = 0 if slam_pts_k.size == 0 else slam_pts_k.shape[0]
    fig.text(0.04, 0.95,
             f"SLAM MAP  ·  t = {t_k:5.2f}  ·  {n_shown:3d} / {n_total} landmarks",
             color=_TEXT, fontsize=10, family="monospace", fontweight="bold")
    fig.text(0.04, 0.92,
             "triangulated KLT landmarks  ·  coloured by reproj RMS",
             color=_DIM, fontsize=8, family="monospace")

    fig.savefig(out_path, dpi=dpi, facecolor=_BG)
    plt.close(fig)


def _plot_triangulation_map(
    triangulation: dict[int, dict],
    r_moon: np.ndarray,
    r_true: np.ndarray,
    outpath: Path,
) -> None:
    if not triangulation:
        return
    pts = np.array(
        [v["X_world"] for v in triangulation.values()], dtype=float,
    )
    rms = np.array(
        [v["rms_reproj_px"] for v in triangulation.values()], dtype=float,
    )
    dist_from_moon = np.linalg.norm(pts - r_moon[None, :], axis=1)
    radial_err_km = (dist_from_moon - _R_MOON_ND) * _EM_DISTANCE_KM

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 5.2))
    fig.patch.set_facecolor(_BG)

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.set_facecolor(_PANEL)

    u = np.linspace(0, 2 * np.pi, 64)
    v = np.linspace(0, np.pi, 32)
    ax.plot_surface(
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
        color=_BORDER, alpha=0.22, linewidth=0,
    )
    sc = ax.scatter(
        (pts[:, 0] - r_moon[0]) / _R_MOON_ND,
        (pts[:, 1] - r_moon[1]) / _R_MOON_ND,
        (pts[:, 2] - r_moon[2]) / _R_MOON_ND,
        c=rms, cmap="plasma", s=18, depthshade=False,
    )
    lim = 2.2
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x  [R☾]", color=_TEXT)
    ax.set_ylabel("y  [R☾]", color=_TEXT)
    ax.set_zlabel("z  [R☾]", color=_TEXT)
    ax.set_title(
        "Triangulated Landmarks (SLAM map)\n"
        "origin at Moon center, axes in lunar radii",
        color=_TEXT, fontsize=10,
    )
    ax.tick_params(colors=_TEXT)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.04)
    cb.set_label("reproj RMS [px]", color=_TEXT)
    cb.ax.yaxis.set_tick_params(color=_TEXT)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=_TEXT)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor(_PANEL)
    for sp in ax2.spines.values():
        sp.set_edgecolor(_BORDER)
    ax2.grid(True, color=_BORDER, alpha=0.6)
    ax2.hist(radial_err_km, bins=24, color=_GREEN, alpha=0.85)
    ax2.axvline(0.0, color=_RED, lw=1.0)
    ax2.set_title(
        "Radial error: |X − r_moon| − R_moon\n"
        f"N = {pts.shape[0]}   median = {np.median(radial_err_km):+.0f} km",
        color=_TEXT, fontsize=10,
    )
    ax2.set_xlabel("km (relative to lunar surface)", color=_TEXT)
    ax2.set_ylabel("count", color=_TEXT)
    ax2.tick_params(colors=_TEXT)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, facecolor=_BG)
    plt.close(fig)



def _draw_combined_panel(
    ax: plt.Axes,
    frame_bgr: np.ndarray,
    det: dict,
    tracks: list[_Track],
    t: float,
    r_px: float,
    fx: float,
    cam_w: int, cam_h: int,
) -> None:
    ax.set_facecolor("#050709")
    ax.imshow(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), origin="upper",
               extent=[0, cam_w, cam_h, 0])
    ax.set_xlim(0, cam_w); ax.set_ylim(cam_h, 0); ax.set_axis_off()

    if det["found"]:
        cx2, cy2     = det["centroid"]
        ecx, ecy, er = det["circle"]
        x, y, w, h   = det["bbox"]

        pts = det["contour"].reshape(-1, 2).astype(float)
        ax.plot(pts[:, 0], pts[:, 1], color=_AMBER, lw=1.1, alpha=0.85, zorder=4)

        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False,
                                    edgecolor=_GREEN, lw=0.9, ls="--", zorder=5))

        ax.add_patch(plt.Circle((ecx, ecy), er, fill=False,
                                 edgecolor=_VIOLET, lw=1.1, alpha=0.85, zorder=5))

        ax.scatter([cx2], [cy2], s=40, color=_CYAN, zorder=6)
        ax.plot([cx2-6, cx2+6], [cy2, cy2], color=_CYAN, lw=1.0, zorder=6)
        ax.plot([cx2, cx2], [cy2-6, cy2+6], color=_CYAN, lw=1.0, zorder=6)

        ax.annotate("", xy=(ecx+er, ecy), xytext=(ecx, ecy),
                    arrowprops=dict(arrowstyle="-", color=_VIOLET, lw=0.8), zorder=5)

        alpha_rad = er / fx
        lx = min(ecx + er + 4, cam_w - 90)
        ax.text(lx, ecy,
                f"α = {alpha_rad*1e3:.1f} mrad\n"
                f"ρ ≈ {_R_MOON_ND/max(alpha_rad,1e-9)*_EM_DISTANCE_KM:.0f} km",
                color=_RED, fontsize=8, family="monospace", va="center",
                bbox=dict(facecolor=(0,0,0,0.65), edgecolor="none", pad=3), zorder=8)

    # Cap to top-N by age so the Moon stays legible during close-approach
    # frames; sort newest-first is a noisy disco of overlapping rings.
    MAX_TRACKS_SHOWN = 35
    tracks_to_draw = (
        sorted(tracks, key=lambda t: -t.age)[:MAX_TRACKS_SHOWN]
        if len(tracks) > MAX_TRACKS_SHOWN else list(tracks)
    )

    for tk in tracks_to_draw:
        col = tk.color
        hist = list(tk.history)
        if len(hist) >= 2:
            arr = np.array([(u, v) for (_, u, v) in hist], dtype=float)
            n = len(arr)
            for j in range(1, n):
                alpha_t = 0.12 + 0.70 * (j / n)
                ax.plot(arr[j-1:j+1, 0], arr[j-1:j+1, 1],
                        color=col, lw=1.0, alpha=alpha_t, zorder=4)

        u, v = tk.uv
        r_mark = 3.5 if tk.age < 4 else 5.0
        ax.scatter([u], [v], s=(r_mark * 4.0), facecolors="none", edgecolors=col,
                   linewidths=1.0, zorder=7)

        if len(hist) >= 2:
            _, up, vp = hist[-2]
            du = u - up
            dv = v - vp
            speed = np.hypot(du, dv)
            if speed > 0.4 and r_px > 10:
                scale = min(3.0, 8.0 / max(speed, 1e-3))
                ax.annotate(
                    "",
                    xy=(u + du * scale, v + dv * scale),
                    xytext=(u, v),
                    arrowprops=dict(
                        arrowstyle="-|>", color=col, lw=0.8, mutation_scale=5,
                    ),
                    zorder=8,
                )

    n_tracks = len(tracks)
    mean_age = float(np.mean([tk.age for tk in tracks])) if tracks else 0.0
    ax.text(
        5, 13,
        f"t = {t:.3f} dim  |  {n_tracks} KLT tracks  |  mean age {mean_age:.0f} f",
        color=_AMBER, fontsize=9, family="monospace",
        bbox=dict(facecolor=(0, 0, 0, 0.55), edgecolor="none", pad=4),
    )

    ax.set_title(
        "Moon centroid + angular radius (blob) · real KLT feature tracks\n"
        "u, v, α → LOS + range proxy  ·  KLT flow → mono-SLAM front-end",
        color=_TEXT, fontsize=9, pad=4,
    )



def _make_frame_fig(
    frame_bgr: np.ndarray,
    det: dict,
    tracks: list[_Track],
    t: float,
    r_px: float,
    fx: float,
    cam_w: int, cam_h: int,
) -> plt.Figure:
    dpi   = 100
    fig_w = (cam_w + 20) / dpi
    fig_h = (cam_h + 72) / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=_BG)
    ax  = fig.add_axes([0.0, 0.09, 1.0, 0.85])

    _draw_combined_panel(
        ax, frame_bgr, det, tracks, t, r_px, fx, cam_w, cam_h,
    )

    fig.legend(handles=[
        Line2D([0],[0], color=_CYAN,  lw=2, label="centroid (u,v)"),
        Line2D([0],[0], color=_AMBER, lw=2, label="blob contour"),
        Line2D([0],[0], color=_VIOLET,lw=1.5, label="enclosing circle (α)"),
        Line2D([0],[0], color=_RED,   lw=0, marker="s", markersize=7,
               label="ρ ≈ R☾/α"),
        Line2D([0],[0], color=_GREEN, lw=2, label="KLT track history"),
        Line2D([0],[0], color=_ORANGE,lw=0, marker=">", markersize=7,
               label="KLT optical flow"),
    ], loc="lower center", ncol=6, fontsize=7.5,
       facecolor=_PANEL, edgecolor=_BORDER, labelcolor=_TEXT,
       bbox_to_anchor=(0.5, 0.0))
    return fig



def _run_ffmpeg(frames_dir: Path, out: Path, fps: int) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _run_ffmpeg_stack(a_dir: Path, b_dir: Path, out: Path, fps: int,
                       *, orientation: str = "vstack") -> None:
    """Combine two frame sequences into one video. orientation='vstack'
    stacks a on top of b; orientation='hstack' puts a left of b."""
    if orientation not in ("vstack", "hstack"):
        raise ValueError(f"orientation must be 'vstack' or 'hstack', got {orientation!r}")
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps), "-i", str(a_dir / "frame_%05d.png"),
        "-framerate", str(fps), "-i", str(b_dir / "frame_%05d.png"),
        "-filter_complex",
        "[0:v]pad=ceil(iw/2)*2:ceil(ih/2)*2[a];"
        "[1:v]pad=ceil(iw/2)*2:ceil(ih/2)*2[b];"
        f"[a][b]{orientation}=inputs=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
        str(out),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _ffmpeg_trim(src: Path, out: Path, start_s: float, dur_s: float,
                  *, slowmo: float = 1.0) -> None:
    """Trim a video to [start_s, start_s+dur_s] seconds. If slowmo > 1,
    the clip also plays back that many times slower via PTS stretch
    (e.g. slowmo=1.5 makes a 6 s trim play as 9 s on screen).
    Re-encodes so the trimmed clip has keyframes at its own start."""
    # -ss and -t must be INPUT options (before -i) so ffmpeg seeks in
    # the source and caps *input* duration. Placing -t after -i caps
    # output duration instead, which silently truncates slow-mo clips.
    cmd = ["ffmpeg", "-y",
           "-ss", f"{start_s:.3f}", "-t", f"{dur_s:.3f}",
           "-i", str(src)]
    if float(slowmo) != 1.0:
        cmd.extend(["-vf", f"setpts={float(slowmo):.3f}*PTS"])
    cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
                 str(out)])
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)



def _track_flow_stats(tracker: KLTTracker) -> tuple[float, float, float]:
    flows = []
    ages  = []
    for tk in tracker.tracks.values():
        if not tk.alive:
            continue
        ages.append(tk.age)
        if len(tk.history) >= 2:
            _, up, vp = tk.history[-2]
            _, u,  v  = tk.history[-1]
            flows.append(float(np.hypot(u - up, v - vp)))
    if not flows:
        return float("nan"), float("nan"), (float(np.mean(ages)) if ages else 0.0)
    arr = np.asarray(flows, dtype=float)
    mean_age = float(np.mean(ages)) if ages else 0.0
    return float(np.mean(arr)), float(np.max(arr)), mean_age


def main() -> None:
    args = parse_args()
    apply_dark_theme()

    plots_dir = repo_path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = repo_path(args.videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = videos_dir / "08_feature_frames"
    if frames_dir.exists():
        for old in frames_dir.glob("frame_*.png"):
            try:
                old.unlink()
            except OSError:
                pass
    frames_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = repo_path(args.metrics_csv)

    tex = _load_texture(repo_path(args.texture_path), allow_download=bool(args.download_texture))

    mu     = 0.0121505856
    model  = CR3BP(mu=mu)
    L1x    = model.lagrange_points()["L1"][0]
    x0     = np.array([L1x - 1.2e-3, 2.5e-4, 0.0, 0.0, 0.045, 0.0], dtype=float)
    r_moon = np.asarray(model.primary2, dtype=float)

    fps = int(args.fps)
    dt = (1.0 / fps) / float(args.slowmo)
    t_arr = np.arange(0.0, float(args.duration) + 1e-12, dt)
    if args.max_frames is not None and t_arr.size > int(args.max_frames):
        t_arr = np.linspace(0.0, float(args.duration), int(args.max_frames))
    N      = len(t_arr)

    cam_w, cam_h = 640, 480
    fx = fy = 700.0
    cx, cy  = cam_w / 2.0, cam_h / 2.0
    K_cam = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float,
    )
    rng     = np.random.default_rng(int(args.seed))

    res = propagate(model.eom, (0.0, float(args.duration)), x0, t_eval=t_arr)
    if not res.success:
        raise RuntimeError(f"Propagation failed: {res.message}")
    r_true = res.x[:, :3]

    tracker = KLTTracker(
        max_features=args.max_features,
        min_features=args.min_features,
        quality=args.feature_quality,
        min_distance=args.feature_min_dist,
        fb_threshold=args.fb_threshold,
        win_size=args.klt_win,
        pyr_levels=args.klt_levels,
    )

    pose_R: list[np.ndarray] = []
    pose_r: list[np.ndarray] = []

    print(f"Rendering {N} frames ...")

    strip_frames: list[tuple] = []
    STRIP_N = int(args.strip_count)
    stride  = max(1, N // STRIP_N)
    metric_rows: list[dict] = []

    for k in range(N):
        t_k  = float(t_arr[k])
        r_sc = r_true[k]
        bore = r_moon - r_sc
        R_cam = camera_dcm_from_boresight(bore, camera_forward_axis="+z")
        lc    = R_cam @ bore

        pose_R.append(R_cam.copy())
        pose_r.append(r_sc.copy())

        if lc[2] <= 0:
            fb = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            u_c = v_c = cx; r_px = 0.0
            rng_nd = float(np.linalg.norm(bore))
        else:
            u_c   = float(fx * lc[0] / lc[2] + cx)
            v_c   = float(fy * lc[1] / lc[2] + cy)
            rng_nd = float(np.linalg.norm(bore))
            r_px   = (_R_MOON_ND / max(rng_nd, 1e-6)) * fx
            fb     = _render_moon(
                cam_w=cam_w, cam_h=cam_h,
                u_c=u_c, v_c=v_c, r_px=r_px,
                R_cam=R_cam, r_body=r_moon, r_sc=r_sc,
                fx=fx, fy=fy, cx=cx, cy=cy,
                tex=tex, rng=rng,
            )

        det = _detect(fb)
        tracks = tracker.update(fb, det.get("mask"), frame_idx=k)

        mean_flow, max_flow, mean_age = _track_flow_stats(tracker)

        metric_rows.append(
            _detection_metrics(
                t_k=t_k,
                det=det,
                u_true=u_c,
                v_true=v_c,
                r_px_true=r_px,
                range_nd_true=rng_nd,
                fx=fx,
                n_tracks_alive=len(tracks),
                mean_track_age=mean_age,
                mean_flow_px=mean_flow,
                max_flow_px=max_flow,
            )
        )

        fig = _make_frame_fig(fb, det, tracks, t_k, r_px, fx, cam_w, cam_h)
        fig.savefig(frames_dir / f"frame_{k:05d}.png", dpi=100, facecolor=_BG)
        plt.close(fig)

        if k % stride == 0 and det["found"] and det["circle"][2] > 2.0:
            if len(strip_frames) < STRIP_N:
                strip_frames.append((
                    fb.copy(), det,
                    [_Track(tid=tk.tid, uv=tk.uv,
                             history=deque(tk.history), age=tk.age,
                             alive=tk.alive, color=tk.color)
                     for tk in tracks],
                    t_k, r_px,
                ))

    print("Triangulating tracks against known camera poses ...")
    triangulation_raw = _triangulate_tracks(
        tracker,
        K=K_cam,
        pose_R=pose_R,
        pose_r=pose_r,
        min_obs=int(args.triangulate_min_obs),
        min_baseline_px=float(args.triangulate_min_baseline_px),
    )
    max_rms = float(args.triangulate_max_rms_px)
    triangulation = {
        tid: rec for tid, rec in triangulation_raw.items()
        if rec["rms_reproj_px"] <= max_rms
    }
    if triangulation_raw:
        rms_all = np.array(
            [v["rms_reproj_px"] for v in triangulation_raw.values()], dtype=float,
        )
        print(
            f"  {len(triangulation_raw)} raw / {len(triangulation)} kept"
            f" (RMS ≤ {max_rms:.1f} px)"
            f"  |  all-median RMS {np.median(rms_all):.2f} px"
            f"  |  P90 {np.percentile(rms_all, 90):.2f} px"
        )
    else:
        print("  No tracks met triangulation thresholds.")

    _write_metrics_csv(metrics_csv, metric_rows)
    metrics_plot = plots_dir / "08_feature_tracking_metrics.png"
    _plot_metrics(metric_rows, metrics_plot, triangulation=triangulation)
    print(f"Wrote metrics CSV: {metrics_csv}")
    print(f"Wrote metrics plot: {metrics_plot}")

    if triangulation:
        map_plot = plots_dir / "08_feature_tracking_map.png"
        _plot_triangulation_map(triangulation, r_moon, r_true, map_plot)
        print(f"Wrote SLAM map: {map_plot}")

    # Render side-by-side SLAM-map panel (right) to pair with the existing
    # image-plane panel (left). Landmarks accumulate over time as their
    # min_obs-th observation is captured, so the map visibly builds up.
    slam_frames_dir = videos_dir / "08_feature_slam_frames"
    if (
        bool(getattr(args, "slam_panel", True))
        and triangulation
        and not args.skip_video
    ):
        slam_frames_dir.mkdir(parents=True, exist_ok=True)
        reveal = _compute_reveal_frames(tracker, triangulation,
                                        min_obs=int(args.triangulate_min_obs))
        pts_all = np.array(
            [rec["X_world"] for rec in triangulation.values()], dtype=float,
        )
        rms_all = np.array(
            [rec["rms_reproj_px"] for rec in triangulation.values()], dtype=float,
        )
        tids    = np.array(list(triangulation.keys()), dtype=int)
        final_rms_median = float(np.median(rms_all)) if rms_all.size else 1.0

        reveal_arr = np.array(
            [reveal.get(int(t), N + 1) for t in tids], dtype=int,
        )

        slam_w = cam_w + 20
        slam_h = cam_h + 72
        print(f"Rendering {N} SLAM-map frames ...")
        for k in range(N):
            mask_k = reveal_arr <= k
            _render_slam_panel_frame(
                out_path = slam_frames_dir / f"frame_{k:05d}.png",
                width_px  = slam_w,
                height_px = slam_h,
                t_k       = float(t_arr[k]),
                r_sc_k    = r_true[k],
                r_moon    = r_moon,
                slam_pts_k = pts_all[mask_k] if pts_all.size else pts_all,
                slam_rms_k = rms_all[mask_k] if rms_all.size else rms_all,
                tex       = tex,
                n_total   = int(len(triangulation)),
                final_rms_median = final_rms_median,
            )

    if not args.skip_video and shutil.which("ffmpeg"):
        out_mp4 = videos_dir / "08_feature_tracking.mp4"
        if slam_frames_dir.exists() and any(slam_frames_dir.iterdir()):
            _run_ffmpeg_stack(frames_dir, slam_frames_dir, out_mp4, fps=fps,
                               orientation="vstack")
            print(f"Wrote side-by-side video: {out_mp4}  "
                  f"({N} frames @ {fps} fps = {N/fps:.1f} s)")
        else:
            _run_ffmpeg(frames_dir, out_mp4, fps=fps)
            print(f"Wrote video: {out_mp4}  ({N} frames @ {fps} fps = {N/fps:.1f} s)")

        if getattr(args, "highlight", False):
            highlight_mp4 = videos_dir / "08_feature_tracking_highlight.mp4"
            # Pick the landmark-accumulation sweet spot: frames where the
            # cumulative revealed count sits roughly in [64%, 87%] of the
            # final total (e.g. ~220 → ~300 for n_total = 344). That cuts
            # the fast "approach zoom" at the start and the "fly-away" at
            # the end, leaving the dense middle where the map is filling in.
            n_total_lm = int(len(triangulation)) if triangulation else 0
            if n_total_lm > 0:
                cum = np.zeros(N + 1, dtype=int)
                for r in reveal_arr:
                    if 0 <= r < N:
                        cum[r + 1:] += 1
                lo_count = int(round(0.64 * n_total_lm))
                hi_count = int(round(0.87 * n_total_lm))
                k_lo = int(np.searchsorted(cum, lo_count))
                k_hi = int(np.searchsorted(cum, hi_count))
                k_lo = max(0, min(N - 1, k_lo))
                k_hi = max(k_lo + 1, min(N - 1, k_hi))
                start_s = k_lo / fps
                dur_s   = (k_hi - k_lo) / fps
            else:
                total_s = N / fps
                start_s = 0.30 * total_s
                dur_s   = 0.40 * total_s
            # 1.5x slow-mo on top of the trim keeps the middle steady.
            slowmo = 1.5
            _ffmpeg_trim(out_mp4, highlight_mp4,
                         start_s=start_s, dur_s=dur_s, slowmo=slowmo)
            play_s = dur_s * slowmo
            print(f"Wrote highlight clip: {highlight_mp4}  "
                  f"(trim {start_s:.1f}s → {start_s + dur_s:.1f}s "
                  f"= {dur_s:.1f}s source, {slowmo}× slowmo → {play_s:.1f}s playback)")
    elif args.skip_video:
        print("Skipping video by request.")
    else:
        print("ffmpeg not found — skipping video.")

    if strip_frames:
        n = len(strip_frames)
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.0))
        fig.patch.set_facecolor(_BG)
        if n == 1: axes = [axes]

        for col, (fb, det, tks, t_k, rpx) in enumerate(strip_frames):
            ax = axes[col]
            _draw_combined_panel(
                ax, fb, det, tks, t_k, rpx, fx, cam_w, cam_h,
            )
            ax.set_title(
                f"t = {t_k:.1f} dim\n"
                + (f"α={det['circle'][2]/fx*1e3:.0f} mrad  |  {len(tks)} tracks"
                   if det["found"] else "—"),
                color=_TEXT, fontsize=7, pad=3,
            )

        fig.legend(handles=[
            Line2D([0],[0], color=_CYAN,  lw=2, label="centroid (u,v)"),
            Line2D([0],[0], color=_AMBER, lw=2, label="blob contour"),
            Line2D([0],[0], color=_VIOLET,lw=1.5, label="enclosing circle (α)"),
            Line2D([0],[0], color=_RED,   lw=0, marker="s", markersize=7,
                   label="ρ ≈ R☾/α"),
            Line2D([0],[0], color=_GREEN, lw=2, label="KLT tracks"),
        ], loc="lower center", ncol=5, fontsize=8,
           facecolor=_PANEL, edgecolor=_BORDER, labelcolor=_TEXT,
           bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(
            "Moon centroid + KLT feature tracks  |  Earth–Moon CR3BP",
            color=_TEXT, fontsize=11, y=1.02,
        )

        strip_path = plots_dir / "08_feature_tracking_strip.png"
        fig.savefig(strip_path, dpi=200, bbox_inches="tight", facecolor=_BG)
        plt.close(fig)
        print(f"Wrote strip: {strip_path}")

    print("08 feature tracking demo complete.")


if __name__ == "__main__":
    main()
