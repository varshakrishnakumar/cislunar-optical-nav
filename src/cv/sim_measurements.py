"""
cv/sim_measurements.py — simulate "CV-like" pixel detections

Goal
- Given truth geometry (spacecraft state, target position) and a camera model,
  output what CV would produce:
    - pixel measurement (u_px, v_px) and/or bounding box
    - pixel noise model (sigma_px)
- Pure simulation utilities; does NOT know EKF details.

Includes "later upgrades" (implemented here):
- bounding box simulation (sphere silhouette)
- detection dropouts
- centroid bias (toward limb)
- distortion (simple Brown–Conrady radial+tangential model)

Assumptions
- Global frame positions: r_sc, r_body in same frame.
- R_cam_from_frame maps global-frame vectors into camera frame.
- Pinhole camera intrinsics (from cv.camera.Intrinsics); distortion is optional here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union, Dict, Any

import numpy as np

from .camera import Intrinsics, los_cam_to_pixel, rotate_vector


ArrayLike = Union[float, int, np.ndarray]
OutOfFramePolicy = Literal["drop", "keep_invalid", "clamp"]
BehindPolicy = Literal["drop", "keep_invalid"]
NoiseMode = Literal["none", "gaussian"]


@dataclass(frozen=True, slots=True)
class Distortion:
    """
    Brown–Conrady distortion model (OpenCV-style parameters).
    All coefficients default to 0 (no distortion).

    Radial: x_d = x * (1 + k1 r^2 + k2 r^4 + k3 r^6)
    Tangential:
      x_d += 2 p1 x y + p2 (r^2 + 2 x^2)
      y_d += p1 (r^2 + 2 y^2) + 2 p2 x y
    """
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    def is_zero(self) -> bool:
        return (self.k1 == 0.0 and self.k2 == 0.0 and self.k3 == 0.0 and self.p1 == 0.0 and self.p2 == 0.0)


@dataclass(frozen=True, slots=True)
class PixelMeasurement:
    """Single simulated CV detection."""
    t: float
    u_px: float
    v_px: float
    sigma_px: float
    valid: bool
    bbox_xyxy: Optional[Tuple[float, float, float, float]] = None  # (xmin, ymin, xmax, ymax)
    meta: Optional[Dict[str, Any]] = None


def _as3(x: ArrayLike, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a


def _asR(R: Optional[ArrayLike]) -> np.ndarray:
    if R is None:
        return np.eye(3, dtype=np.float64)
    a = np.asarray(R, dtype=np.float64)
    if a.shape != (3, 3):
        raise ValueError(f"R_cam_from_frame must have shape (3,3), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("R_cam_from_frame must be finite")
    return a


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= eps or not np.isfinite(n):
        return np.full((3,), np.nan, dtype=np.float64)
    return v / n


def _apply_distortion(x: np.ndarray, y: np.ndarray, d: Distortion) -> Tuple[np.ndarray, np.ndarray]:
    # x,y are normalized coordinates
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + d.k1 * r2 + d.k2 * r4 + d.k3 * r6

    x_tan = 2.0 * d.p1 * x * y + d.p2 * (r2 + 2.0 * x * x)
    y_tan = d.p1 * (r2 + 2.0 * y * y) + 2.0 * d.p2 * x * y

    x_d = x * radial + x_tan
    y_d = y * radial + y_tan
    return x_d, y_d


def _normalized_to_pixel_with_distortion(
    x_n: float,
    y_n: float,
    intr: Intrinsics,
    distortion: Optional[Distortion],
) -> Tuple[float, float]:
    if distortion is not None and not distortion.is_zero():
        x_n, y_n = _apply_distortion(np.array(x_n), np.array(y_n), distortion)
        x_n = float(x_n)
        y_n = float(y_n)

    u = intr.fx * x_n + intr.cx
    v = intr.fy * y_n + intr.cy
    return float(u), float(v)


def _los_to_pixel_with_distortion(
    u_cam: np.ndarray,
    intr: Intrinsics,
    *,
    distortion: Optional[Distortion],
    behind: BehindPolicy,
) -> Tuple[float, float, bool]:
    """
    Returns (u_px, v_px, valid). Uses distortion if provided.
    """
    if u_cam.shape != (3,):
        raise ValueError("u_cam must be shape (3,)")
    if not np.all(np.isfinite(u_cam)):
        return (np.nan, np.nan, False)

    if u_cam[2] <= 0:
        if behind == "drop":
            return (np.nan, np.nan, False)
        return (np.nan, np.nan, False)

    # Convert to normalized
    x_n = float(u_cam[0] / u_cam[2])
    y_n = float(u_cam[1] / u_cam[2])

    u_px, v_px = _normalized_to_pixel_with_distortion(x_n, y_n, intr, distortion)
    return (u_px, v_px, True)


def _rng_or_default(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _noise_uv(
    rng: np.random.Generator,
    sigma_px: float,
    *,
    mode: NoiseMode,
) -> Tuple[float, float]:
    if mode == "none" or sigma_px == 0.0:
        return 0.0, 0.0
    if mode != "gaussian":
        raise ValueError(f"Unknown noise mode: {mode}")
    du, dv = rng.normal(loc=0.0, scale=sigma_px, size=2)
    return float(du), float(dv)


def _handle_bounds(
    intr: Intrinsics,
    u: float,
    v: float,
    *,
    policy: OutOfFramePolicy,
) -> Tuple[float, float, bool]:
    if intr.width is None or intr.height is None:
        # If not provided, we can't enforce bounds; treat as valid.
        return u, v, True

    inb = (0.0 <= u < float(intr.width)) and (0.0 <= v < float(intr.height))
    if inb:
        return u, v, True

    if policy == "drop":
        return np.nan, np.nan, False
    if policy == "keep_invalid":
        return u, v, False
    if policy == "clamp":
        u2 = float(np.clip(u, 0.0, float(intr.width) - 1.0))
        v2 = float(np.clip(v, 0.0, float(intr.height) - 1.0))
        return u2, v2, True

    raise ValueError(f"Unknown out_of_frame policy: {policy}")


def simulate_pixel_measurement(
    r_sc: ArrayLike,
    r_body: ArrayLike,
    intrinsics: Intrinsics,
    R_cam_from_frame: Optional[ArrayLike] = None,
    sigma_px: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    *,
    t: float = 0.0,
    dropout_p: float = 0.0,
    noise_mode: NoiseMode = "gaussian",
    out_of_frame: OutOfFramePolicy = "drop",
    behind: BehindPolicy = "drop",
    distortion: Optional[Distortion] = None,
    centroid_bias_px: float = 0.0,
) -> PixelMeasurement:
    """
    Simulate a point-detection (centroid) pixel measurement.

    Steps:
      1) rho = r_body - r_sc
      2) u_global = rho / ||rho||
      3) u_cam = R_cam_from_frame @ u_global
      4) project to pixel (with optional distortion)
      5) add noise
      6) optional bounds handling
      7) optional centroid bias (simple constant-magnitude push away from image center)

    centroid_bias_px:
      - if >0, shifts centroid radially outward from principal point by this many pixels
        (a crude "toward limb" bias).
    """
    rng = _rng_or_default(rng)

    if dropout_p > 0.0 and rng.random() < dropout_p:
        return PixelMeasurement(t=t, u_px=np.nan, v_px=np.nan, sigma_px=float(sigma_px), valid=False, meta={"dropped": True})

    r_sc = _as3(r_sc, "r_sc")
    r_body = _as3(r_body, "r_body")
    R = _asR(R_cam_from_frame)

    rho = r_body - r_sc
    u_global = _unit(rho)
    if not np.all(np.isfinite(u_global)):
        return PixelMeasurement(t=t, u_px=np.nan, v_px=np.nan, sigma_px=float(sigma_px), valid=False, meta={"reason": "zero_range"})

    u_cam = rotate_vector(R, u_global)

    u_ideal, v_ideal, ok = _los_to_pixel_with_distortion(u_cam, intrinsics, distortion=distortion, behind=behind)
    if not ok:
        return PixelMeasurement(t=t, u_px=np.nan, v_px=np.nan, sigma_px=float(sigma_px), valid=False, meta={"reason": "behind_camera"})

    if centroid_bias_px != 0.0 and np.isfinite(u_ideal) and np.isfinite(v_ideal):
        dx = u_ideal - float(intrinsics.cx)
        dy = v_ideal - float(intrinsics.cy)
        n = float(np.hypot(dx, dy))
        if n > 1e-9:
            u_ideal += float(centroid_bias_px) * (dx / n)
            v_ideal += float(centroid_bias_px) * (dy / n)

    du, dv = _noise_uv(rng, float(sigma_px), mode=noise_mode)
    u_meas = u_ideal + du
    v_meas = v_ideal + dv

    u_meas, v_meas, valid = _handle_bounds(intrinsics, u_meas, v_meas, policy=out_of_frame)

    return PixelMeasurement(
        t=t,
        u_px=float(u_meas),
        v_px=float(v_meas),
        sigma_px=float(sigma_px),
        valid=bool(valid),
        meta={
            "u_ideal": float(u_ideal),
            "v_ideal": float(v_ideal),
            "dropout_p": float(dropout_p),
            "centroid_bias_px": float(centroid_bias_px),
        },
    )


def simulate_bbox_measurement(
    r_sc: ArrayLike,
    r_body: ArrayLike,
    body_radius: float,
    intrinsics: Intrinsics,
    R_cam_from_frame: Optional[ArrayLike] = None,
    sigma_px: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    *,
    t: float = 0.0,
    dropout_p: float = 0.0,
    noise_mode: NoiseMode = "gaussian",
    out_of_frame: OutOfFramePolicy = "drop",
    behind: BehindPolicy = "drop",
    distortion: Optional[Distortion] = None,
    centroid_bias_fraction: float = 0.0,
) -> PixelMeasurement:
    """
    Simulate a bounding box detection for a spherical body silhouette.

    Method:
      - Compute line-of-sight unit vector to center u_cam and range r = ||rho||
      - Apparent half-angle alpha = asin(R_body / r) (clamped)
      - Build an orthonormal basis (e1,e2) perpendicular to u_cam
      - Rim directions:
          d1 = u_cam*cos(alpha) + e1*sin(alpha)
          d2 = u_cam*cos(alpha) - e1*sin(alpha)
          d3 = u_cam*cos(alpha) + e2*sin(alpha)
          d4 = u_cam*cos(alpha) - e2*sin(alpha)
      - Project center + rim directions to pixels, take min/max => bbox
      - Add pixel noise to bbox edges (simple)
      - Optional centroid bias "toward limb": moves centroid outward along radial direction
        by centroid_bias_fraction * bbox_half_diag.

    Returns PixelMeasurement with bbox_xyxy set and u_px/v_px as (noisy) bbox center.
    """
    rng = _rng_or_default(rng)

    if body_radius <= 0 or not np.isfinite(body_radius):
        raise ValueError(f"body_radius must be finite and > 0, got {body_radius}")

    if dropout_p > 0.0 and rng.random() < dropout_p:
        return PixelMeasurement(t=t, u_px=np.nan, v_px=np.nan, sigma_px=float(sigma_px), valid=False, bbox_xyxy=None, meta={"dropped": True})

    r_sc = _as3(r_sc, "r_sc")
    r_body = _as3(r_body, "r_body")
    R = _asR(R_cam_from_frame)

    rho = r_body - r_sc
    r = float(np.linalg.norm(rho))
    if not np.isfinite(r) or r <= 1e-12:
        return PixelMeasurement(t=t, u_px=np.nan, v_px=np.nan, sigma_px=float(sigma_px), valid=False, bbox_xyxy=None, meta={"reason": "zero_range"})

    u_global = rho / r
    u_cam = rotate_vector(R, u_global)

    # Behind camera?
    if u_cam[2] <= 0:
        return PixelMeasurement(t=t, u_px=np.nan, v_px=np.nan, sigma_px=float(sigma_px), valid=False, bbox_xyxy=None, meta={"reason": "behind_camera"})

    # Apparent angular radius (clamp to avoid asin domain errors)
    s = float(np.clip(body_radius / r, 0.0, 1.0))
    alpha = float(np.arcsin(s))

    # Build perpendicular basis to u_cam
    u_cam_u = _unit(u_cam)
    if not np.all(np.isfinite(u_cam_u)):
        return PixelMeasurement(t=t, u_px=np.nan, v_px=np.nan, sigma_px=float(sigma_px), valid=False, bbox_xyxy=None, meta={"reason": "bad_unit"})

    # Choose a helper vector not colinear with u_cam_u
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(helper, u_cam_u))) > 0.95:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    e1 = np.cross(u_cam_u, helper)
    e1 = _unit(e1)
    e2 = _unit(np.cross(u_cam_u, e1))

    ca = float(np.cos(alpha))
    sa = float(np.sin(alpha))

    rim_dirs = [
        u_cam_u * ca + e1 * sa,
        u_cam_u * ca - e1 * sa,
        u_cam_u * ca + e2 * sa,
        u_cam_u * ca - e2 * sa,
    ]

    # Project center + rim
    u_c, v_c, okc = _los_to_pixel_with_distortion(u_cam_u, intrinsics, distortion=distortion, behind=behind)
    if not okc:
        return PixelMeasurement(t=t, u_px=np.nan, v_px=np.nan, sigma_px=float(sigma_px), valid=False, bbox_xyxy=None, meta={"reason": "center_projection_failed"})

    uv_rim = []
    for d in rim_dirs:
        u_i, v_i, ok = _los_to_pixel_with_distortion(d, intrinsics, distortion=distortion, behind=behind)
        if ok and np.isfinite(u_i) and np.isfinite(v_i):
            uv_rim.append((u_i, v_i))

    if len(uv_rim) < 2:
        return PixelMeasurement(t=t, u_px=np.nan, v_px=np.nan, sigma_px=float(sigma_px), valid=False, bbox_xyxy=None, meta={"reason": "rim_projection_failed"})

    us = np.array([p[0] for p in uv_rim] + [u_c], dtype=np.float64)
    vs = np.array([p[1] for p in uv_rim] + [v_c], dtype=np.float64)

    xmin = float(np.min(us))
    xmax = float(np.max(us))
    ymin = float(np.min(vs))
    ymax = float(np.max(vs))

    # Add noise to edges (simple independent)
    if noise_mode != "none" and float(sigma_px) != 0.0:
        dxmin, dymin = _noise_uv(rng, float(sigma_px), mode=noise_mode)
        dxmax, dymax = _noise_uv(rng, float(sigma_px), mode=noise_mode)
        xmin += dxmin
        ymin += dymin
        xmax += dxmax
        ymax += dymax

    u_meas = 0.5 * (xmin + xmax)
    v_meas = 0.5 * (ymin + ymax)

    if centroid_bias_fraction != 0.0 and np.isfinite(u_meas) and np.isfinite(v_meas):
        dx = u_meas - float(intrinsics.cx)
        dy = v_meas - float(intrinsics.cy)
        n = float(np.hypot(dx, dy))
        half_diag = 0.5 * float(np.hypot(xmax - xmin, ymax - ymin))
        if n > 1e-9 and half_diag > 0.0:
            shift = float(centroid_bias_fraction) * half_diag
            u_meas += shift * (dx / n)
            v_meas += shift * (dy / n)

    u_meas2, v_meas2, valid = _handle_bounds(intrinsics, u_meas, v_meas, policy=out_of_frame)

    if not valid and out_of_frame == "drop":
        return PixelMeasurement(t=t, u_px=np.nan, v_px=np.nan, sigma_px=float(sigma_px), valid=False, bbox_xyxy=None, meta={"reason": "out_of_frame"})

    if out_of_frame == "clamp" and intrinsics.width is not None and intrinsics.height is not None:
        xmin = float(np.clip(xmin, 0.0, float(intrinsics.width) - 1.0))
        xmax = float(np.clip(xmax, 0.0, float(intrinsics.width) - 1.0))
        ymin = float(np.clip(ymin, 0.0, float(intrinsics.height) - 1.0))
        ymax = float(np.clip(ymax, 0.0, float(intrinsics.height) - 1.0))

    return PixelMeasurement(
        t=t,
        u_px=float(u_meas2),
        v_px=float(v_meas2),
        sigma_px=float(sigma_px),
        valid=bool(valid),
        bbox_xyxy=(float(xmin), float(ymin), float(xmax), float(ymax)),
        meta={
            "u_center_ideal": float(u_c),
            "v_center_ideal": float(v_c),
            "range": r,
            "body_radius": float(body_radius),
            "alpha_rad": float(alpha),
            "dropout_p": float(dropout_p),
            "centroid_bias_fraction": float(centroid_bias_fraction),
        },
    )


__all__ = [
    "Distortion",
    "PixelMeasurement",
    "simulate_pixel_measurement",
    "simulate_bbox_measurement",
]