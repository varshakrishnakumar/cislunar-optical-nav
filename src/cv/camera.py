"""
cv/camera.py — camera geometry + conversions (pure geometry utilities)

Goal:
- pixel <-> normalized image coordinates
- normalized <-> LOS unit vector in camera frame
- 3D point in camera frame -> pixel projection

Notes:
- No CR3BP / EKF knowledge; pure pinhole intrinsics (no distortion).
- NumPy-array-friendly (supports scalars, vectors, or Nx arrays via broadcasting).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Literal, overload

import numpy as np

ArrayLike = Union[float, int, np.ndarray]
BehindPolicy = Literal["raise", "nan"]


@dataclass(frozen=True, slots=True)
class Intrinsics:
    """Pinhole camera intrinsics (no distortion)."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: Optional[int] = None
    height: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("fx", "fy"):
            v = float(getattr(self, name))
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and > 0, got {v}")
        for name in ("cx", "cy"):
            v = float(getattr(self, name))
            if not np.isfinite(v):
                raise ValueError(f"{name} must be finite, got {v}")

        if self.width is not None and self.width <= 0:
            raise ValueError(f"width must be > 0, got {self.width}")
        if self.height is not None and self.height <= 0:
            raise ValueError(f"height must be > 0, got {self.height}")

    def as_matrix(self, dtype=np.float64) -> np.ndarray:
        """Return 3x3 intrinsics matrix."""
        K = np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=dtype,
        )
        return K

    def in_bounds(self, u_px: ArrayLike, v_px: ArrayLike, *, margin: float = 0.0) -> np.ndarray:
        """
        Check if pixel coordinate(s) are within [0,width) x [0,height) with optional margin.
        Returns a boolean array broadcast to u/v shape.

        Requires width/height to be provided.
        """
        if self.width is None or self.height is None:
            raise ValueError("in_bounds requires Intrinsics.width and Intrinsics.height to be set.")

        u = np.asarray(u_px, dtype=np.float64)
        v = np.asarray(v_px, dtype=np.float64)

        return (
            (u >= -margin) & (u < (self.width + margin)) &
            (v >= -margin) & (v < (self.height + margin))
        )


def _as_intrinsics(K: Union[Intrinsics, np.ndarray]) -> Intrinsics:
    """Accept either an Intrinsics or a 3x3 matrix-like and return Intrinsics."""
    if isinstance(K, Intrinsics):
        return K

    Km = np.asarray(K, dtype=np.float64)
    if Km.shape != (3, 3):
        raise ValueError(f"K must be Intrinsics or shape (3,3), got {Km.shape}")

    fx = float(Km[0, 0])
    fy = float(Km[1, 1])
    cx = float(Km[0, 2])
    cy = float(Km[1, 2])
    return Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy)


def _broadcast_pair(a: ArrayLike, b: ArrayLike, *, dtype=np.float64) -> Tuple[np.ndarray, np.ndarray]:
    """Convert to arrays and broadcast to common shape."""
    a_arr = np.asarray(a, dtype=dtype)
    b_arr = np.asarray(b, dtype=dtype)
    return np.broadcast_arrays(a_arr, b_arr)


def _normalize_vectors(v: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Normalize vectors along last axis; returns NaNs if norm is too small."""
    v = np.asarray(v, dtype=np.float64)
    if v.shape[-1] != 3:
        raise ValueError(f"Expected vectors with last dim 3, got shape {v.shape}")

    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = v / n
    out = np.where(n > eps, out, np.nan)
    return out


def pixel_to_normalized(u_px: ArrayLike, v_px: ArrayLike, K: Union[Intrinsics, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert pixel coordinates -> normalized image coordinates.

    x_n = (u - cx) / fx
    y_n = (v - cy) / fy
    """
    intr = _as_intrinsics(K)
    u, v = _broadcast_pair(u_px, v_px)
    x_n = (u - intr.cx) / intr.fx
    y_n = (v - intr.cy) / intr.fy
    return x_n, y_n


def normalized_to_pixel(x_n: ArrayLike, y_n: ArrayLike, K: Union[Intrinsics, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert normalized image coordinates -> pixel coordinates.

    u = fx * x_n + cx
    v = fy * y_n + cy
    """
    intr = _as_intrinsics(K)
    x, y = _broadcast_pair(x_n, y_n)
    u = intr.fx * x + intr.cx
    v = intr.fy * y + intr.cy
    return u, v


def pixel_to_los_cam(u_px: ArrayLike, v_px: ArrayLike, K: Union[Intrinsics, np.ndarray]) -> np.ndarray:
    """
    Pixel -> LOS unit vector in camera frame.

    Steps:
      - compute normalized (x_n, y_n)
      - form ray r = [x_n, y_n, 1]
      - return normalized r / ||r||
    Output shape: (..., 3)
    """
    x_n, y_n = pixel_to_normalized(u_px, v_px, K)
    x_n, y_n = _broadcast_pair(x_n, y_n)
    r = np.stack([x_n, y_n, np.ones_like(x_n)], axis=-1)
    return _normalize_vectors(r)


def los_cam_to_pixel(
    u_cam: ArrayLike,
    K: Union[Intrinsics, np.ndarray],
    *,
    behind: BehindPolicy = "nan",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LOS vector in camera frame -> pixel.

    If u_cam[...,2] <= 0 (behind camera):
      - behind="raise": raise ValueError
      - behind="nan": return NaNs for those entries

    For z>0:
      x_n = X/Z, y_n = Y/Z
      then normalized_to_pixel.
    """
    intr = _as_intrinsics(K)
    u = np.asarray(u_cam, dtype=np.float64)
    if u.shape[-1] != 3:
        raise ValueError(f"u_cam must have last dimension 3, got shape {u.shape}")

    z = u[..., 2]
    bad = z <= 0

    if np.any(bad) and behind == "raise":
        raise ValueError("los_cam_to_pixel: some rays have z <= 0 (behind camera).")

    with np.errstate(invalid="ignore", divide="ignore"):
        x_n = u[..., 0] / z
        y_n = u[..., 1] / z

    u_px, v_px = normalized_to_pixel(x_n, y_n, intr)

    if np.any(bad):
        u_px = np.where(bad, np.nan, u_px)
        v_px = np.where(bad, np.nan, v_px)

    return u_px, v_px


def project_point_cam_to_pixel(
    p_cam: ArrayLike,
    K: Union[Intrinsics, np.ndarray],
    *,
    behind: BehindPolicy = "nan",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D point(s) in camera frame -> pixel.

    Same as los_cam_to_pixel but for arbitrary point(s):
      x_n = X/Z, y_n = Y/Z, then intrinsics mapping.

    If Z <= 0:
      - behind="raise": raise ValueError
      - behind="nan": return NaNs for those entries
    """
    intr = _as_intrinsics(K)
    p = np.asarray(p_cam, dtype=np.float64)
    if p.shape[-1] != 3:
        raise ValueError(f"p_cam must have last dimension 3, got shape {p.shape}")

    z = p[..., 2]
    bad = z <= 0

    if np.any(bad) and behind == "raise":
        raise ValueError("project_point_cam_to_pixel: some points have Z <= 0 (behind camera).")

    with np.errstate(invalid="ignore", divide="ignore"):
        x_n = p[..., 0] / z
        y_n = p[..., 1] / z

    u_px, v_px = normalized_to_pixel(x_n, y_n, intr)

    if np.any(bad):
        u_px = np.where(bad, np.nan, u_px)
        v_px = np.where(bad, np.nan, v_px)

    return u_px, v_px


def rotate_vector(R: ArrayLike, v: ArrayLike) -> np.ndarray:
    """
    Optional rotation helper: rotate vector(s) by R.

    - R: (3,3)
    - v: (...,3)
    Returns: (...,3)

    Equivalent to: (R @ v.T).T for batched v.
    """
    Rm = np.asarray(R, dtype=np.float64)
    if Rm.shape != (3, 3):
        raise ValueError(f"R must be shape (3,3), got {Rm.shape}")

    vec = np.asarray(v, dtype=np.float64)
    if vec.shape[-1] != 3:
        raise ValueError(f"v must have last dimension 3, got shape {vec.shape}")

    # einsum handles batching cleanly
    return np.einsum("ij,...j->...i", Rm, vec)


__all__ = [
    "Intrinsics",
    "pixel_to_normalized",
    "normalized_to_pixel",
    "pixel_to_los_cam",
    "los_cam_to_pixel",
    "project_point_cam_to_pixel",
    "rotate_vector",
]