
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Literal, overload

import numpy as np

ArrayLike = Union[float, int, np.ndarray]
BehindPolicy = Literal["raise", "nan"]


@dataclass(frozen=True, slots=True)
class Intrinsics:
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
        K = np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=dtype,
        )
        return K

    def in_bounds(self, u_px: ArrayLike, v_px: ArrayLike, *, margin: float = 0.0) -> np.ndarray:
        if self.width is None or self.height is None:
            raise ValueError("in_bounds requires Intrinsics.width and Intrinsics.height to be set.")

        u = np.asarray(u_px, dtype=np.float64)
        v = np.asarray(v_px, dtype=np.float64)

        return (
            (u >= -margin) & (u < (self.width + margin)) &
            (v >= -margin) & (v < (self.height + margin))
        )


def _as_intrinsics(K: Union[Intrinsics, np.ndarray]) -> Intrinsics:
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
    a_arr = np.asarray(a, dtype=dtype)
    b_arr = np.asarray(b, dtype=dtype)
    return np.broadcast_arrays(a_arr, b_arr)


def _normalize_vectors(v: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.shape[-1] != 3:
        raise ValueError(f"Expected vectors with last dim 3, got shape {v.shape}")

    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = v / n
    out = np.where(n > eps, out, np.nan)
    return out


def pixel_to_normalized(u_px: ArrayLike, v_px: ArrayLike, K: Union[Intrinsics, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    intr = _as_intrinsics(K)
    u, v = _broadcast_pair(u_px, v_px)
    x_n = (u - intr.cx) / intr.fx
    y_n = (v - intr.cy) / intr.fy
    return x_n, y_n


def normalized_to_pixel(x_n: ArrayLike, y_n: ArrayLike, K: Union[Intrinsics, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    intr = _as_intrinsics(K)
    x, y = _broadcast_pair(x_n, y_n)
    u = intr.fx * x + intr.cx
    v = intr.fy * y + intr.cy
    return u, v


def pixel_to_los_cam(u_px: ArrayLike, v_px: ArrayLike, K: Union[Intrinsics, np.ndarray]) -> np.ndarray:
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


def is_point_visible_cam(
    p_cam: ArrayLike,
    K: Union[Intrinsics, np.ndarray],
    z_forward_positive: bool = True,
) -> np.ndarray:
    intr = _as_intrinsics(K)
    if intr.width is None or intr.height is None:
        raise ValueError(
            "is_point_visible_cam requires Intrinsics.width and Intrinsics.height to be set."
        )

    p = np.asarray(p_cam, dtype=np.float64)
    if p.shape[-1] != 3:
        raise ValueError(f"p_cam must have last dimension 3, got shape {p.shape}")

    z = p[..., 2]
    front = z > 0 if z_forward_positive else z < 0

    z_eff = z if z_forward_positive else -z
    with np.errstate(invalid="ignore", divide="ignore"):
        x_n = p[..., 0] / z_eff
        y_n = p[..., 1] / z_eff

    u_px, v_px = normalized_to_pixel(x_n, y_n, intr)
    in_bounds = intr.in_bounds(u_px, v_px)
    finite = np.isfinite(u_px) & np.isfinite(v_px)
    return front & finite & in_bounds


def project_point_cam(
    p_cam: ArrayLike,
    K: Union[Intrinsics, np.ndarray],
    nan_if_invalid: bool = True,
) -> np.ndarray:
    behind: BehindPolicy = "nan" if nan_if_invalid else "raise"
    u_px, v_px = project_point_cam_to_pixel(p_cam, K, behind=behind)
    return np.stack([u_px, v_px], axis=-1)


def rotate_vector(R: ArrayLike, v: ArrayLike) -> np.ndarray:
    Rm = np.asarray(R, dtype=np.float64)
    if Rm.shape != (3, 3):
        raise ValueError(f"R must be shape (3,3), got {Rm.shape}")

    vec = np.asarray(v, dtype=np.float64)
    if vec.shape[-1] != 3:
        raise ValueError(f"v must have last dimension 3, got shape {vec.shape}")

    return np.einsum("ij,...j->...i", Rm, vec)


__all__ = [
    "Intrinsics",
    "is_point_visible_cam",
    "project_point_cam",
    "pixel_to_normalized",
    "normalized_to_pixel",
    "pixel_to_los_cam",
    "los_cam_to_pixel",
    "project_point_cam_to_pixel",
    "rotate_vector",
]
