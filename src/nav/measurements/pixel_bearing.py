
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Literal

import numpy as np

from cv.camera import Intrinsics, pixel_to_los_cam, rotate_vector


ArrayLike = Union[float, int, np.ndarray]
SigmaApprox = Literal["fx_only", "rms_fx_fy"]


def _as_intrinsics(K: Union[Intrinsics, np.ndarray]) -> Intrinsics:
    if isinstance(K, Intrinsics):
        return K
    Km = np.asarray(K, dtype=np.float64)
    if Km.shape != (3, 3):
        raise ValueError(f"K must be Intrinsics or shape (3,3), got {Km.shape}")
    return Intrinsics(
        fx=float(Km[0, 0]),
        fy=float(Km[1, 1]),
        cx=float(Km[0, 2]),
        cy=float(Km[1, 2]),
    )


def _asR(R: Optional[ArrayLike], name: str) -> np.ndarray:
    if R is None:
        return np.eye(3, dtype=np.float64)
    Rm = np.asarray(R, dtype=np.float64)
    if Rm.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3,3), got {Rm.shape}")
    if not np.all(np.isfinite(Rm)):
        raise ValueError(f"{name} must be finite")
    return Rm


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= eps:
        return np.full((3,), np.nan, dtype=np.float64)
    return v / n


def pixel_to_los_global(
    u_px: ArrayLike,
    v_px: ArrayLike,
    intrinsics: Union[Intrinsics, np.ndarray],
    R_frame_from_cam: Optional[ArrayLike] = None,
) -> np.ndarray:
    K = _as_intrinsics(intrinsics)
    R_fc = _asR(R_frame_from_cam, "R_frame_from_cam")

    u_cam = pixel_to_los_cam(u_px, v_px, K)
    u_global = rotate_vector(R_fc, u_cam)

    u_global = np.asarray(u_global, dtype=np.float64)
    n = np.linalg.norm(u_global, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        u_global = u_global / n
    u_global = np.where(n > 1e-12, u_global, np.nan)
    return u_global


def pixel_noise_to_sigma_theta(
    sigma_px: float,
    intrinsics: Union[Intrinsics, np.ndarray],
    *,
    approx: SigmaApprox = "fx_only",
) -> float:
    if not np.isfinite(sigma_px) or sigma_px < 0:
        raise ValueError(f"sigma_px must be finite and >= 0, got {sigma_px}")

    K = _as_intrinsics(intrinsics)
    fx = float(K.fx)
    fy = float(K.fy)

    if approx == "fx_only":
        return float(sigma_px / fx)

    if approx == "rms_fx_fy":
        a = sigma_px / fx
        b = sigma_px / fy
        return float(np.sqrt(a * a + b * b) / np.sqrt(2.0))

    raise ValueError(f"Unknown approx mode: {approx}")


def pixel_detection_to_bearing(
    u_px: ArrayLike,
    v_px: ArrayLike,
    sigma_px: float,
    intrinsics: Union[Intrinsics, np.ndarray],
    R_frame_from_cam: Optional[ArrayLike] = None,
    *,
    sigma_approx: SigmaApprox = "fx_only",
) -> Tuple[np.ndarray, float]:
    u_global = pixel_to_los_global(u_px, v_px, intrinsics, R_frame_from_cam)
    sigma_theta = pixel_noise_to_sigma_theta(sigma_px, intrinsics, approx=sigma_approx)
    return u_global, sigma_theta


__all__ = [
    "pixel_to_los_global",
    "pixel_noise_to_sigma_theta",
    "pixel_detection_to_bearing",
]
