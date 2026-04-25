"""Realism helpers for the camera-bearing sensor (item 5 of the brief).

Each function answers a single physical question and returns a small typed
result. They live alongside ``camera_bearing.py`` so the sensor can compose
them via opt-in parameters rather than chained wrappers.

What's included:
  - Earth disk occlusion (drops the measurement when Earth blocks the LOS)
  - Sun-target-camera phase angle (drops when phase exceeds threshold;
    needs the scenario to know where the Sun is)
  - Heavy-tailed centroid noise (mixture of Gaussians, with a small fraction
    of measurements drawn from an inflated-sigma distribution)

What's deferred:
  - Motion blur (needs angular-rate computation across measurement steps)
  - True domain-randomized intrinsics (handled at config level, not sensor)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class OcclusionResult:
    occluded: bool
    angular_separation_rad: float
    blocker_angular_radius_rad: float


def disk_occludes_los(
    r_camera: Array,
    r_target: Array,
    r_blocker: Array,
    blocker_radius_length_units: float,
) -> OcclusionResult:
    """Does ``r_blocker`` (with the given physical radius) cover the LOS to
    ``r_target`` as seen from ``r_camera``? Lengths must share units.

    Geometry: angle at the camera between the LOS to the target and the
    LOS to the blocker centre, vs the blocker's angular radius.
    """
    rho_t = np.asarray(r_target, dtype=float).reshape(3) - np.asarray(r_camera, dtype=float).reshape(3)
    rho_b = np.asarray(r_blocker, dtype=float).reshape(3) - np.asarray(r_camera, dtype=float).reshape(3)
    n_t = float(np.linalg.norm(rho_t))
    n_b = float(np.linalg.norm(rho_b))
    if n_t <= 0.0 or n_b <= 0.0:
        return OcclusionResult(False, float("nan"), float("nan"))
    cos_sep = float(np.clip(np.dot(rho_t, rho_b) / (n_t * n_b), -1.0, 1.0))
    sep = float(np.arccos(cos_sep))

    # Blocker only counts if it is between the camera and the target.
    if n_b >= n_t:
        return OcclusionResult(False, sep, 0.0)

    s = float(np.clip(blocker_radius_length_units / n_b, 0.0, 1.0))
    blocker_ang_radius = float(np.arcsin(s))
    return OcclusionResult(sep < blocker_ang_radius, sep, blocker_ang_radius)


def phase_angle_rad(
    r_camera: Array,
    r_target: Array,
    r_sun: Array,
) -> float:
    """Sun--target--camera angle. 0 = full-disc, π = back-lit (new moon)."""
    rt = np.asarray(r_target, dtype=float).reshape(3)
    to_camera = np.asarray(r_camera, dtype=float).reshape(3) - rt
    to_sun = np.asarray(r_sun, dtype=float).reshape(3) - rt
    nc = float(np.linalg.norm(to_camera))
    ns = float(np.linalg.norm(to_sun))
    if nc <= 0.0 or ns <= 0.0:
        return float("nan")
    c = float(np.clip(np.dot(to_camera, to_sun) / (nc * ns), -1.0, 1.0))
    return float(np.arccos(c))


def heavy_tailed_sigma_px(
    base_sigma_px: float,
    *,
    outlier_p: float,
    outlier_scale: float,
    rng: np.random.Generator,
) -> float:
    """Return either ``base_sigma_px`` or ``outlier_scale * base_sigma_px``.

    Picking which sigma to use *before* drawing the noise (rather than
    sampling from a Student-t) lets us keep the existing Gaussian noise path
    in ``simulate_pixel_measurement`` and still get heavy-tail behaviour.
    """
    if outlier_p <= 0.0 or outlier_scale <= 1.0:
        return float(base_sigma_px)
    if rng.random() < float(outlier_p):
        return float(base_sigma_px) * float(outlier_scale)
    return float(base_sigma_px)
