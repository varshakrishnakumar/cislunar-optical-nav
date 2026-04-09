from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

Array = np.ndarray


def normalize(v: Array, eps: float = 1e-12) -> Array:
    vec = np.asarray(v, dtype=np.float64)
    if vec.shape != (3,):
        raise ValueError(f"v must have shape (3,), got {vec.shape}")

    n = float(np.linalg.norm(vec))
    if not np.isfinite(n) or n <= eps:
        raise ValueError(f"Cannot normalize vector with norm <= {eps}: {vec}")

    return vec / n


def desired_los_from_estimate(
    xhat_sc: Array,
    target_pos_I: Array,
) -> Array:
    xhat = np.asarray(xhat_sc, dtype=np.float64)
    target = np.asarray(target_pos_I, dtype=np.float64)

    if xhat.ndim != 1 or xhat.shape[0] < 3:
        raise ValueError(f"xhat_sc must be a 1D state with at least 3 entries, got shape {xhat.shape}")
    if target.shape != (3,):
        raise ValueError(f"target_pos_I must have shape (3,), got {target.shape}")

    sc_pos_est = xhat[:3]
    rho_I = target - sc_pos_est
    return normalize(rho_I)


def _resolve_up_hint(up_hint_I: Optional[Array], boresight_I: Array, eps: float = 1e-12) -> Array:
    z_cam_I = normalize(boresight_I, eps=eps)

    if up_hint_I is None:
        candidate = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        candidate = np.asarray(up_hint_I, dtype=np.float64)
        if candidate.shape != (3,):
            raise ValueError(f"up_hint_I must have shape (3,), got {candidate.shape}")

    fallbacks = (
        candidate,
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )

    for vec in fallbacks:
        try:
            u = normalize(vec, eps=eps)
        except ValueError:
            continue

        cross_n = np.linalg.norm(np.cross(u, z_cam_I))
        if np.isfinite(cross_n) and cross_n > 10.0 * eps:
            return u

    raise ValueError("Could not find a valid up_hint_I not parallel to boresight_I.")


def _axis_vector_from_boresight(
    boresight_I: Array,
    camera_forward_axis: str,
    eps: float = 1e-12,
) -> Tuple[str, Array]:
    b = normalize(boresight_I, eps=eps)
    axis = camera_forward_axis.strip().lower()

    mapping = {
        "+x": ("x", b),
        "-x": ("x", -b),
        "+y": ("y", b),
        "-y": ("y", -b),
        "+z": ("z", b),
        "-z": ("z", -b),
    }
    if axis not in mapping:
        raise ValueError(
            "camera_forward_axis must be one of '+x', '-x', '+y', '-y', '+z', '-z'; "
            f"got {camera_forward_axis!r}"
        )
    return mapping[axis]


def camera_dcm_from_boresight(
    boresight_I: Array,
    up_hint_I: Array | None = None,
    camera_forward_axis: str = "+z",
) -> Array:
    axis_name, axis_I = _axis_vector_from_boresight(boresight_I, camera_forward_axis)
    up = _resolve_up_hint(up_hint_I, axis_I)

    right_I = normalize(np.cross(up, axis_I))
    true_up_I = normalize(np.cross(axis_I, right_I))

    if axis_name == "z":
        x_cam_I = right_I
        y_cam_I = true_up_I
        z_cam_I = axis_I
    elif axis_name == "x":
        x_cam_I = axis_I
        y_cam_I = true_up_I
        z_cam_I = normalize(np.cross(x_cam_I, y_cam_I))
    elif axis_name == "y":
        x_cam_I = right_I
        y_cam_I = axis_I
        z_cam_I = normalize(np.cross(x_cam_I, y_cam_I))
    else:
        raise AssertionError(f"Unexpected axis_name: {axis_name}")

    R_I_to_C = np.vstack([x_cam_I, y_cam_I, z_cam_I])

    det = float(np.linalg.det(R_I_to_C))
    if not np.isfinite(det) or det <= 0.0:
        raise ValueError("Constructed DCM is not right-handed.")

    return R_I_to_C


def estimate_based_camera_attitude(
    xhat_sc: Array,
    target_pos_I: Array,
    up_hint_I: Array | None = None,
    camera_forward_axis: str = "+z",
) -> tuple[Array, Array]:
    los_I = desired_los_from_estimate(xhat_sc, target_pos_I)
    R_I_to_C = camera_dcm_from_boresight(
        boresight_I=los_I,
        up_hint_I=up_hint_I,
        camera_forward_axis=camera_forward_axis,
    )
    return los_I, R_I_to_C


def off_boresight_angle(
    los_true_I: Array,
    boresight_I: Array,
) -> float:
    los_true = normalize(los_true_I)
    boresight = normalize(boresight_I)
    c = float(np.clip(np.dot(los_true, boresight), -1.0, 1.0))
    return float(np.arccos(c))


__all__ = [
    "normalize",
    "desired_los_from_estimate",
    "camera_dcm_from_boresight",
    "estimate_based_camera_attitude",
    "off_boresight_angle",
]
