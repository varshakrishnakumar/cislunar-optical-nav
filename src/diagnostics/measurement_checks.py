from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

from diagnostics.config import CameraMode
from diagnostics.types import HypothesisResult

from cv.camera import (
    Intrinsics,
    los_cam_to_pixel,
    pixel_to_los_cam,
    rotate_vector,
)
from cv.sim_measurements import simulate_pixel_measurement
from cv.pointing import camera_dcm_from_boresight, desired_los_from_estimate


Array = np.ndarray
VisibilityClass = Literal["visible", "behind_camera", "out_of_frame", "invalid"]


@dataclass(frozen=True)
class MeasurementCheckResult:
    name: str
    passed: bool
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MeasurementCheckSuiteResult:
    checks: list[MeasurementCheckResult]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_hypotheses(self) -> list[HypothesisResult]:
        out: list[HypothesisResult] = []
        for c in self.checks:
            out.append(
                HypothesisResult(
                    name=c.name,
                    passed=c.passed,
                    severity="info" if c.passed else "failure",
                    summary=c.summary,
                    details=c.details,
                )
            )
        return out


def _unit(v: Array, eps: float = 1e-12) -> Array:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= eps:
        raise ValueError(f"Cannot normalize vector with norm <= {eps}: {v}")
    return v / n


def _safe_allclose(a: Array, b: Array, *, rtol: float, atol: float) -> bool:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return False
    return bool(np.allclose(a, b, rtol=rtol, atol=atol))


def _angle_between(u: Array, v: Array) -> float:
    uu = _unit(u)
    vv = _unit(v)
    c = float(np.clip(np.dot(uu, vv), -1.0, 1.0))
    return float(np.arccos(c))


def _default_intrinsics() -> Intrinsics:
    return Intrinsics(
        fx=400.0,
        fy=400.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )


def _moon_position(mu: float) -> Array:
    return np.array([1.0 - mu, 0.0, 0.0], dtype=float)


def select_camera_rotation(
    *,
    camera_mode: CameraMode,
    r_sc_true: Array,
    x_hat_for_pointing: Array,
    r_body: Array,
    R_fixed: Array,
    up_hint: Optional[Array] = None,
) -> Array:
    if camera_mode == "fixed":
        return np.asarray(R_fixed, dtype=float)

    if camera_mode == "truth_tracking":
        boresight = _unit(np.asarray(r_body, dtype=float) - np.asarray(r_sc_true, dtype=float))
        return camera_dcm_from_boresight(boresight, up_hint_I=up_hint, camera_forward_axis="+z")

    if camera_mode == "estimate_tracking":
        boresight = desired_los_from_estimate(
            xhat_sc=np.asarray(x_hat_for_pointing, dtype=float),
            target_pos_I=np.asarray(r_body, dtype=float),
        )
        return camera_dcm_from_boresight(boresight, up_hint_I=up_hint, camera_forward_axis="+z")

    raise ValueError(f"Unknown camera_mode: {camera_mode}")


def classify_visibility(
    *,
    r_sc: Array,
    r_body: Array,
    intrinsics: Intrinsics,
    R_cam_from_frame: Array,
) -> VisibilityClass:
    r_sc = np.asarray(r_sc, dtype=float).reshape(3)
    r_body = np.asarray(r_body, dtype=float).reshape(3)
    R_cam_from_frame = np.asarray(R_cam_from_frame, dtype=float).reshape(3, 3)

    rho = r_body - r_sc
    if not np.all(np.isfinite(rho)):
        return "invalid"

    rng = float(np.linalg.norm(rho))
    if not np.isfinite(rng) or rng <= 1e-12:
        return "invalid"

    u_frame = rho / rng
    u_cam = rotate_vector(R_cam_from_frame, u_frame)
    u_cam = np.asarray(u_cam, dtype=float).reshape(3)

    if not np.all(np.isfinite(u_cam)):
        return "invalid"

    if u_cam[2] <= 0.0:
        return "behind_camera"

    u_px, v_px = los_cam_to_pixel(u_cam, intrinsics, behind="nan")
    u_px = float(np.asarray(u_px))
    v_px = float(np.asarray(v_px))

    if not (np.isfinite(u_px) and np.isfinite(v_px)):
        return "invalid"

    if intrinsics.width is None or intrinsics.height is None:
        return "visible"

    in_bounds = bool(intrinsics.in_bounds(u_px, v_px))
    return "visible" if in_bounds else "out_of_frame"


def check_camera_projection_sanity(
    *,
    intrinsics: Intrinsics,
    points_cam: Optional[list[Array]] = None,
    atol: float = 1e-12,
) -> MeasurementCheckResult:
    if points_cam is None:
        points_cam = [
            np.array([0.0, 0.0, 1.0]),
            np.array([0.1, -0.05, 1.0]),
            np.array([-0.2, 0.15, 2.0]),
            np.array([0.4, 0.2, 1.5]),
        ]

    finite_count = 0
    bad_count = 0
    behind_ok = True

    for p in points_cam:
        u = _unit(np.asarray(p, dtype=float).reshape(3))
        u_px, v_px = los_cam_to_pixel(u, intrinsics, behind="nan")
        ok = np.isfinite(u_px) and np.isfinite(v_px)
        finite_count += int(ok)
        bad_count += int(not ok)

    behind_vecs = [
        np.array([0.0, 0.0, -1.0]),
        np.array([0.2, -0.1, -1.0]),
    ]
    for p in behind_vecs:
        u = _unit(np.asarray(p, dtype=float).reshape(3))
        u_px, v_px = los_cam_to_pixel(u, intrinsics, behind="nan")
        if np.isfinite(u_px) or np.isfinite(v_px):
            behind_ok = False

    passed = (finite_count == len(points_cam)) and behind_ok
    return MeasurementCheckResult(
        name="camera_projection_sanity",
        passed=passed,
        summary=(
            "Forward LOS vectors projected to finite pixels and behind-camera vectors were rejected."
            if passed
            else "Camera projection sanity check failed."
        ),
        details={
            "num_forward_cases": len(points_cam),
            "num_finite_forward": finite_count,
            "num_bad_forward": bad_count,
            "behind_nan_behavior_ok": behind_ok,
            "atol": atol,
        },
    )


def check_pixel_los_roundtrip(
    *,
    intrinsics: Intrinsics,
    pixel_cases: Optional[list[tuple[float, float]]] = None,
    atol_px: float = 1e-9,
    rtol: float = 1e-9,
) -> MeasurementCheckResult:
    if pixel_cases is None:
        pixel_cases = [
            (intrinsics.cx, intrinsics.cy),
            (intrinsics.cx + 10.0, intrinsics.cy - 5.0),
            (100.0, 80.0),
            (540.0, 400.0),
        ]

    errs: list[float] = []
    failed_cases: list[dict[str, Any]] = []

    for u0, v0 in pixel_cases:
        los_cam = pixel_to_los_cam(u0, v0, intrinsics)
        u1, v1 = los_cam_to_pixel(los_cam, intrinsics, behind="nan")

        du = float(np.asarray(u1) - u0)
        dv = float(np.asarray(v1) - v0)
        err = float(np.hypot(du, dv))
        errs.append(err)

        if not (np.isfinite(u1) and np.isfinite(v1)) or not _safe_allclose(
            np.array([u1, v1]), np.array([u0, v0]), rtol=rtol, atol=atol_px
        ):
            failed_cases.append(
                {
                    "u0": u0,
                    "v0": v0,
                    "u1": float(np.asarray(u1)),
                    "v1": float(np.asarray(v1)),
                    "pixel_error": err,
                }
            )

    max_err = float(np.max(errs)) if errs else float("nan")
    passed = len(failed_cases) == 0

    return MeasurementCheckResult(
        name="pixel_los_roundtrip",
        passed=passed,
        summary=(
            "Pixel -> LOS -> pixel roundtrip stayed within tolerance."
            if passed
            else "Pixel -> LOS -> pixel roundtrip exceeded tolerance."
        ),
        details={
            "num_cases": len(pixel_cases),
            "max_pixel_error": max_err,
            "atol_px": atol_px,
            "rtol": rtol,
            "failed_cases": failed_cases,
        },
    )


def check_frame_rotation_consistency(
    *,
    intrinsics: Intrinsics,
    R_cam_from_frame: Array,
    pixel_cases: Optional[list[tuple[float, float]]] = None,
    angle_tol_rad: float = 1e-10,
) -> MeasurementCheckResult:
    if pixel_cases is None:
        pixel_cases = [
            (intrinsics.cx, intrinsics.cy),
            (intrinsics.cx + 40.0, intrinsics.cy + 25.0),
            (intrinsics.cx - 60.0, intrinsics.cy - 20.0),
        ]

    R_cf = np.asarray(R_cam_from_frame, dtype=float).reshape(3, 3)
    R_fc = R_cf.T

    angle_errors: list[float] = []
    for u_px, v_px in pixel_cases:
        u_cam = pixel_to_los_cam(u_px, v_px, intrinsics)
        u_frame = rotate_vector(R_fc, u_cam)
        u_cam_back = rotate_vector(R_cf, u_frame)
        angle_errors.append(_angle_between(u_cam, u_cam_back))

    max_angle = float(np.max(angle_errors)) if angle_errors else float("nan")
    passed = bool(np.isfinite(max_angle) and max_angle <= angle_tol_rad)

    return MeasurementCheckResult(
        name="frame_rotation_consistency",
        passed=passed,
        summary=(
            "Camera/frame rotation roundtrip preserved LOS direction."
            if passed
            else "Camera/frame rotation roundtrip changed LOS direction beyond tolerance."
        ),
        details={
            "num_cases": len(pixel_cases),
            "max_angle_error_rad": max_angle,
            "angle_tol_rad": angle_tol_rad,
        },
    )


def check_camera_mode_consistency(
    *,
    mu: float,
    r_sc_true: Array,
    x_hat_for_pointing: Array,
    R_fixed: Optional[Array] = None,
    intrinsics: Optional[Intrinsics] = None,
    boresight_tol_rad: float = 1e-10,
) -> MeasurementCheckResult:
    intrinsics = _default_intrinsics() if intrinsics is None else intrinsics
    R_fixed = np.eye(3, dtype=float) if R_fixed is None else np.asarray(R_fixed, dtype=float).reshape(3, 3)
    r_body = _moon_position(mu)

    results: dict[str, Any] = {}

    for mode in ("fixed", "truth_tracking", "estimate_tracking"):
        R_cf = select_camera_rotation(
            camera_mode=mode,
            r_sc_true=r_sc_true,
            x_hat_for_pointing=x_hat_for_pointing,
            r_body=r_body,
            R_fixed=R_fixed,
        )
        results[mode] = np.asarray(R_cf, dtype=float)

    true_los = _unit(r_body - np.asarray(r_sc_true, dtype=float).reshape(3))
    true_los_cam = rotate_vector(results["truth_tracking"], true_los)
    true_los_cam = np.asarray(true_los_cam, dtype=float).reshape(3)
    truth_tracking_angle = _angle_between(true_los_cam, np.array([0.0, 0.0, 1.0], dtype=float))

    est_los = desired_los_from_estimate(
        xhat_sc=np.asarray(x_hat_for_pointing, dtype=float),
        target_pos_I=r_body,
    )
    est_los_cam = rotate_vector(results["estimate_tracking"], est_los)
    est_los_cam = np.asarray(est_los_cam, dtype=float).reshape(3)
    estimate_tracking_angle = _angle_between(est_los_cam, np.array([0.0, 0.0, 1.0], dtype=float))

    passed = (
        np.isfinite(truth_tracking_angle)
        and np.isfinite(estimate_tracking_angle)
        and truth_tracking_angle <= boresight_tol_rad
        and estimate_tracking_angle <= boresight_tol_rad
    )

    return MeasurementCheckResult(
        name="camera_mode_consistency",
        passed=passed,
        summary=(
            "Tracking camera modes align their intended LOS with the camera boresight."
            if passed
            else "At least one tracking camera mode failed to align with the expected boresight."
        ),
        details={
            "truth_tracking_boresight_error_rad": truth_tracking_angle,
            "estimate_tracking_boresight_error_rad": estimate_tracking_angle,
            "boresight_tol_rad": boresight_tol_rad,
            "fixed_R_cam_from_frame": results["fixed"],
            "truth_tracking_R_cam_from_frame": results["truth_tracking"],
            "estimate_tracking_R_cam_from_frame": results["estimate_tracking"],
        },
    )


def check_visibility_classification(
    *,
    mu: float,
    r_sc: Array,
    x_hat_for_pointing: Array,
    camera_mode: CameraMode,
    intrinsics: Optional[Intrinsics] = None,
    R_fixed: Optional[Array] = None,
) -> MeasurementCheckResult:
    intrinsics = _default_intrinsics() if intrinsics is None else intrinsics
    R_fixed = np.eye(3, dtype=float) if R_fixed is None else np.asarray(R_fixed, dtype=float).reshape(3, 3)
    r_body = _moon_position(mu)

    R_cf = select_camera_rotation(
        camera_mode=camera_mode,
        r_sc_true=r_sc,
        x_hat_for_pointing=x_hat_for_pointing,
        r_body=r_body,
        R_fixed=R_fixed,
    )

    geom_class = classify_visibility(
        r_sc=r_sc,
        r_body=r_body,
        intrinsics=intrinsics,
        R_cam_from_frame=R_cf,
    )

    pm = simulate_pixel_measurement(
        r_sc=np.asarray(r_sc, dtype=float).reshape(3),
        r_body=r_body,
        intrinsics=intrinsics,
        R_cam_from_frame=R_cf,
        sigma_px=0.0,
        rng=np.random.default_rng(123),
        t=0.0,
        dropout_p=0.0,
        out_of_frame="drop",
        behind="drop",
    )

    sim_valid = bool(pm.valid) and np.isfinite(pm.u_px) and np.isfinite(pm.v_px)

    expected_valid = geom_class == "visible"
    passed = sim_valid == expected_valid

    return MeasurementCheckResult(
        name="visibility_classification",
        passed=passed,
        summary=(
            "Simulator validity matched geometric visibility classification."
            if passed
            else "Simulator validity disagreed with geometric visibility classification."
        ),
        details={
            "camera_mode": camera_mode,
            "geometry_class": geom_class,
            "sim_valid": sim_valid,
            "sim_u_px": float(pm.u_px) if np.isfinite(pm.u_px) else None,
            "sim_v_px": float(pm.v_px) if np.isfinite(pm.v_px) else None,
            "measurement_meta": pm.meta,
        },
    )


def run_measurement_check_suite(
    *,
    mu: float,
    r_sc_true: Array,
    x_hat_for_pointing: Array,
    intrinsics: Optional[Intrinsics] = None,
    R_fixed: Optional[Array] = None,
) -> MeasurementCheckSuiteResult:
    intrinsics = _default_intrinsics() if intrinsics is None else intrinsics
    R_fixed = np.eye(3, dtype=float) if R_fixed is None else np.asarray(R_fixed, dtype=float).reshape(3, 3)

    r_body = _moon_position(mu)
    R_truth = select_camera_rotation(
        camera_mode="truth_tracking",
        r_sc_true=r_sc_true,
        x_hat_for_pointing=x_hat_for_pointing,
        r_body=r_body,
        R_fixed=R_fixed,
    )

    checks = [
        check_camera_projection_sanity(intrinsics=intrinsics),
        check_pixel_los_roundtrip(intrinsics=intrinsics),
        check_frame_rotation_consistency(intrinsics=intrinsics, R_cam_from_frame=R_truth),
        check_camera_mode_consistency(
            mu=mu,
            r_sc_true=r_sc_true,
            x_hat_for_pointing=x_hat_for_pointing,
            R_fixed=R_fixed,
            intrinsics=intrinsics,
        ),
        check_visibility_classification(
            mu=mu,
            r_sc=r_sc_true,
            x_hat_for_pointing=x_hat_for_pointing,
            camera_mode="fixed",
            intrinsics=intrinsics,
            R_fixed=R_fixed,
        ),
        check_visibility_classification(
            mu=mu,
            r_sc=r_sc_true,
            x_hat_for_pointing=x_hat_for_pointing,
            camera_mode="truth_tracking",
            intrinsics=intrinsics,
            R_fixed=R_fixed,
        ),
        check_visibility_classification(
            mu=mu,
            r_sc=r_sc_true,
            x_hat_for_pointing=x_hat_for_pointing,
            camera_mode="estimate_tracking",
            intrinsics=intrinsics,
            R_fixed=R_fixed,
        ),
    ]
    return MeasurementCheckSuiteResult(checks=checks)
