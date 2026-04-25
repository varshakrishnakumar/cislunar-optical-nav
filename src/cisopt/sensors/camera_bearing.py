"""Pixel-bearing camera sensor wrapping the existing simulate_pixel_measurement.

Output payload is (u_global, sigma_theta, r_body) so the IEKF estimator can
consume it without knowing pixel-space details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from cv.camera import Intrinsics
from cv.pointing import camera_dcm_from_boresight
from cv.sim_measurements import simulate_pixel_measurement
from nav.measurements.pixel_bearing import pixel_detection_to_bearing

from ..protocols import Measurement, Scenario
from .realism import (
    disk_occludes_los,
    heavy_tailed_sigma_px,
    phase_angle_rad,
)


Array = np.ndarray
PointingMode = Literal["fixed", "truth_tracking", "estimate_tracking"]
_VALID_POINTING = ("fixed", "truth_tracking", "estimate_tracking")


@dataclass
class CameraBearingSensor:
    scenario: Scenario
    target_body: str = "Moon"
    fx: float = 400.0
    fy: float = 400.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480
    sigma_px: float = 1.5
    dropout_p: float = 0.0
    pointing: PointingMode = "estimate_tracking"

    # Realism (item 5). Each is opt-in -- defaults are no-ops so existing
    # configs keep their behaviour.
    earth_radius_for_occlusion: float | None = None  # in scenario length units
    phase_angle_max_deg: float | None = None         # drops if phase exceeds
    noise_outlier_p: float = 0.0
    noise_outlier_scale: float = 5.0

    name: str = "camera_bearing"

    def __post_init__(self) -> None:
        if self.pointing not in _VALID_POINTING:
            raise ValueError(
                f"pointing must be one of {_VALID_POINTING}, got {self.pointing!r}"
            )
        self._intr = Intrinsics(
            fx=float(self.fx), fy=float(self.fy),
            cx=float(self.cx), cy=float(self.cy),
            width=int(self.width), height=int(self.height),
        )
        self._R_fixed = np.eye(3, dtype=float)
        self._sun_unavailable = False

    def _R_cam(self, t_s: float, x_truth: Array, x_estimate: Array | None) -> Array:
        r_body = self.scenario.body_position(self.target_body, t_s)
        if self.pointing == "fixed":
            return self._R_fixed
        if self.pointing == "truth_tracking":
            return camera_dcm_from_boresight(r_body - x_truth[:3], camera_forward_axis="+z")
        if x_estimate is None:
            raise ValueError(
                "estimate_tracking pointing requires x_estimate to be provided."
            )
        return camera_dcm_from_boresight(r_body - x_estimate[:3], camera_forward_axis="+z")

    def measure(
        self,
        t_s: float,
        x_truth: Array,
        x_estimate: Array | None,
        *,
        rng: np.random.Generator,
    ) -> Measurement:
        r_body = self.scenario.body_position(self.target_body, float(t_s))
        R_cam = self._R_cam(float(t_s), x_truth, x_estimate)

        # Realism gates run BEFORE the pixel simulator so they don't burn
        # noise samples on rays we'd drop anyway.
        if self.earth_radius_for_occlusion is not None and self.target_body.lower() != "earth":
            try:
                r_earth = self.scenario.body_position("Earth", float(t_s))
            except KeyError:
                r_earth = None
            if r_earth is not None:
                occ = disk_occludes_los(
                    x_truth[:3], r_body, r_earth,
                    float(self.earth_radius_for_occlusion),
                )
                if occ.occluded:
                    return Measurement(
                        t_s=float(t_s),
                        valid=False,
                        payload=None,
                        meta={"reason": "earth_occluded",
                              "angular_sep": occ.angular_separation_rad,
                              "blocker_radius": occ.blocker_angular_radius_rad},
                    )

        if self.phase_angle_max_deg is not None and not self._sun_unavailable:
            try:
                r_sun = self.scenario.body_position("Sun", float(t_s))
            except KeyError:
                self._sun_unavailable = True
                r_sun = None
            if r_sun is not None:
                phi = phase_angle_rad(x_truth[:3], r_body, r_sun)
                if np.degrees(phi) > float(self.phase_angle_max_deg):
                    return Measurement(
                        t_s=float(t_s),
                        valid=False,
                        payload=None,
                        meta={"reason": "phase_angle_exceeded",
                              "phase_deg": float(np.degrees(phi))},
                    )

        sigma_px_eff = heavy_tailed_sigma_px(
            float(self.sigma_px),
            outlier_p=float(self.noise_outlier_p),
            outlier_scale=float(self.noise_outlier_scale),
            rng=rng,
        )

        meas_px = simulate_pixel_measurement(
            r_sc=x_truth[:3],
            r_body=r_body,
            intrinsics=self._intr,
            R_cam_from_frame=R_cam,
            sigma_px=float(sigma_px_eff),
            rng=rng,
            t=float(t_s),
            dropout_p=float(self.dropout_p),
            out_of_frame="drop",
            behind="drop",
        )

        if not meas_px.valid or not np.isfinite(meas_px.u_px):
            return Measurement(
                t_s=float(t_s),
                valid=False,
                payload=None,
                meta={"reason": "invalid_pixel", "pixel": meas_px.meta},
            )

        # Pass the *base* sigma_px to the bearing converter so the filter sees
        # the nominal measurement covariance. Outlier draws inflate the actual
        # error but we keep R consistent with the nominal model -- the goal of
        # heavy-tailed noise is to test filter robustness to outliers, not to
        # broadcast their existence to the filter.
        u_global, sigma_theta = pixel_detection_to_bearing(
            meas_px.u_px, meas_px.v_px, float(self.sigma_px), self._intr, R_cam.T
        )
        if not np.all(np.isfinite(u_global)):
            return Measurement(
                t_s=float(t_s),
                valid=False,
                payload=None,
                meta={"reason": "non_finite_los"},
            )

        payload: dict[str, Any] = {
            "kind": "bearing",
            "u_global": np.asarray(u_global, dtype=float),
            "sigma_theta": float(sigma_theta),
            "r_body": np.asarray(r_body, dtype=float),
        }
        return Measurement(
            t_s=float(t_s),
            valid=True,
            payload=payload,
            meta={
                "u_px": float(meas_px.u_px),
                "v_px": float(meas_px.v_px),
                "sigma_px": float(self.sigma_px),
                "sigma_px_actual": float(sigma_px_eff),
                "pointing": self.pointing,
            },
        )


def build_camera_bearing(params: dict[str, Any], scenario: Scenario) -> CameraBearingSensor:
    return CameraBearingSensor(scenario=scenario, **params)
