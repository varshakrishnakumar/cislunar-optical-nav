
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


Array = np.ndarray


@dataclass
class BlobDetectionResult:
    found: bool
    centroid_xy: tuple[float, float] | None
    area_px: float
    contour_xy: Array | None
    mask: Array
    bbox_xywh: tuple[int, int, int, int] | None
    enclosing_circle_xy_r: tuple[float, float, float] | None
    threshold_value: float | None


def _validate_kernel_size(k: int) -> int:
    if k <= 0:
        return 0
    if k % 2 == 0:
        k += 1
    return k


def _to_grayscale(image_bgr: Array) -> Array:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected BGR image with shape (H, W, 3).")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def _apply_blur(gray: Array, blur_ksize: int) -> Array:
    blur_ksize = _validate_kernel_size(blur_ksize)
    if blur_ksize <= 1:
        return gray
    return cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0.0)


def _threshold_image(gray: Array, threshold: float | None, invert: bool) -> tuple[float, Array]:
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    if threshold is None:
        used_thresh, binary = cv2.threshold(
            gray, 0, 255, thresh_type | cv2.THRESH_OTSU
        )
    else:
        used_thresh, binary = cv2.threshold(
            gray, float(threshold), 255, thresh_type
        )
    return float(used_thresh), binary


def _apply_morphology(mask: Array, morph_open: int, morph_close: int) -> Array:
    out = mask.copy()

    if morph_open > 0:
        k = np.ones((morph_open, morph_open), dtype=np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)

    if morph_close > 0:
        k = np.ones((morph_close, morph_close), dtype=np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)

    return out


def _largest_valid_contour(mask: Array, min_area: int) -> Optional[Array]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    valid = [c for c in contours if cv2.contourArea(c) >= float(min_area)]
    if not valid:
        return None

    return max(valid, key=cv2.contourArea)


def _contour_centroid(contour: Array) -> tuple[float, float] | None:
    moments = cv2.moments(contour)
    m00 = moments["m00"]
    if abs(m00) < 1e-12:
        return None
    cx = moments["m10"] / m00
    cy = moments["m01"] / m00
    return float(cx), float(cy)


def detect_primary_blob(
    image_bgr: Array,
    threshold: float | None = None,
    min_area: int = 25,
    blur_ksize: int = 5,
    morph_open: int = 0,
    morph_close: int = 3,
    invert: bool = False,
) -> BlobDetectionResult:
    gray = _to_grayscale(image_bgr)
    gray = _apply_blur(gray, blur_ksize=blur_ksize)

    used_thresh, binary = _threshold_image(gray, threshold=threshold, invert=invert)
    mask = _apply_morphology(binary, morph_open=morph_open, morph_close=morph_close)

    contour = _largest_valid_contour(mask, min_area=min_area)
    if contour is None:
        return BlobDetectionResult(
            found=False,
            centroid_xy=None,
            area_px=0.0,
            contour_xy=None,
            mask=mask,
            bbox_xywh=None,
            enclosing_circle_xy_r=None,
            threshold_value=used_thresh,
        )

    area_px = float(cv2.contourArea(contour))
    centroid_xy = _contour_centroid(contour)
    x, y, w, h = cv2.boundingRect(contour)
    (center_xy, circ_r) = cv2.minEnclosingCircle(contour)
    circ_x, circ_y = center_xy

    return BlobDetectionResult(
        found=True,
        centroid_xy=centroid_xy,
        area_px=area_px,
        contour_xy=contour,
        mask=mask,
        bbox_xywh=(int(x), int(y), int(w), int(h)),
        enclosing_circle_xy_r=(float(circ_x), float(circ_y), float(circ_r)),
        threshold_value=used_thresh,
    )
