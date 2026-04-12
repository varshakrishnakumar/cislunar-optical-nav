
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from vision.blob_detection import BlobDetectionResult
from visualization.style import plt


Array = np.ndarray


def annotate_crop(image_bgr: np.ndarray, detection: BlobDetectionResult) -> np.ndarray:
    canvas = image_bgr.copy()

    if detection.found and detection.contour_xy is not None:
        cv2.drawContours(canvas, [detection.contour_xy], -1, (0, 255, 0), 1)

    if detection.found and detection.centroid_xy is not None:
        cx, cy = detection.centroid_xy
        px, py = int(round(cx)), int(round(cy))
        cv2.drawMarker(
            canvas,
            (px, py),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=1,
        )
        cv2.circle(canvas, (px, py), 2, (0, 0, 255), -1)

    return canvas


def save_video_from_frames(
    frames_bgr: Sequence[np.ndarray],
    output_path: Path,
    fps: float = 8.0,
    *,
    timestamps: Sequence[float] | None = None,
) -> None:
    """Write frames to an mp4 file.

    Parameters
    ----------
    timestamps:
        Optional sequence of simulation times (one per frame).  When provided,
        each frame is annotated with ``frame N/M  t=<value>`` in the top-left
        corner so that detections can be correlated back to the EKF timeline.
    """
    if not frames_bgr:
        raise ValueError("No frames provided.")
    if timestamps is not None and len(timestamps) != len(frames_bgr):
        raise ValueError(
            f"timestamps length ({len(timestamps)}) must match "
            f"frames_bgr length ({len(frames_bgr)})."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    total = len(frames_bgr)
    for i, frame in enumerate(frames_bgr):
        if frame.shape[:2] != (h, w):
            raise ValueError("All frames must have same size.")
        out_frame = frame.copy()
        label_parts = [f"{i + 1}/{total}"]
        if timestamps is not None:
            label_parts.append(f"t={float(timestamps[i]):.3f}")
        cv2.putText(
            out_frame,
            "  ".join(label_parts),
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        writer.write(out_frame)

    writer.release()


def save_crop_strip(
    frames_bgr: Sequence[np.ndarray],
    labels: Sequence[str],
    output_path: Path,
    sample_count: int = 8,
    title: str | None = None,
) -> None:
    if not frames_bgr:
        raise ValueError("No frames provided.")

    n = len(frames_bgr)
    sample_count = min(sample_count, n)
    idx = np.linspace(0, n - 1, sample_count).astype(int)

    fig, axes = plt.subplots(1, sample_count, figsize=(3 * sample_count, 3))
    if sample_count == 1:
        axes = [axes]

    for ax, i in zip(axes, idx):
        rgb = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(labels[i], fontsize=8)
        ax.axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _bgr_to_rgb(image_bgr: Array) -> Array:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _draw_overlay(image_bgr: Array, detection: BlobDetectionResult) -> Array:
    canvas = image_bgr.copy()

    if not detection.found:
        return canvas

    if detection.contour_xy is not None:
        cv2.drawContours(canvas, [detection.contour_xy], -1, (0, 255, 0), 2)

    if detection.bbox_xywh is not None:
        x, y, w, h = detection.bbox_xywh
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 0), 1)

    if detection.enclosing_circle_xy_r is not None:
        cx, cy, r = detection.enclosing_circle_xy_r
        cv2.circle(canvas, (int(round(cx)), int(round(cy))), int(round(r)), (255, 0, 255), 1)

    if detection.centroid_xy is not None:
        cx, cy = detection.centroid_xy
        px = int(round(cx))
        py = int(round(cy))
        cv2.drawMarker(
            canvas,
            (px, py),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=16,
            thickness=2,
        )
        cv2.circle(canvas, (px, py), 3, (0, 0, 255), -1)

    return canvas


def save_annotated_frame(
    image_bgr: Array,
    detection: BlobDetectionResult,
    output_path: Path,
    title: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overlay_bgr = _draw_overlay(image_bgr, detection)
    overlay_rgb = _bgr_to_rgb(overlay_bgr)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay_rgb)
    plt.axis("off")
    if title:
        plt.title(title)

    if detection.found and detection.centroid_xy is not None:
        cx, cy = detection.centroid_xy
        text = f"centroid=({cx:.1f}, {cy:.1f}), area={detection.area_px:.0f}px"
    else:
        text = "no blob detected"

    plt.figtext(0.5, 0.02, text, ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_comparison_panel(
    image_bgr: Array,
    detection: BlobDetectionResult,
    output_path: Path,
    title: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_rgb = _bgr_to_rgb(image_bgr)
    overlay_rgb = _bgr_to_rgb(_draw_overlay(image_bgr, detection))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(raw_rgb)
    axes[0].set_title("Raw frame")
    axes[0].axis("off")

    axes[1].imshow(overlay_rgb)
    axes[1].set_title("Centroid + blob")
    axes[1].axis("off")

    # Pass the 2-D binary mask directly so matplotlib applies the colormap;
    # an already-stacked RGB array would ignore the cmap parameter entirely.
    axes[2].imshow(detection.mask, cmap="hot")
    axes[2].set_title("Detected mask")
    axes[2].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_frame_strip(
    images_bgr: Sequence[Array],
    detections: Sequence[BlobDetectionResult],
    labels: Sequence[str],
    output_path: Path,
    title: str | None = None,
) -> None:
    if not (len(images_bgr) == len(detections) == len(labels)):
        raise ValueError("images_bgr, detections, and labels must have equal length.")

    n = len(images_bgr)
    if n == 0:
        raise ValueError("At least one frame is required.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, image_bgr, detection, label in zip(axes, images_bgr, detections, labels):
        overlay_rgb = _bgr_to_rgb(_draw_overlay(image_bgr, detection))
        ax.imshow(overlay_rgb)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
