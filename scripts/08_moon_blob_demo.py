from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from _common import ensure_src_on_path, repo_path

ensure_src_on_path()

from vision.blob_detection import BlobDetectionResult, detect_primary_blob
from vision.plotting import (
    annotate_crop,
    save_video_from_frames,
    save_crop_strip,
)
from visualization.style import (
    AMBER,
    BG,
    BORDER,
    CYAN,
    GREEN,
    PANEL,
    RED,
    TEXT,
    apply_dark_theme,
    plt,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a Moon-blob centroid detector on a fixed image ROI and save "
            "annotated frames, a strip, and presentation-ready metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--glob", type=str, default="*.png")
    parser.add_argument("--output-dir", type=Path, default=Path("results/vision/08_moon_blob_demo"))
    parser.add_argument("--roi", nargs=4, type=int, required=True, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--strip-count", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-area", type=int, default=10)
    parser.add_argument("--blur-ksize", type=int, default=5)
    parser.add_argument("--morph-close", type=int, default=3)
    parser.add_argument("--morph-open", type=int, default=0)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Optional metrics CSV path. Defaults to output-dir/moon_blob_metrics.csv.",
    )
    parser.add_argument(
        "--summary-plot",
        type=Path,
        default=None,
        help="Optional summary plot path. Defaults to output-dir/moon_blob_metrics_summary.png.",
    )
    return parser.parse_args()


def load_paths(input_dir: Path, pattern: str, max_frames: int | None):
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    paths = sorted(p for p in input_dir.glob(pattern) if p.suffix.lower() in IMAGE_EXTS)
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths:
        raise FileNotFoundError("No frames found.")
    return paths


def validate_roi(roi: tuple[int, int, int, int], image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = roi
    h, w = image_shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(
            f"ROI {roi} is outside image bounds width={w}, height={h}. "
            "Use X0 Y0 X1 Y1 pixel coordinates."
        )
    return roi


def crop_roi(image_bgr: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = roi
    return image_bgr[y0:y1, x0:x1].copy()


def detection_metrics_row(
    *,
    frame_index: int,
    image_path: Path,
    roi: tuple[int, int, int, int],
    detection: BlobDetectionResult,
) -> dict[str, float | int | str]:
    x0, y0, x1, y1 = roi
    row: dict[str, float | int | str] = {
        "frame_index": int(frame_index),
        "frame_name": image_path.name,
        "detected": int(detection.found),
        "roi_x0_px": int(x0),
        "roi_y0_px": int(y0),
        "roi_x1_px": int(x1),
        "roi_y1_px": int(y1),
        "threshold_value": float(detection.threshold_value)
        if detection.threshold_value is not None
        else float("nan"),
        "crop_centroid_x_px": float("nan"),
        "crop_centroid_y_px": float("nan"),
        "full_centroid_x_px": float("nan"),
        "full_centroid_y_px": float("nan"),
        "area_px2": float(detection.area_px),
        "enclosing_radius_px": float("nan"),
        "bbox_width_px": float("nan"),
        "bbox_height_px": float("nan"),
        "bbox_aspect": float("nan"),
        "circularity_area_over_circle": float("nan"),
    }
    if not detection.found:
        return row
    if detection.centroid_xy is not None:
        cx, cy = detection.centroid_xy
        row["crop_centroid_x_px"] = float(cx)
        row["crop_centroid_y_px"] = float(cy)
        row["full_centroid_x_px"] = float(cx + x0)
        row["full_centroid_y_px"] = float(cy + y0)
    if detection.enclosing_circle_xy_r is not None:
        _, _, radius_px = detection.enclosing_circle_xy_r
        row["enclosing_radius_px"] = float(radius_px)
        if radius_px > 0.0:
            row["circularity_area_over_circle"] = float(
                detection.area_px / (np.pi * radius_px * radius_px)
            )
    if detection.bbox_xywh is not None:
        _, _, bw, bh = detection.bbox_xywh
        row["bbox_width_px"] = float(bw)
        row["bbox_height_px"] = float(bh)
        row["bbox_aspect"] = float(bw / bh) if bh else float("nan")
    return row


def write_metrics_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No metrics rows to write.")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _nanmedian_or_nan(vals: np.ndarray) -> float:
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float("nan")


def save_metrics_summary_plot(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    apply_dark_theme()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.array([float(row["frame_index"]) for row in rows], dtype=float)
    detected = np.array([float(row["detected"]) for row in rows], dtype=float)
    area = np.array([float(row["area_px2"]) for row in rows], dtype=float)
    radius = np.array([float(row["enclosing_radius_px"]) for row in rows], dtype=float)
    full_x = np.array([float(row["full_centroid_x_px"]) for row in rows], dtype=float)
    full_y = np.array([float(row["full_centroid_y_px"]) for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.patch.set_facecolor(BG)
    for ax in axes.flat:
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(True, color=BORDER, alpha=0.75)
        ax.tick_params(colors=TEXT)

    axes[0, 0].step(frame, detected, where="mid", color=GREEN)
    axes[0, 0].set_title("Detection Availability", color=TEXT)
    axes[0, 0].set_ylabel("detected [0/1]", color=TEXT)

    axes[0, 1].plot(frame, area, color=CYAN, marker="o", ms=3)
    axes[0, 1].set_title("Detected Moon Area", color=TEXT)
    axes[0, 1].set_ylabel("area [px²]", color=TEXT)

    axes[1, 0].plot(frame, radius, color=AMBER, marker="o", ms=3)
    axes[1, 0].set_title("Angular-Size Proxy", color=TEXT)
    axes[1, 0].set_ylabel("enclosing radius [px]", color=TEXT)
    axes[1, 0].set_xlabel("frame index", color=TEXT)

    axes[1, 1].plot(full_x, full_y, color=RED, marker="o", ms=3)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("Centroid Track in Full Frame", color=TEXT)
    axes[1, 1].set_xlabel("u [px]", color=TEXT)
    axes[1, 1].set_ylabel("v [px]", color=TEXT)

    detected_mask = detected > 0
    summary = (
        f"detection rate = {np.mean(detected):.0%}\n"
        f"median radius = {_nanmedian_or_nan(radius[detected_mask]):.1f} px\n"
        f"median area = {_nanmedian_or_nan(area[detected_mask]):.0f} px²"
    )
    fig.suptitle("08 Moon Blob ROI Metrics\n" + summary, color=TEXT, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(output_path, dpi=220, facecolor=BG)
    plt.close(fig)


def main():
    args = parse_args()
    input_dir = repo_path(args.input_dir)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = repo_path(args.metrics_csv) if args.metrics_csv else output_dir / "moon_blob_metrics.csv"
    summary_plot = repo_path(args.summary_plot) if args.summary_plot else output_dir / "moon_blob_metrics_summary.png"

    roi = tuple(args.roi)
    frame_paths = load_paths(input_dir, args.glob, args.max_frames)

    annotated_frames = []
    labels = []
    metrics_rows: list[dict[str, float | int | str]] = []

    for i, path in enumerate(frame_paths):
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue

        roi = validate_roi(roi, image_bgr.shape)
        crop_bgr = crop_roi(image_bgr, roi)

        detection = detect_primary_blob(
            image_bgr=crop_bgr,
            threshold=args.threshold,
            min_area=args.min_area,
            blur_ksize=args.blur_ksize,
            morph_open=args.morph_open,
            morph_close=args.morph_close,
            invert=args.invert,
        )

        annotated = annotate_crop(crop_bgr, detection)
        annotated_frames.append(annotated)
        labels.append(path.stem)
        metrics_rows.append(
            detection_metrics_row(
                frame_index=i,
                image_path=path,
                roi=roi,
                detection=detection,
            )
        )

    if not annotated_frames:
        raise RuntimeError("No readable frames were processed.")

    video_path = output_dir / "moon_blob_demo.mp4"
    strip_path = output_dir / "moon_blob_strip.png"

    if not args.skip_video:
        save_video_from_frames(
            frames_bgr=annotated_frames,
            output_path=video_path,
            fps=args.fps,
        )

    save_crop_strip(
        frames_bgr=annotated_frames,
        labels=labels,
        output_path=strip_path,
        sample_count=args.strip_count,
        title="Detected Moon region across trajectory",
    )
    write_metrics_csv(metrics_csv, metrics_rows)
    save_metrics_summary_plot(metrics_rows, summary_plot)

    if not args.skip_video:
        print(f"Saved video: {video_path}")
    else:
        print("Skipped video by request.")
    print(f"Saved strip: {strip_path}")
    print(f"Saved metrics CSV: {metrics_csv}")
    print(f"Saved summary plot: {summary_plot}")


if __name__ == "__main__":
    main()
