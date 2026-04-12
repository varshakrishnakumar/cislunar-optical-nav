
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
    save_annotated_frame,
    save_comparison_panel,
    save_frame_strip,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate blob/centroid demo figures from Moon camera frames."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing camera frames.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.png",
        help="Glob pattern for selecting frames inside input-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/vision/08_blob_centroid_demo"),
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=6,
        help="Maximum number of frames to process.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional manual threshold in [0, 255]. If omitted, Otsu thresholding is used.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=25,
        help="Minimum connected-component area in pixels.",
    )
    parser.add_argument(
        "--blur-ksize",
        type=int,
        default=5,
        help="Gaussian blur kernel size (odd integer, 0 disables blur).",
    )
    parser.add_argument(
        "--morph-open",
        type=int,
        default=0,
        help="Morphological opening kernel size in pixels (0 disables).",
    )
    parser.add_argument(
        "--morph-close",
        type=int,
        default=3,
        help="Morphological closing kernel size in pixels (0 disables).",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert threshold logic if target is dark on bright background.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Optional metrics CSV path. Defaults to output-dir/blob_metrics.csv.",
    )
    parser.add_argument(
        "--summary-plot",
        type=Path,
        default=None,
        help="Optional summary plot path. Defaults to output-dir/blob_metrics_summary.png.",
    )
    return parser.parse_args()


def find_images(input_dir: Path, pattern: str, max_frames: int) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    paths = sorted(p for p in input_dir.glob(pattern) if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(
            f"No images found in {input_dir} matching pattern {pattern!r}"
        )
    return paths[:max_frames]


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detection_metrics_row(
    *,
    frame_index: int,
    image_path: Path,
    detection: BlobDetectionResult,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "frame_index": int(frame_index),
        "frame_name": image_path.name,
        "detected": int(detection.found),
        "threshold_value": float(detection.threshold_value)
        if detection.threshold_value is not None
        else float("nan"),
        "centroid_x_px": float("nan"),
        "centroid_y_px": float("nan"),
        "area_px2": float(detection.area_px),
        "enclosing_radius_px": float("nan"),
        "bbox_x_px": float("nan"),
        "bbox_y_px": float("nan"),
        "bbox_width_px": float("nan"),
        "bbox_height_px": float("nan"),
        "bbox_aspect": float("nan"),
        "circularity_area_over_circle": float("nan"),
    }
    if not detection.found:
        return row

    if detection.centroid_xy is not None:
        row["centroid_x_px"] = float(detection.centroid_xy[0])
        row["centroid_y_px"] = float(detection.centroid_xy[1])
    if detection.enclosing_circle_xy_r is not None:
        _, _, radius_px = detection.enclosing_circle_xy_r
        row["enclosing_radius_px"] = float(radius_px)
        if radius_px > 0.0:
            row["circularity_area_over_circle"] = float(
                detection.area_px / (np.pi * radius_px * radius_px)
            )
    if detection.bbox_xywh is not None:
        x, y, w, h = detection.bbox_xywh
        row["bbox_x_px"] = float(x)
        row["bbox_y_px"] = float(y)
        row["bbox_width_px"] = float(w)
        row["bbox_height_px"] = float(h)
        row["bbox_aspect"] = float(w / h) if h else float("nan")
    return row


def write_metrics_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No metrics rows to write.")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _median_finite(vals: np.ndarray) -> float:
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float("nan")


def save_metrics_summary_plot(
    rows: list[dict[str, float | int | str]],
    output_path: Path,
) -> None:
    apply_dark_theme()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.array([float(row["frame_index"]) for row in rows], dtype=float)
    detected = np.array([float(row["detected"]) for row in rows], dtype=float)
    area = np.array([float(row["area_px2"]) for row in rows], dtype=float)
    radius = np.array([float(row["enclosing_radius_px"]) for row in rows], dtype=float)
    circularity = np.array(
        [float(row["circularity_area_over_circle"]) for row in rows], dtype=float
    )

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
    axes[0, 1].set_title("Detected Blob Area", color=TEXT)
    axes[0, 1].set_ylabel("area [px²]", color=TEXT)

    axes[1, 0].plot(frame, radius, color=AMBER, marker="o", ms=3)
    axes[1, 0].set_title("Enclosing Circle Radius", color=TEXT)
    axes[1, 0].set_ylabel("radius [px]", color=TEXT)
    axes[1, 0].set_xlabel("frame index", color=TEXT)

    axes[1, 1].plot(frame, circularity, color=RED, marker="o", ms=3)
    axes[1, 1].axhline(1.0, color=BORDER, lw=1.0)
    axes[1, 1].set_title("Shape Compactness", color=TEXT)
    axes[1, 1].set_ylabel("area / enclosing-circle area", color=TEXT)
    axes[1, 1].set_xlabel("frame index", color=TEXT)

    finite_area = area[np.isfinite(area) & (detected > 0)]
    finite_circ = circularity[np.isfinite(circularity) & (detected > 0)]
    summary = (
        f"detection rate = {np.mean(detected):.0%}\n"
        f"median area = {_median_finite(finite_area):.0f} px²\n"
        f"median compactness = {_median_finite(finite_circ):.2f}"
    )
    fig.suptitle(
        "08 Blob Centroid Demo Metrics\n" + summary,
        color=TEXT,
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(output_path, dpi=220, facecolor=BG)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = repo_path(args.input_dir)
    output_dir = repo_path(args.output_dir)
    ensure_dir(output_dir)

    metrics_csv = repo_path(args.metrics_csv) if args.metrics_csv else output_dir / "blob_metrics.csv"
    summary_plot = repo_path(args.summary_plot) if args.summary_plot else output_dir / "blob_metrics_summary.png"

    image_paths = find_images(input_dir, args.glob, args.max_frames)

    raw_images: list[np.ndarray] = []
    detections: list[BlobDetectionResult] = []
    frame_names: list[str] = []
    metrics_rows: list[dict[str, float | int | str]] = []

    print(f"Found {len(image_paths)} frame(s).")

    for idx, image_path in enumerate(image_paths):
        image_bgr = load_image(image_path)

        detection = detect_primary_blob(
            image_bgr=image_bgr,
            threshold=args.threshold,
            min_area=args.min_area,
            blur_ksize=args.blur_ksize,
            morph_open=args.morph_open,
            morph_close=args.morph_close,
            invert=args.invert,
        )

        frame_stem = image_path.stem
        annotated_path = output_dir / f"{idx:02d}_{frame_stem}_annotated.png"
        comparison_path = output_dir / f"{idx:02d}_{frame_stem}_comparison.png"

        save_annotated_frame(
            image_bgr=image_bgr,
            detection=detection,
            output_path=annotated_path,
            title=f"{frame_stem}: centroid + blob",
        )

        save_comparison_panel(
            image_bgr=image_bgr,
            detection=detection,
            output_path=comparison_path,
            title=f"{frame_stem}: raw vs centroid vs blob",
        )

        raw_images.append(image_bgr)
        detections.append(detection)
        frame_names.append(frame_stem)
        metrics_rows.append(
            detection_metrics_row(
                frame_index=idx,
                image_path=image_path,
                detection=detection,
            )
        )

        status = "OK" if detection.found else "NO_BLOB"
        print(
            f"[{idx:02d}] {image_path.name}: {status}, "
            f"centroid={detection.centroid_xy}, area={detection.area_px}"
        )

    strip_path = output_dir / "frame_strip.png"
    save_frame_strip(
        images_bgr=raw_images,
        detections=detections,
        labels=frame_names,
        output_path=strip_path,
        title="Centroid vs detected Moon blob across frames",
    )
    write_metrics_csv(metrics_csv, metrics_rows)
    save_metrics_summary_plot(metrics_rows, summary_plot)

    print(f"\nSaved outputs to: {output_dir}")
    print(f"Saved metrics CSV: {metrics_csv}")
    print(f"Saved summary plot: {summary_plot}")


if __name__ == "__main__":
    main()
