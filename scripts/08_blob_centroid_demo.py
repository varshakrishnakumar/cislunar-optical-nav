
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from vision.blob_detection import BlobDetectionResult, detect_primary_blob
from vision.plotting import (
    save_annotated_frame,
    save_comparison_panel,
    save_frame_strip,
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


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    image_paths = find_images(args.input_dir, args.glob, args.max_frames)

    annotated_paths: list[Path] = []
    raw_images: list[np.ndarray] = []
    detections: list[BlobDetectionResult] = []
    frame_names: list[str] = []

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
        annotated_path = args.output_dir / f"{idx:02d}_{frame_stem}_annotated.png"
        comparison_path = args.output_dir / f"{idx:02d}_{frame_stem}_comparison.png"

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

        annotated_paths.append(annotated_path)
        raw_images.append(image_bgr)
        detections.append(detection)
        frame_names.append(frame_stem)

        status = "OK" if detection.found else "NO_BLOB"
        print(
            f"[{idx:02d}] {image_path.name}: {status}, "
            f"centroid={detection.centroid_xy}, area={detection.area_px}"
        )

    strip_path = args.output_dir / "frame_strip.png"
    save_frame_strip(
        images_bgr=raw_images,
        detections=detections,
        labels=frame_names,
        output_path=strip_path,
        title="Centroid vs detected Moon blob across frames",
    )

    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
