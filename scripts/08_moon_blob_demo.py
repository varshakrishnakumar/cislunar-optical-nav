from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.vision.blob_detection import detect_primary_blob
from src.vision.plotting import (
    annotate_crop,
    save_video_from_frames,
    save_crop_strip,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--glob", type=str, default="*.png")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roi", nargs=4, type=int, required=True, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--strip-count", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-area", type=int, default=10)
    parser.add_argument("--blur-ksize", type=int, default=5)
    parser.add_argument("--morph-close", type=int, default=3)
    parser.add_argument("--morph-open", type=int, default=0)
    parser.add_argument("--invert", action="store_true")
    return parser.parse_args()


def load_paths(input_dir: Path, pattern: str, max_frames: int | None):
    paths = sorted(input_dir.glob(pattern))
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths:
        raise FileNotFoundError("No frames found.")
    return paths


def crop_roi(image_bgr: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = roi
    return image_bgr[y0:y1, x0:x1].copy()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    roi = tuple(args.roi)
    frame_paths = load_paths(args.input_dir, args.glob, args.max_frames)

    annotated_frames = []
    labels = []

    for i, path in enumerate(frame_paths):
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue

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

    video_path = args.output_dir / "moon_blob_demo.mp4"
    strip_path = args.output_dir / "moon_blob_strip.png"

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

    print(f"Saved video: {video_path}")
    print(f"Saved strip: {strip_path}")


if __name__ == "__main__":
    main()
