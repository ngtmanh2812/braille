"""Small command-line entry point for a Braille image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from pipeline import PipelineConfig, run_pipeline
from preprocessing import PreprocessConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Recognize one line of Grade 1 Latin Braille.")
    parser.add_argument("image", type=Path, help="Path to JPG or PNG input")
    parser.add_argument("--threshold", choices=("global", "otsu", "adaptive"), default="otsu")
    parser.add_argument("--threshold-value", type=int, default=127)
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--save-debug", type=Path, help="Optional path to save the annotated image")
    args = parser.parse_args()
    config = PipelineConfig(preprocessing=PreprocessConfig(
        threshold_method=args.threshold, threshold_value=args.threshold_value,
        denoise=not args.no_denoise,
    ))
    try:
        result = run_pipeline(args.image, config)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Text: {result.text}")
    print(f"Braille: {result.braille}")
    print(f"Dots: {len(result.dots)} | Cells: {len(result.cells)} | Time: {result.processing_time_ms:.2f} ms")
    if args.save_debug:
        args.save_debug.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_debug), result.debug_images["annotated"])
        print(f"Annotated image: {args.save_debug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
