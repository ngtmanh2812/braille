"""End-to-end Braille image recognition pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from decoder import bits_to_unicode, decode_pattern
from dot_detection import Dot, DotDetectionConfig, detect_dots, draw_detections
from grid_alignment import AlignedCell, GridConfig, align_dots
from preprocessing import PreprocessConfig, preprocess_image


@dataclass(frozen=True)
class PipelineConfig:
    preprocessing: PreprocessConfig = field(default_factory=PreprocessConfig)
    dot_detection: DotDetectionConfig = field(default_factory=DotDetectionConfig)
    grid: GridConfig = field(default_factory=GridConfig)


@dataclass
class PipelineResult:
    """Recognition output and visual diagnostics for one input image."""

    text: str
    braille: str
    patterns: list[str]
    dots: list[Dot]
    cells: list[AlignedCell]
    debug_images: dict[str, np.ndarray]
    processing_time_ms: float
    image_shape: tuple[int, ...]


def _load_input(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, (str, Path)):
        path = Path(image)
        loaded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if loaded is None:
            raise ValueError(f"Could not read image file: {path}")
        return loaded
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Input must be an image path or a non-empty NumPy image.")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ValueError("Expected a grayscale, BGR, or BGRA image.")
    return image


def run_pipeline(image: str | Path | np.ndarray, config: PipelineConfig | None = None) -> PipelineResult:
    """Run preprocessing, dot detection, grid alignment, and character decoding."""
    started = time.perf_counter()
    config = config or PipelineConfig()
    original = _load_input(image)
    prep = preprocess_image(original, config.preprocessing)
    dots = detect_dots(prep.binary, config.dot_detection)
    cells = align_dots(dots, original.shape, config.grid)
    patterns = [cell.bits for cell in cells]
    text = "".join(decode_pattern(pattern) for pattern in patterns)
    braille = "".join(bits_to_unicode(pattern) for pattern in patterns)
    annotated = draw_detections(original, dots)
    if cells:
        for cell in cells:
            x, y = (int(round(v)) for v in cell.origin)
            cv2.putText(annotated, f"{cell.index}:{cell.bits}", (x, max(14, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 100, 255), 1, cv2.LINE_AA)
    elapsed = (time.perf_counter() - started) * 1000.0
    return PipelineResult(
        text=text,
        braille=braille,
        patterns=patterns,
        dots=dots,
        cells=cells,
        debug_images={"original": original.copy(), "grayscale": prep.grayscale,
                      "binary": prep.binary, "annotated": annotated},
        processing_time_ms=elapsed,
        image_shape=tuple(int(v) for v in original.shape),
    )
