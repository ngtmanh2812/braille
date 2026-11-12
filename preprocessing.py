"""Image preprocessing for dark-dot Braille on a light background."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


ThresholdMethod = Literal["global", "otsu", "adaptive"]
MorphologyMethod = Literal["none", "open", "close"]


@dataclass(frozen=True)
class PreprocessConfig:
    """Options for grayscale conversion, denoising, binarization, and cleanup."""

    threshold_method: ThresholdMethod = "otsu"
    threshold_value: int = 127
    adaptive_block_size: int = 31
    adaptive_c: int = 7
    denoise: bool = True
    blur_kernel_size: int = 3
    morphology: MorphologyMethod = "none"
    morphology_kernel_size: int = 3


@dataclass
class PreprocessResult:
    """Grayscale input and inverted binary mask (dots are white/255)."""

    grayscale: np.ndarray
    binary: np.ndarray


def _validate_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Image is empty or is not a NumPy array.")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("Expected a grayscale, BGR, or BGRA image.")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def preprocess_image(image: np.ndarray, config: PreprocessConfig | None = None) -> PreprocessResult:
    """Convert an image to grayscale and a binary mask with dark dots as foreground."""
    config = config or PreprocessConfig()
    gray = _validate_image(image)
    if config.denoise:
        kernel = max(1, int(config.blur_kernel_size))
        if kernel % 2 == 0:
            kernel += 1
        working = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    else:
        working = gray

    method = config.threshold_method.lower()
    if method == "global":
        _, binary = cv2.threshold(working, int(config.threshold_value), 255, cv2.THRESH_BINARY_INV)
    elif method == "otsu":
        _, binary = cv2.threshold(working, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    elif method == "adaptive":
        block = max(3, int(config.adaptive_block_size))
        if block % 2 == 0:
            block += 1
        binary = cv2.adaptiveThreshold(
            working, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            block, int(config.adaptive_c),
        )
    else:
        raise ValueError(f"Unsupported threshold method: {config.threshold_method!r}")

    morphology = config.morphology.lower()
    if morphology != "none":
        if morphology not in ("open", "close"):
            raise ValueError(f"Unsupported morphology operation: {config.morphology!r}")
        size = max(1, int(config.morphology_kernel_size))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        op = cv2.MORPH_OPEN if morphology == "open" else cv2.MORPH_CLOSE
        binary = cv2.morphologyEx(binary, op, kernel)

    return PreprocessResult(grayscale=gray, binary=binary)
