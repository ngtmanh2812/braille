"""Render labeled Braille cells using the same alphabet as the decoder."""

from __future__ import annotations

import cv2
import numpy as np

from decoder import text_to_patterns
from latin_line import line_to_patterns


def render_braille(text: str, *, dot_spacing: int = 18, radius: int = 6,
                   margin: int = 24) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Draw one line of black dots and return the image and ground-truth centers.

    Input supports lowercase Latin a-z only. Empty text and geometry that would
    overlap or crop dots raise ValueError. Returned images are uint8 grayscale.
    """
    patterns = text_to_patterns(text)
    return _render_patterns(patterns, dot_spacing=dot_spacing, radius=radius, margin=margin)


def render_braille_line(text: str, *, dot_spacing: int = 18, radius: int = 6,
                        margin: int = 24) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Render a reviewed Latin line, including word gaps and basic punctuation."""
    patterns = line_to_patterns(text)
    return _render_patterns(patterns, dot_spacing=dot_spacing, radius=radius, margin=margin)


def _render_patterns(patterns: list[str], *, dot_spacing: int, radius: int,
                      margin: int) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Shared geometry for letters and optional blank word-separator cells."""
    if not patterns:
        raise ValueError("Vui lòng nhập ít nhất một chữ cái a–z.")
    if any(type(value) is not int or value <= 0 for value in (dot_spacing, radius, margin)):
        raise ValueError("Khoảng cách chấm, bán kính và lề phải là số nguyên dương.")
    if 2 * radius >= dot_spacing or margin <= radius:
        raise ValueError("Cần khoảng cách chấm lớn hơn đường kính và lề lớn hơn bán kính chấm.")

    stride = 3 * dot_spacing
    width = margin * 2 + (len(patterns) - 1) * stride + dot_spacing + 1
    height = margin * 2 + 2 * dot_spacing + 1
    image = np.full((height, width), 255, dtype=np.uint8)
    centers: list[tuple[float, float]] = []
    for index, bits in enumerate(patterns):
        for dot_index, bit in enumerate(bits):
            if bit == "1":
                x = margin + index * stride + (dot_index // 3) * dot_spacing
                y = margin + (dot_index % 3) * dot_spacing
                cv2.circle(image, (x, y), radius, 0, -1, cv2.LINE_AA)
                centers.append((float(x), float(y)))
    return image, centers
