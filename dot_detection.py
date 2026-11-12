"""Connected-component dot detection and debug visualization."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DotDetectionConfig:
    min_area: int = 12
    max_area: int = 5000
    min_dimension: int = 3
    max_dimension: int = 120
    min_aspect_ratio: float = 0.45
    max_aspect_ratio: float = 2.2
    min_circularity: float = 0.30


@dataclass(frozen=True)
class Dot:
    """One detected foreground component."""

    center: tuple[float, float]
    area: int
    bbox: tuple[int, int, int, int]
    circularity: float


def detect_dots(binary: np.ndarray, config: DotDetectionConfig | None = None) -> list[Dot]:
    """Find dot-like connected components in a binary mask (foreground is white)."""
    config = config or DotDetectionConfig()
    if not isinstance(binary, np.ndarray) or binary.size == 0:
        raise ValueError("Binary image is empty or invalid.")
    if binary.ndim != 2:
        raise ValueError("Dot detection expects a single-channel binary image.")
    mask = np.where(binary > 0, 255, 0).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_circularity: dict[tuple[int, int, int, int], float] = {}
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
        contour_circularity[(x, y, w, h)] = circularity

    dots: list[Dot] = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if not (config.min_area <= area <= config.max_area):
            continue
        if min(w, h) < config.min_dimension or max(w, h) > config.max_dimension:
            continue
        ratio = w / float(h)
        if not (config.min_aspect_ratio <= ratio <= config.max_aspect_ratio):
            continue
        circularity = contour_circularity.get((x, y, w, h), 0.0)
        if circularity < config.min_circularity:
            continue
        cx, cy = centroids[label]
        dots.append(Dot((float(cx), float(cy)), area, (x, y, w, h), circularity))
    dots.sort(key=lambda dot: (dot.center[1], dot.center[0]))
    return dots


def draw_detections(image: np.ndarray, dots: list[Dot]) -> np.ndarray:
    """Return a copy annotated with component bounds and center points."""
    if image.ndim == 2:
        annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        annotated = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        annotated = image.copy()
    for index, dot in enumerate(dots, start=1):
        x, y, w, h = dot.bbox
        center = tuple(int(round(v)) for v in dot.center)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 180, 0), 1)
        cv2.circle(annotated, center, 2, (0, 0, 255), -1)
        cv2.putText(annotated, str(index), (x, max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 0, 0), 1, cv2.LINE_AA)
    return annotated
