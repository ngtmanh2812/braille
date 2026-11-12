"""Map detected dot centers onto the six positions of a Braille cell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from dot_detection import Dot


GridMode = Literal["estimated", "fixed"]


@dataclass(frozen=True)
class GridConfig:
    """Grid geometry. Fixed mode is useful for calibrated, standard images."""

    mode: GridMode = "estimated"
    dot_spacing: float | None = None
    cell_stride: float | None = None
    origin_x: float | None = None
    origin_y: float | None = None
    tolerance_fraction: float = 0.38
    estimate_rotation: bool = True


@dataclass(frozen=True)
class AlignedCell:
    """A cell with one optional detected center for each numbered dot position."""

    index: int
    origin: tuple[float, float]
    positions: tuple[tuple[float, float] | None, ...]
    bits: str


def _centers(dots: Sequence[Dot | tuple[float, float]]) -> np.ndarray:
    vals = [d.center if isinstance(d, Dot) else d for d in dots]
    return np.asarray(vals, dtype=np.float64).reshape((-1, 2))


def _estimate_spacing(points: np.ndarray, requested: float | None) -> float:
    if requested is not None:
        if requested <= 0:
            raise ValueError("dot_spacing must be positive.")
        return float(requested)
    if len(points) >= 2:
        distances = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
        distances[distances < 3.0] = np.inf
        nearest = np.min(distances, axis=1)
        nearest = nearest[np.isfinite(nearest)]
        if nearest.size:
            # Braille's nearest geometric interval is commonly visible among repeated dots;
            # the lower quartile reduces bias from gaps between isolated cells.
            estimate = float(np.percentile(nearest, 25))
            # A repeated one-dot pattern such as "aaaa" has no within-cell pair:
            # its nearest horizontal interval is the three-pitch cell stride.
            unique_x = np.unique(np.round(points[:, 0], 1))
            x_gaps = np.diff(unique_x)
            x_gaps = x_gaps[x_gaps >= 3.0]
            same_row = float(np.ptp(points[:, 1])) <= max(3.0, estimate * 0.15)
            regular_x_gaps = (len(x_gaps) >= 1
                              and float(np.std(x_gaps)) / max(float(np.mean(x_gaps)), 1.0) < 0.12)
            if same_row and regular_x_gaps and estimate > 2.0 * 18.0:
                estimate /= 3.0
            if estimate >= 3:
                return estimate
    return 18.0  # documented fallback for a typical small scanned or synthetic line


def _estimate_slope(points: np.ndarray, spacing: float, enabled: bool) -> float:
    """Estimate a small line rotation by minimizing row-coordinate residues."""
    if not enabled or len(points) < 4 or np.ptp(points[:, 0]) < spacing * 2:
        return 0.0
    pairs_i, pairs_j = np.triu_indices(len(points), 1)
    dx = points[pairs_j, 0] - points[pairs_i, 0]
    dy = points[pairs_j, 1] - points[pairs_i, 1]
    keep = np.abs(dx) >= spacing * 0.8
    dx, dy = dx[keep], dy[keep]
    if len(dx) > 30000:
        selection = np.linspace(0, len(dx) - 1, 30000).astype(int)
        dx, dy = dx[selection], dy[selection]
    best_slope, best_score = 0.0, float("inf")
    for slope in np.linspace(-0.12, 0.12, 97):
        delta = dy - slope * dx
        residue = np.abs((delta + spacing / 2) % spacing - spacing / 2)
        score = float(np.median(residue))
        if score < best_score:
            best_slope, best_score = float(slope), score
    return best_slope if best_score < spacing * 0.24 else 0.0


def _nearest_x_slot(x: float, origin: float, stride: float, spacing: float) -> tuple[float, int, int]:
    """Find a nearby left/right column without scanning every cell in the line."""
    relative_cell = (x - origin) / stride
    lower = int(np.floor(relative_cell))
    options = [
        (abs(x - (origin + cell_i * stride + col * spacing)), cell_i, col)
        for cell_i in range(max(0, lower - 1), lower + 3)
        for col in (0, 1)
    ]
    return min(options)


def _candidate_x_layout(points_x: np.ndarray, spacing: float, config: GridConfig) -> tuple[float, float]:
    stride_choices = [float(config.cell_stride)] if config.cell_stride else [2.7 * spacing, 3.0 * spacing, 3.3 * spacing]
    tol = max(2.0, config.tolerance_fraction * spacing)
    if config.origin_x is not None:
        return float(config.origin_x), stride_choices[0]

    min_x = float(np.min(points_x))
    best: tuple[int, float, float, float, float] | None = None
    for stride in stride_choices:
        origins = {min_x - phase for phase in (0.0, spacing)}
        for x in points_x:
            approximate_cell = int(np.floor((float(x) - min_x) / stride))
            for cell_i in range(max(0, approximate_cell - 1), approximate_cell + 3):
                for column in (0, 1):
                    origin = float(x - cell_i * stride - column * spacing)
                    if min_x - spacing - tol <= origin <= min_x + tol:
                        origins.add(origin)
        for origin in origins:
            errors = [_nearest_x_slot(float(x), origin, stride, spacing)[0] for x in points_x]
            valid = sum(error <= tol for error in errors)
            mean_error = float(np.mean(np.square(errors))) if errors else 0.0
            # When the first observed column is ambiguous, prefer treating it as
            # the left column so a repeated single-dot "a" stays dot 1.
            key = (valid, -mean_error, -abs(stride / spacing - 3.0), origin, stride)
            if best is None or key > best:
                best = key
                chosen = (origin, stride)
    return chosen if best is not None else (min_x, stride_choices[0])


def align_dots(
    dots: Sequence[Dot | tuple[float, float]],
    image_shape: tuple[int, ...] | None = None,
    config: GridConfig | None = None,
) -> list[AlignedCell]:
    """Assign each detected center to dots 1–6, leaving absent locations empty.

    The default estimated mode supports one text line with mild rotation. For known
    calibration images, fixed mode accepts explicit spacing and grid origins.
    """
    config = config or GridConfig()
    if config.mode not in ("estimated", "fixed"):
        raise ValueError(f"Unsupported grid mode: {config.mode!r}")
    points = _centers(dots)
    if not len(points):
        return []
    spacing = _estimate_spacing(points, config.dot_spacing)
    slope = _estimate_slope(points, spacing, config.estimate_rotation and config.mode == "estimated")
    projected_y = points[:, 1] - slope * points[:, 0]
    y_origin = float(config.origin_y) if config.origin_y is not None else float(np.min(projected_y))
    x_origin, stride = _candidate_x_layout(points[:, 0], spacing, config)
    if config.mode == "fixed":
        if config.dot_spacing is None:
            raise ValueError("Fixed grid mode requires dot_spacing.")
        if config.origin_y is None:
            y_origin = float(np.min(projected_y))

    tol = max(2.0, config.tolerance_fraction * spacing)
    cell_data: dict[int, list[tuple[tuple[float, float] | None, float]]] = {}
    for (x, y), py in zip(points, projected_y):
        row = int(round((py - y_origin) / spacing))
        if row not in (0, 1, 2) or abs((py - y_origin) - row * spacing) > tol:
            continue
        error, cell_i, col = _nearest_x_slot(float(x), x_origin, stride, spacing)
        if error > tol:
            continue
        slot = row if col == 0 else row + 3
        position_index = cell_data.setdefault(cell_i, [((None), float("inf")) for _ in range(6)])
        previous_point, previous_error = position_index[slot]
        if error < previous_error:
            position_index[slot] = ((float(x), float(y)), error)

    if not cell_data:
        return []
    cells: list[AlignedCell] = []
    for cell_i in range(max(cell_data) + 1):
        values = cell_data.get(cell_i, [((None), float("inf")) for _ in range(6)])
        positions = tuple(item[0] for item in values)
        bits = "".join("1" if point is not None else "0" for point in positions)
        cells.append(AlignedCell(cell_i, (x_origin + cell_i * stride, y_origin), positions, bits))
    return cells
