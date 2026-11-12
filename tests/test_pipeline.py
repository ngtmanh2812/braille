import cv2
import numpy as np
import pytest

from pipeline import PipelineConfig, run_pipeline
from preprocessing import PreprocessConfig


def _render(text: str) -> np.ndarray:
    from decoder import LETTER_TO_BITS
    spacing, radius, margin = 18, 6, 24
    image = np.full((2 * margin + 2 * spacing + 1, 2 * margin + (len(text) - 1) * 3 * spacing + spacing + 1), 255, np.uint8)
    for i, char in enumerate(text):
        for bit_i, bit in enumerate(LETTER_TO_BITS[char]):
            if bit == "1":
                x = margin + i * 3 * spacing + (bit_i // 3) * spacing
                y = margin + (bit_i % 3) * spacing
                cv2.circle(image, (x, y), radius, 0, -1, cv2.LINE_AA)
    return image


def test_pipeline_decodes_multi_cell_numpy_image():
    result = run_pipeline(_render("abc"))
    assert result.text == "abc"
    assert len(result.patterns) == 3
    assert len(result.dots) == 5
    assert set(result.debug_images) == {"original", "grayscale", "binary", "annotated"}
    assert result.processing_time_ms >= 0


def test_empty_image_is_a_non_crashing_empty_prediction():
    result = run_pipeline(np.full((80, 100), 255, np.uint8))
    assert result.text == ""
    assert result.braille == ""
    assert result.dots == []
    assert result.cells == []


def test_invalid_path_raises_clear_error(tmp_path):
    with pytest.raises(ValueError, match="Could not read image"):
        run_pipeline(tmp_path / "missing.png")


def test_malformed_numpy_image_raises_clear_error():
    with pytest.raises(ValueError, match="Expected a grayscale"):
        run_pipeline(np.zeros((20, 20, 2), np.uint8))


def test_pipeline_accepts_image_path(tmp_path):
    image_path = tmp_path / "line.png"
    assert cv2.imwrite(str(image_path), _render("a"))
    result = run_pipeline(image_path, PipelineConfig(preprocessing=PreprocessConfig(threshold_method="global")))
    assert result.text == "a"
