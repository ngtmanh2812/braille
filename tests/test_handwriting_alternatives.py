"""Check optional re-reading inputs and raw candidates without model weights."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import drawing_canvas
import handwriting


def _word_image() -> np.ndarray:
    image = np.full((150, 430, 3), 255, dtype=np.uint8)
    # Two separated ink groups allow checking that cropping preserves word gaps.
    cv2.rectangle(image, (55, 35), (100, 115), (0, 0, 0), -1)
    cv2.rectangle(image, (260, 35), (320, 115), (0, 0, 0), -1)
    return image


def test_variants_preserve_original_and_keep_white_margins():
    original = _word_image()
    before = original.copy()
    variants = handwriting.make_handwriting_variants(original)
    assert len(variants) == 3
    np.testing.assert_array_equal(original, before)
    np.testing.assert_array_equal(variants[0], before)
    assert not np.shares_memory(variants[0], original)
    for image in variants[1:]:
        assert image.ndim == 3 and image.shape[2] == 3
        assert image.dtype == np.uint8
        assert np.any(image < 100)
        assert np.all(image[0] == 255)
        assert np.all(image[-1] == 255)
        assert np.all(image[:, 0] == 255)
        assert np.all(image[:, -1] == 255)


def test_variant_cropping_preserves_the_gap_between_ink_groups():
    for image in handwriting.make_handwriting_variants(_word_image())[1:]:
        ink_columns = np.any(np.any(image < 100, axis=2), axis=0)
        active = np.flatnonzero(ink_columns)
        assert active.size > 0
        gaps = np.diff(active)
        # Two groups must remain separated rather than independently packed.
        assert gaps.max() > 20


def test_alternative_wrapper_keeps_raw_candidates_for_review(monkeypatch, tmp_path):
    calls = []
    raw = ("or ) anln .", "manh .", "manh", "Manh")

    class FakeReader:
        def recognize_alternatives(self, rgb):
            calls.append(rgb.copy())
            return handwriting.HandwritingAlternatives(raw, 7.5)

    monkeypatch.setattr(handwriting, "get_recognizer", lambda model_dir: FakeReader())
    bgr = _word_image()
    bgr[40, 60] = (10, 30, 80)
    result = handwriting.recognize_handwriting_alternatives(bgr, model_dir=tmp_path)
    assert result.texts == raw
    assert result.processing_time_ms == 7.5
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][40, 60], [80, 30, 10])


@pytest.mark.parametrize("image", [
    np.full((60, 100, 3), 255, dtype=np.uint8),
    np.empty((0, 0), dtype=np.uint8),
    np.zeros((40, 80, 2), dtype=np.uint8),
])
def test_invalid_alternative_input_is_rejected_before_model_load(monkeypatch, image):
    def unexpected_load(model_dir):
        pytest.fail("Invalid/blank images must not load the OCR model.")

    monkeypatch.setattr(handwriting, "get_recognizer", unexpected_load)
    with pytest.raises(ValueError):
        handwriting.recognize_handwriting_alternatives(image)


def _blank_canvas() -> np.ndarray:
    return np.full((drawing_canvas.CANVAS_HEIGHT, drawing_canvas.CANVAS_WIDTH, 3),
                   255, dtype=np.uint8)


def test_blank_and_centered_strokes_do_not_trigger_border_warning():
    canvas = _blank_canvas()
    assert not drawing_canvas.drawing_touches_edge(canvas)
    cv2.line(canvas, (100, 60), (200, 150), (0, 0, 0), 5)
    assert not drawing_canvas.drawing_touches_edge(canvas)


@pytest.mark.parametrize("edge", ["left", "right", "top", "bottom"])
def test_stroke_near_each_canvas_edge_is_detected(edge):
    canvas = _blank_canvas()
    height, width = canvas.shape[:2]
    locations = {"left": (2, height // 2), "right": (width - 3, height // 2),
                 "top": (width // 2, 2), "bottom": (width // 2, height - 3)}
    x, y = locations[edge]
    canvas[y, x] = 0
    assert drawing_canvas.drawing_touches_edge(canvas, margin=6)


@pytest.mark.parametrize("margin", [-1, 1.5, True])
def test_edge_warning_rejects_invalid_margin(margin):
    with pytest.raises(ValueError):
        drawing_canvas.drawing_touches_edge(_blank_canvas(), margin=margin)


def test_edge_warning_rejects_invalid_canvas_shape():
    with pytest.raises(ValueError):
        drawing_canvas.drawing_touches_edge(np.full((20, 30, 3), 255, dtype=np.uint8))
