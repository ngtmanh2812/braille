"""Single-letter preprocessing and classifier contracts, without trained weights."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import handwritten_letters as letters


def _h_image() -> np.ndarray:
    """A simple connected h-shaped fixture; not an OCR accuracy example."""
    image = np.full((100, 100), 255, dtype=np.uint8)
    cv2.line(image, (30, 15), (30, 85), 0, 6)
    cv2.line(image, (30, 50), (60, 50), 0, 6)
    cv2.line(image, (60, 50), (60, 85), 0, 6)
    return image


def test_letter_is_white_ink_on_black_28_pixel_background_without_stretching():
    normalized = letters.preprocess_letter(_h_image())
    assert normalized.shape == (28, 28)
    assert normalized.dtype == np.uint8
    assert normalized.max() > 200
    assert np.all(normalized[:3] == 0)
    assert np.all(normalized[-3:] == 0)
    assert np.all(normalized[:, :3] == 0)
    assert np.all(normalized[:, -3:] == 0)
    ys, xs = np.where(normalized > 64)
    height = int(ys.max() - ys.min() + 1)
    width = int(xs.max() - xs.min() + 1)
    assert 18 <= height <= 20
    assert width < height * 0.65
    assert abs((xs.min() + xs.max()) / 2 - 13.5) <= 1
    assert abs((ys.min() + ys.max()) / 2 - 13.5) <= 1


def test_letter_dot_is_preserved_as_a_separate_component():
    image = np.full((100, 100), 255, dtype=np.uint8)
    cv2.circle(image, (50, 20), 5, 0, -1)
    cv2.rectangle(image, (47, 42), (53, 82), 0, -1)
    normalized = letters.preprocess_letter(image)
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        np.where(normalized > 64, 255, 0).astype(np.uint8), connectivity=8,
    )
    assert count == 3  # Background, dot, and stem; no largest-component-only filter.
    assert all(int(area) > 0 for area in stats[1:, cv2.CC_STAT_AREA])


def test_letter_preprocessing_accepts_grayscale_color_and_image_path(tmp_path):
    gray = _h_image()
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    path = tmp_path / "letter.png"
    assert cv2.imwrite(str(path), bgr)
    expected = letters.preprocess_letter(gray)
    np.testing.assert_array_equal(letters.preprocess_letter(bgr), expected)
    np.testing.assert_array_equal(letters.preprocess_letter(path), expected)


def test_transparent_canvas_is_composited_on_white_before_letter_preprocessing():
    gray = _h_image()
    bgra = np.zeros((100, 100, 4), dtype=np.uint8)
    bgra[:, :, 3] = np.where(gray == 0, 255, 0).astype(np.uint8)
    np.testing.assert_array_equal(letters.preprocess_letter(bgra), letters.preprocess_letter(gray))


@pytest.mark.parametrize("image", [
    None,
    np.empty((0, 0), dtype=np.uint8),
    np.full((50, 50), 255, dtype=np.uint8),
    np.zeros((50, 50), dtype=np.uint8),
    np.zeros((50, 50, 4), dtype=np.uint8),
    np.zeros((50, 50, 2), dtype=np.uint8),
    np.zeros((50,), dtype=np.uint8),
    _h_image().astype(np.float32),
    _h_image().astype(np.uint16),
    _h_image()[:7],
])
def test_blank_or_unsupported_input_is_rejected(image):
    with pytest.raises(ValueError):
        letters.preprocess_letter(image)


def test_missing_and_corrupt_letter_files_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        letters.preprocess_letter(tmp_path / "missing.png")
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        letters.preprocess_letter(path)


def test_model_is_not_ready_when_required_local_files_are_missing_or_empty(tmp_path):
    assert not letters.letter_model_ready(tmp_path)
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    assert not letters.letter_model_ready(tmp_path)
    (tmp_path / "model.pt").write_bytes(b"")
    assert not letters.letter_model_ready(tmp_path)


def test_letter_recognition_passes_normalized_pixels_to_classifier(monkeypatch, tmp_path):
    seen = []

    class FakeClassifier:
        def predict(self, normalized):
            seen.append(normalized.copy())
            return letters.LetterPrediction("h", ("h", "k", "b"), 1.5)

    monkeypatch.setattr(letters, "get_letter_recognizer", lambda model_dir: FakeClassifier())
    prediction = letters.recognize_letter(_h_image(), model_dir=tmp_path)
    assert prediction.text == "h"
    assert prediction.alternatives == ("h", "k", "b")
    assert prediction.processing_time_ms >= 0
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], letters.preprocess_letter(_h_image()))


def test_invalid_letter_is_rejected_before_loading_weights(monkeypatch, tmp_path):
    def unexpected_load(model_dir):
        pytest.fail("An empty letter image must not load classifier weights.")

    monkeypatch.setattr(letters, "get_letter_recognizer", unexpected_load)
    with pytest.raises(ValueError):
        letters.recognize_letter(np.full((40, 40), 255, dtype=np.uint8), model_dir=tmp_path)


def test_letter_network_returns_one_logit_per_latin_class():
    torch = pytest.importorskip("torch")
    network = letters.create_letter_model().eval()
    with torch.inference_mode():
        logits = network(torch.zeros((2, 1, 28, 28)))
    assert logits.shape == (2, 26)
    assert bool(torch.isfinite(logits).all())
