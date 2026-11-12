"""Offline checks for handwriting input and confirmed Latin-to-Braille text.

No test downloads model files or loads neural-network weights.
"""

import cv2
import numpy as np
import pytest

import handwriting
from braille_renderer import render_braille_line
from decoder import encode_text
from latin_line import encode_latin_line, normalize_latin_line


def _writing_image() -> np.ndarray:
    image = np.full((20, 30), 255, dtype=np.uint8)
    image[5:15, 8:10] = 0
    return image


def test_confirmed_latin_line_normalizes_case_and_repeated_spaces():
    assert normalize_latin_line("  Hello   WORLD  ") == "hello world"
    assert encode_latin_line(" A  B ") == "⠁\u2800⠃"


@pytest.mark.parametrize("symbol,expected", [(",", "\u2802"), ("'", "\u2804"), (".", "\u2832")])
def test_basic_punctuation_uses_correct_braille_cells(symbol, expected):
    assert encode_latin_line(symbol) == expected


def test_sentence_preserves_apostrophe_comma_period_and_spaces():
    assert normalize_latin_line("  Don't,  stop. ") == "don't, stop."
    expected = "⠙⠕⠝⠄⠞⠂\u2800⠎⠞⠕⠏⠲"
    assert encode_latin_line("Don't, stop.") == expected


def test_punctuation_renderer_uses_same_dot_patterns_as_unicode():
    image, centers = render_braille_line(",'.")
    assert centers == [(24.0, 42.0), (78.0, 60.0),
                       (132.0, 42.0), (150.0, 42.0), (150.0, 60.0)]
    assert image.shape == (85, 175)


@pytest.mark.parametrize("text", [
    "", "   ", None, 123, "a\nb", "a\tb", "abc1", "hello!", "a-b",
    "xin chào", "a" * 201,
])
def test_unsupported_confirmed_text_is_not_silently_discarded(text):
    with pytest.raises(ValueError):
        normalize_latin_line(text)
    with pytest.raises(ValueError):
        encode_latin_line(text)


def test_line_renderer_keeps_an_empty_cell_for_a_word_space():
    image, centers = render_braille_line("a b")
    assert centers == [(24.0, 24.0), (132.0, 24.0), (132.0, 42.0)]
    assert image.dtype == np.uint8
    assert image.shape == (85, 175)
    assert np.all(image[:, 70:105] == 255)
    assert image[24, 24] == 0
    assert image[24, 132] == 0
    normalized_image, normalized_centers = render_braille_line("  A   B  ")
    np.testing.assert_array_equal(normalized_image, image)
    assert normalized_centers == centers


def test_old_letter_only_encoder_remains_strict():
    with pytest.raises(ValueError):
        encode_text("a b")


def test_grayscale_handwriting_is_prepared_as_rgb():
    image = _writing_image()
    prepared = handwriting.prepare_handwriting_image(image)
    assert prepared.shape == (20, 30, 3)
    assert prepared.dtype == np.uint8
    for channel in range(3):
        np.testing.assert_array_equal(prepared[:, :, channel], image)


def test_opencv_bgr_channels_are_converted_to_rgb():
    image = cv2.cvtColor(_writing_image(), cv2.COLOR_GRAY2BGR)
    image[7, 9] = (11, 22, 33)
    prepared = handwriting.prepare_handwriting_image(image)
    np.testing.assert_array_equal(prepared[7, 9], [33, 22, 11])


def test_transparent_background_is_flattened_onto_white():
    image = np.zeros((20, 30, 4), dtype=np.uint8)
    image[5:15, 8:10, 3] = 255
    image[7, 8, 3] = 128
    prepared = handwriting.prepare_handwriting_image(image)
    np.testing.assert_array_equal(prepared[0, 0], [255, 255, 255])
    np.testing.assert_array_equal(prepared[5, 8], [0, 0, 0])
    np.testing.assert_allclose(prepared[7, 8], [127, 127, 127], atol=1)


def test_handwriting_image_can_be_loaded_from_a_path(tmp_path):
    path = tmp_path / "writing.png"
    assert cv2.imwrite(str(path), _writing_image())
    prepared = handwriting.prepare_handwriting_image(path)
    assert prepared.shape == (20, 30, 3)
    assert prepared[6, 8, 0] == 0


@pytest.mark.parametrize("image", [
    np.empty((0, 0), dtype=np.uint8),
    np.full((20, 30), 255, dtype=np.uint8),
    np.zeros((20, 30), dtype=np.uint8),
    np.zeros((20, 30, 4), dtype=np.uint8),
    np.zeros((20, 30, 2), dtype=np.uint8),
    _writing_image()[:7],
    np.zeros((20,), dtype=np.uint8),
    _writing_image().astype(np.float32),
    _writing_image().astype(np.uint16),
])
def test_empty_blank_or_unsupported_images_raise_clear_errors(image):
    with pytest.raises(ValueError):
        handwriting.prepare_handwriting_image(image)


def test_missing_and_corrupt_image_files_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        handwriting.prepare_handwriting_image(tmp_path / "missing.png")
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not a PNG file")
    with pytest.raises(ValueError):
        handwriting.prepare_handwriting_image(corrupt)


def test_excessive_pixel_count_is_rejected_without_allocating_a_large_image():
    row = np.full((1, 4500), 255, dtype=np.uint8)
    row[0, 0] = 0
    oversized = np.broadcast_to(row, (4500, 4500))
    with pytest.raises(ValueError):
        handwriting.prepare_handwriting_image(oversized)


def test_empty_model_directory_is_reported_as_not_ready(tmp_path):
    assert not handwriting.model_files_ready(tmp_path)


def test_recognition_preserves_raw_ocr_for_user_review(monkeypatch, tmp_path):
    received = []

    class FakeRecognizer:
        def recognize(self, rgb_image):
            received.append(rgb_image)
            return handwriting.HandwritingResult(
                text="Hello, world!", processing_time_ms=2.5, truncated=True,
            )

    monkeypatch.setattr(handwriting, "get_recognizer", lambda model_dir: FakeRecognizer())
    result = handwriting.recognize_handwriting(_writing_image(), model_dir=tmp_path)
    assert result.text == "Hello, world!"
    assert result.truncated is True
    assert result.processing_time_ms >= 0
    assert len(received) == 1
    assert received[0].shape == (20, 30, 3)
    with pytest.raises(ValueError):
        encode_latin_line(result.text)


def test_invalid_image_is_rejected_before_loading_any_model(monkeypatch, tmp_path):
    def unexpected_model_load(model_dir):
        pytest.fail("A blank image must be rejected before loading model weights.")

    monkeypatch.setattr(handwriting, "get_recognizer", unexpected_model_load)
    with pytest.raises(ValueError):
        handwriting.recognize_handwriting(
            np.full((20, 30), 255, dtype=np.uint8), model_dir=tmp_path,
        )
