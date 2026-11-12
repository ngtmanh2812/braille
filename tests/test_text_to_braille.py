"""Checks for text encoding, downloadable images, and recognition round trips."""

import string

import cv2
import numpy as np
import pytest

from braille_renderer import render_braille
from decoder import decode_patterns, encode_text, text_to_patterns
from pipeline import run_pipeline
from scripts.generate_samples import render_text


def test_abc_encodes_to_known_braille_cells():
    assert text_to_patterns("abc") == ["100000", "110000", "100100"]
    assert encode_text("abc") == "⠁⠃⠉"


def test_alphabet_encoding_matches_unicode_and_decodes_back():
    alphabet = string.ascii_lowercase
    expected = "⠁⠃⠉⠙⠑⠋⠛⠓⠊⠚⠅⠇⠍⠝⠕⠏⠟⠗⠎⠞⠥⠧⠺⠭⠽⠵"
    assert encode_text(alphabet) == expected
    text, braille = decode_patterns(text_to_patterns(alphabet))
    assert text == alphabet
    assert braille == expected


def test_empty_text_has_no_encoded_cells():
    assert text_to_patterns("") == []
    assert encode_text("") == ""


@pytest.mark.parametrize("text", ["ABC", "a b", "123", "a!", "chữ", "a\nb"])
def test_unsupported_text_is_rejected_instead_of_silently_changed(text):
    with pytest.raises(ValueError):
        text_to_patterns(text)
    with pytest.raises(ValueError):
        encode_text(text)
    with pytest.raises(ValueError):
        render_braille(text)


def test_rendered_abc_has_expected_dots_and_white_background():
    image, centers = render_braille("abc")
    expected_centers = [(24.0, 24.0), (78.0, 24.0), (78.0, 42.0),
                        (132.0, 24.0), (150.0, 24.0)]
    assert centers == expected_centers
    assert image.dtype == np.uint8
    assert image.shape == (85, 175)
    assert image[0, 0] == 255
    for x, y in expected_centers:
        assert image[int(y), int(x)] == 0
    result = run_pipeline(image)
    assert len(result.dots) == 5
    assert result.text == "abc"
    assert result.braille == "⠁⠃⠉"


def test_rendered_image_can_be_downloaded_and_loaded_as_png():
    image, _ = render_braille("hello")
    success, png = cv2.imencode(".png", image)
    assert success
    restored = cv2.imdecode(np.frombuffer(png.tobytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    np.testing.assert_array_equal(restored, image)


def test_renderer_rejects_empty_image_content():
    with pytest.raises(ValueError):
        render_braille("")


@pytest.mark.parametrize("geometry", [
    {"dot_spacing": 0},
    {"radius": 0},
    {"margin": 0},
    {"dot_spacing": 18.5},
    {"radius": 9},
    {"margin": 6},
])
def test_invalid_geometry_is_rejected(geometry):
    with pytest.raises(ValueError):
        render_braille("abc", **geometry)


def test_sample_generator_preserves_standard_rendering():
    image, centers = render_braille("abc")
    sample_image, sample_centers = render_text("abc")
    np.testing.assert_array_equal(sample_image, image)
    assert sample_centers == centers
