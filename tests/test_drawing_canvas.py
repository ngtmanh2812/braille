"""Validate browser canvas image payloads without OCR or network access."""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from drawing_canvas import CANVAS_HEIGHT, CANVAS_WIDTH, decode_canvas_payload


def _payload(image: np.ndarray | None = None, *, stroke_count: int = 2) -> dict:
    if image is None:
        image = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), 255, dtype=np.uint8)
        cv2.line(image, (80, 60), (80, 180), (0, 0, 0), 6)
        cv2.line(image, (80, 120), (170, 120), (0, 0, 0), 6)
    encoded, png = cv2.imencode(".png", image)
    assert encoded
    return {
        "version": 1,
        "width": CANVAS_WIDTH,
        "height": CANVAS_HEIGHT,
        "stroke_count": stroke_count,
        "image": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii"),
    }


def test_no_drawing_yet_returns_none():
    assert decode_canvas_payload(None) is None


def test_canvas_snapshot_is_available_when_the_iframe_is_remounted(monkeypatch):
    import drawing_canvas
    import streamlit as st

    session = {}
    calls = []
    response = [_payload()]

    def component(**kwargs):
        calls.append(kwargs)
        return response.pop() if response else kwargs["default"]

    monkeypatch.setattr(st, "session_state", session)
    monkeypatch.setattr(drawing_canvas, "_component", lambda: component)
    first = drawing_canvas.render_drawing_canvas(key="test_board")
    restored = drawing_canvas.render_drawing_canvas(key="test_board")
    np.testing.assert_array_equal(restored, first)
    assert calls[1]["initial_png"] == session["_snapshot_test_board"]["image"]
    assert calls[1]["initial_stroke_count"] == 2
    assert calls[1]["default"] == session["_snapshot_test_board"]


def test_canvas_png_decodes_to_actual_black_strokes_on_white_background():
    assert (CANVAS_WIDTH, CANVAS_HEIGHT) == (1000, 260)
    image = decode_canvas_payload(_payload())
    assert image is not None
    assert image.shape == (260, 1000, 3)
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image[0, 0], [255, 255, 255])
    np.testing.assert_array_equal(image[120, 80], [0, 0, 0])
    np.testing.assert_array_equal(image[120, 150], [0, 0, 0])


def test_cleared_canvas_is_a_valid_blank_image_for_the_caller_to_handle():
    white = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), 255, dtype=np.uint8)
    image = decode_canvas_payload(_payload(white, stroke_count=0))
    assert image is not None
    np.testing.assert_array_equal(image, white)


@pytest.mark.parametrize("value", ["", [], 0, True, b"not a canvas payload"])
def test_non_dictionary_payload_is_rejected(value):
    with pytest.raises(ValueError):
        decode_canvas_payload(value)


@pytest.mark.parametrize("field", ["version", "width", "height", "stroke_count", "image"])
def test_missing_required_payload_fields_are_rejected(field):
    payload = _payload()
    del payload[field]
    with pytest.raises(ValueError):
        decode_canvas_payload(payload)


@pytest.mark.parametrize("field,value", [
    ("version", 2), ("version", "1"), ("version", True),
    ("width", 999), ("width", "1000"), ("width", 1000.0),
    ("height", 259), ("height", "260"), ("height", 260.0),
    ("stroke_count", -1), ("stroke_count", 1.5), ("stroke_count", "1"),
    ("stroke_count", True), ("image", None), ("image", ["image"]),
])
def test_invalid_metadata_types_and_values_are_rejected(field, value):
    payload = _payload()
    payload[field] = value
    with pytest.raises(ValueError):
        decode_canvas_payload(payload)


@pytest.mark.parametrize("image_value", [
    "data:image/png;base64,not base64!",
    "data:image/png;base64,",
    "data:image/png;base64,aGVsbG8=",
    "data:image/jpeg;base64,aGVsbG8=",
    "https://example.com/drawing.png",
])
def test_invalid_data_urls_raise_value_error(image_value):
    payload = _payload()
    payload["image"] = image_value
    with pytest.raises(ValueError):
        decode_canvas_payload(payload)


def test_base64_decoder_does_not_ignore_embedded_whitespace():
    payload = _payload()
    prefix, encoded = payload["image"].split(",", 1)
    payload["image"] = prefix + "," + encoded[:20] + "\n" + encoded[20:]
    with pytest.raises(ValueError):
        decode_canvas_payload(payload)


def test_jpeg_bytes_disguised_as_a_png_data_url_are_rejected():
    image = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), 255, dtype=np.uint8)
    success, jpeg = cv2.imencode(".jpg", image)
    assert success
    payload = _payload()
    payload["image"] = "data:image/png;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii")
    with pytest.raises(ValueError):
        decode_canvas_payload(payload)


def test_png_dimensions_must_match_the_protocol_metadata():
    wrong_size = np.full((CANVAS_HEIGHT, CANVAS_WIDTH - 1, 3), 255, dtype=np.uint8)
    with pytest.raises(ValueError):
        decode_canvas_payload(_payload(wrong_size))


def test_broken_png_with_valid_signature_is_reported_as_value_error():
    payload = _payload()
    payload["image"] = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\ninvalid").decode("ascii")
    with pytest.raises(ValueError):
        decode_canvas_payload(payload)


def test_decoded_payload_above_two_megabytes_is_rejected():
    payload = _payload()
    oversized = b"\x89PNG\r\n\x1a\n" + b"x" * (2 * 1024 * 1024)
    payload["image"] = "data:image/png;base64," + base64.b64encode(oversized).decode("ascii")
    with pytest.raises(ValueError):
        decode_canvas_payload(payload)
