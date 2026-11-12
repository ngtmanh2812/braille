"""Test canvas-to-OCR state changes without browser drawing or model weights."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import drawing_canvas
import handwriting


def _strokes() -> np.ndarray:
    image = np.full((260, 1000, 3), 255, dtype=np.uint8)
    cv2.line(image, (100, 70), (100, 170), (0, 0, 0), 7)
    cv2.line(image, (100, 120), (220, 120), (0, 0, 0), 7)
    return image


@pytest.fixture
def drawing_app(monkeypatch):
    holder = {"image": _strokes()}
    calls = []
    uploaded_image = np.full((60, 200, 3), 255, dtype=np.uint8)
    cv2.line(uploaded_image, (20, 10), (20, 50), (0, 0, 0), 3)
    success, png = cv2.imencode(".png", uploaded_image)
    assert success
    upload = SimpleNamespace(getvalue=lambda: png.tobytes())

    def fake_recognize(image):
        calls.append(image.copy())
        return handwriting.HandwritingResult("Hello world", 3.0, False)

    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    monkeypatch.setattr(drawing_canvas, "render_drawing_canvas", lambda *args, **kwargs: holder["image"])
    monkeypatch.setattr(handwriting, "model_files_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(handwriting, "recognize_handwriting", fake_recognize)
    app = AppTest.from_string(
        "from handwriting_ui import render_handwriting_tab\nrender_handwriting_tab()\n",
        default_timeout=10,
    ).run()
    app.radio(key="handwriting_input_mode").set_value("Viết trực tiếp").run()
    assert not app.exception
    return app, calls, holder


def _recognize_and_convert(app):
    app.button(key="handwriting_recognize").click().run()
    assert not app.exception
    app.button(key="handwriting_convert").click().run()
    assert not app.exception
    assert "handwriting_braille" in app.session_state


def test_opening_drawing_does_not_run_ocr_and_confirmed_text_can_export(drawing_app):
    app, calls, _ = drawing_app
    assert calls == []
    assert app.text_area(key="handwriting_editor").value == ""
    app.button(key="handwriting_recognize").click().run()
    assert not app.exception
    assert len(calls) == 1
    assert calls[0].shape[1] < drawing_canvas.CANVAS_WIDTH
    assert app.session_state["handwriting_ocr_raw"] == "Hello world"
    app.text_area(key="handwriting_editor").set_value("hello there").run()
    app.button(key="handwriting_convert").click().run()
    assert not app.exception
    assert app.session_state["handwriting_normalized"] == "hello there"
    assert "\u2800" in app.session_state["handwriting_braille"]
    assert app.session_state["handwriting_png"].startswith(b"\x89PNG\r\n\x1a\n")
    assert len(calls) == 1


def test_adding_a_stroke_invalidates_old_results_without_repeating_ocr(drawing_app):
    app, calls, holder = drawing_app
    _recognize_and_convert(app)
    changed = holder["image"].copy()
    cv2.line(changed, (350, 70), (350, 160), (0, 0, 0), 7)
    holder["image"] = changed
    app.run()
    assert not app.exception
    assert "handwriting_ocr_raw" not in app.session_state
    assert "handwriting_braille" not in app.session_state
    assert "handwriting_png" not in app.session_state
    assert app.text_area(key="handwriting_editor").value == ""
    assert len(calls) == 1


@pytest.mark.parametrize("cleared_value", [None, "white"])
def test_clearing_the_canvas_removes_outputs_and_recognition_button(drawing_app, cleared_value):
    app, calls, holder = drawing_app
    _recognize_and_convert(app)
    holder["image"] = (None if cleared_value is None else
                       np.full((260, 1000, 3), 255, dtype=np.uint8))
    app.run()
    assert not app.exception
    assert "handwriting_ocr_raw" not in app.session_state
    assert "handwriting_braille" not in app.session_state
    assert "handwriting_png" not in app.session_state
    assert not app.text_area
    assert ("handwriting_editor" not in app.session_state
            or app.session_state["handwriting_editor"] == "")
    assert not any(button.key == "handwriting_recognize" for button in app.button)
    assert len(calls) == 1


def test_switching_back_to_upload_clears_the_drawing_transcription(drawing_app):
    app, calls, _ = drawing_app
    _recognize_and_convert(app)
    app.radio(key="handwriting_input_mode").set_value("Tải ảnh").run()
    assert not app.exception
    assert "handwriting_ocr_raw" not in app.session_state
    assert "handwriting_braille" not in app.session_state
    assert app.text_area(key="handwriting_editor").value == ""
    assert len(calls) == 1


def test_cropping_drawing_removes_outer_margins_but_preserves_word_gap():
    image = np.full((260, 1000, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (100, 80), (120, 130), (0, 0, 0), -1)
    cv2.rectangle(image, (300, 80), (320, 130), (0, 0, 0), -1)
    cropped = drawing_canvas.crop_drawing_for_ocr(image, padding=24)
    assert cropped.shape == (99, 269, 3)
    assert np.all(cropped[:24] == 255)
    assert np.all(cropped[:, :24] == 255)
    assert np.all(cropped[:, 45:224] == 255)
    np.testing.assert_array_equal(cropped[24, 24], [0, 0, 0])
    np.testing.assert_array_equal(cropped[24, 224], [0, 0, 0])


def test_cropping_blank_canvas_fails_before_ocr():
    with pytest.raises(ValueError, match="trống"):
        drawing_canvas.crop_drawing_for_ocr(np.full((260, 1000, 3), 255, dtype=np.uint8))
