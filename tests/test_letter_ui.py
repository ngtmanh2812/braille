"""Check the isolated-letter UI route without model weights or training."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import handwriting
import handwritten_letters


@pytest.fixture
def letter_app(monkeypatch):
    image = np.full((100, 160, 3), 255, dtype=np.uint8)
    cv2.line(image, (45, 20), (45, 80), (0, 0, 0), 5)
    cv2.line(image, (45, 50), (75, 50), (0, 0, 0), 5)
    cv2.line(image, (75, 50), (75, 80), (0, 0, 0), 5)
    ok, png = cv2.imencode(".png", image)
    assert ok
    uploaded = SimpleNamespace(getvalue=lambda: png.tobytes())
    ready = {"value": True}
    calls = []

    def fake_letter(crop):
        calls.append(crop.copy())
        return handwritten_letters.LetterPrediction("h", ("h", "k", "b"), 2.0)

    def unexpected_line_reader(*args, **kwargs):
        pytest.fail("Isolated-letter mode must not call the TrOCR line reader.")

    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded)
    monkeypatch.setattr(handwriting, "model_files_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(handwriting, "recognize_handwriting", unexpected_line_reader)
    monkeypatch.setattr(handwriting, "recognize_handwriting_alternatives", unexpected_line_reader)
    monkeypatch.setattr(handwritten_letters, "letter_model_ready", lambda *args, **kwargs: ready["value"])
    monkeypatch.setattr(handwritten_letters, "recognize_letter", fake_letter)
    app = AppTest.from_string(
        "from handwriting_ui import render_handwriting_tab\nrender_handwriting_tab()\n",
        default_timeout=10,
    ).run()
    app.radio(key="handwriting_recognition_target").set_value("Một ký tự a–z").run()
    assert not app.exception
    return app, calls, ready


def test_letter_mode_only_predicts_after_button_and_exports_h_braille(letter_app):
    app, calls, _ = letter_app
    assert calls == []
    assert not any(button.key == "handwriting_reread" for button in app.button)
    app.button(key="handwriting_recognize").click().run()
    assert not app.exception
    assert len(calls) == 1
    assert app.session_state["handwriting_ocr_raw"] == "h"
    assert app.text_area(key="handwriting_editor").value == "h"
    assert app.session_state["handwriting_alternatives"] == ["h", "k", "b"]
    app.button(key="handwriting_convert").click().run()
    assert not app.exception
    assert app.session_state["handwriting_braille"] == "⠓"
    assert app.session_state["handwriting_png"].startswith(b"\x89PNG\r\n\x1a\n")
    assert len(calls) == 1


def test_letter_alternative_requires_explicit_apply(letter_app):
    app, calls, _ = letter_app
    app.button(key="handwriting_recognize").click().run()
    app.selectbox(key="handwriting_alternative_choice").set_value("k").run()
    assert not app.exception
    assert app.text_area(key="handwriting_editor").value == "h"
    app.button(key="handwriting_apply_alternative").click().run()
    assert not app.exception
    assert app.text_area(key="handwriting_editor").value == "k"
    assert app.session_state["handwriting_ocr_raw"] == "h"
    assert len(calls) == 1


def test_switching_to_line_mode_clears_letter_outputs(letter_app):
    app, calls, _ = letter_app
    app.button(key="handwriting_recognize").click().run()
    app.button(key="handwriting_convert").click().run()
    app.radio(key="handwriting_recognition_target").set_value("Một từ / dòng").run()
    assert not app.exception
    assert "handwriting_ocr_raw" not in app.session_state
    assert "handwriting_alternatives" not in app.session_state
    assert "handwriting_braille" not in app.session_state
    assert app.text_area(key="handwriting_editor").value == ""
    assert len(calls) == 1


def test_missing_letter_model_disables_prediction_and_shows_training_guide(letter_app):
    app, calls, ready = letter_app
    ready["value"] = False
    app.run()
    assert not app.exception
    assert app.button(key="handwriting_recognize").disabled
    assert any("một ký tự" in item.value for item in app.warning)
    assert any("train_handwriting_letters.py" in item.value for item in app.code)
    assert calls == []
