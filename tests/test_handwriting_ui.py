"""Exercise the review-before-Braille workflow without loading an OCR model."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import handwriting


@dataclass(frozen=True)
class _UploadedImage:
    content: bytes

    def getvalue(self) -> bytes:
        return self.content


@pytest.fixture
def handwriting_app(monkeypatch: pytest.MonkeyPatch) -> tuple[AppTest, list[np.ndarray], int]:
    """Provide a real PNG upload and a counted, deterministic OCR substitute."""
    width = 420
    image = np.full((90, width, 3), 255, dtype=np.uint8)
    cv2.putText(image, "Hello world", (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    upload = _UploadedImage(encoded.tobytes())
    calls: list[np.ndarray] = []

    def fake_recognize(crop: np.ndarray) -> handwriting.HandwritingResult:
        calls.append(crop.copy())
        return handwriting.HandwritingResult("Hello, world!", 12.5, False)

    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    monkeypatch.setattr(handwriting, "model_files_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(handwriting, "recognize_handwriting", fake_recognize)
    app = AppTest.from_string(
        "from handwriting_ui import render_handwriting_tab\nrender_handwriting_tab()\n",
        default_timeout=10,
    ).run()
    assert not app.exception
    return app, calls, width


def test_ocr_requires_button_and_review_before_braille(handwriting_app) -> None:
    app, calls, _ = handwriting_app
    assert calls == []  # Opening the tab and previewing the image never invokes OCR.
    assert app.text_area(key="handwriting_editor").value == ""

    app.button(key="handwriting_recognize").click().run()
    assert not app.exception
    assert len(calls) == 1
    assert app.session_state["handwriting_ocr_raw"] == "Hello, world!"
    assert app.text_area(key="handwriting_editor").value == "Hello, world!"
    assert app.code[0].value == "Hello, world!"

    # Unsupported punctuation remains visible and must be corrected by the user.
    app.button(key="handwriting_convert").click().run()
    assert not app.exception
    assert any("Ký tự chưa hỗ trợ" in element.value for element in app.error)
    assert "handwriting_braille" not in app.session_state
    assert app.session_state["handwriting_ocr_raw"] == "Hello, world!"

    app.text_area(key="handwriting_editor").set_value("Hello world").run()
    assert "handwriting_braille" not in app.session_state
    assert len(calls) == 1
    app.button(key="handwriting_convert").click().run()
    assert not app.exception
    assert not app.error
    assert app.session_state["handwriting_normalized"] == "hello world"
    braille = app.session_state["handwriting_braille"]
    assert "\u2800" in braille
    assert len(braille) == len("hello world")
    assert app.session_state["handwriting_png"].startswith(b"\x89PNG\r\n\x1a\n")
    assert app.code[0].value == "Hello, world!"  # The raw OCR is preserved.

    app.text_area(key="handwriting_editor").set_value("Hello there").run()
    assert not app.exception
    assert "handwriting_braille" not in app.session_state
    assert "handwriting_png" not in app.session_state
    assert all(element.value != braille for element in app.code)
    assert len(calls) == 1  # Editing text also never reruns the model.


def test_reviewed_text_with_apostrophe_comma_and_period_converts(handwriting_app) -> None:
    app, calls, _ = handwriting_app
    app.text_area(key="handwriting_editor").set_value("Don't, stop.").run()
    app.button(key="handwriting_convert").click().run()
    assert not app.exception
    assert not app.error
    assert app.session_state["handwriting_braille"] == "⠙⠕⠝⠄⠞⠂\u2800⠎⠞⠕⠏⠲"
    assert app.session_state["handwriting_png"].startswith(b"\x89PNG\r\n\x1a\n")
    assert calls == []


def test_reread_suggestions_require_explicit_selection_before_conversion(handwriting_app, monkeypatch):
    app, calls, _ = handwriting_app
    rereads = []

    def fake_alternatives(image):
        rereads.append(image)
        return handwriting.HandwritingAlternatives(("wrong reading", "manh", "manh ."), 50.0)

    monkeypatch.setattr(handwriting, "recognize_handwriting_alternatives", fake_alternatives)
    app.button(key="handwriting_recognize").click().run()
    original = app.session_state["handwriting_ocr_raw"]
    assert not rereads
    app.button(key="handwriting_reread").click().run()
    assert not app.exception
    assert len(rereads) == 1
    assert app.text_area(key="handwriting_editor").value == original
    app.selectbox(key="handwriting_alternative_choice").set_value("manh").run()
    assert app.text_area(key="handwriting_editor").value == original
    app.button(key="handwriting_apply_alternative").click().run()
    assert not app.exception
    assert app.text_area(key="handwriting_editor").value == "manh"
    assert app.session_state["handwriting_ocr_raw"] == original
    app.button(key="handwriting_convert").click().run()
    assert app.session_state["handwriting_braille"] == "⠍⠁⠝⠓"
    assert len(calls) == 1


def test_new_crop_clears_reread_suggestions(handwriting_app, monkeypatch):
    app, _, width = handwriting_app
    monkeypatch.setattr(handwriting, "recognize_handwriting_alternatives",
                        lambda image: handwriting.HandwritingAlternatives(("first", "second"), 1.0))
    app.button(key="handwriting_reread").click().run()
    assert app.selectbox
    app.checkbox[0].check().run()
    app.slider[0].set_value((0, width - 30)).run()
    assert not app.exception
    assert "handwriting_alternatives" not in app.session_state
    assert not app.selectbox


def test_changing_crop_invalidates_ocr_and_braille(handwriting_app) -> None:
    app, calls, width = handwriting_app
    app.button(key="handwriting_recognize").click().run()
    app.text_area(key="handwriting_editor").set_value("Hello world").run()
    app.button(key="handwriting_convert").click().run()
    assert "handwriting_braille" in app.session_state

    app.checkbox[0].check().run()
    assert not app.exception
    assert "handwriting_ocr_raw" in app.session_state  # The full-image ROI is unchanged.
    app.slider[0].set_value((0, width - 30)).run()
    assert not app.exception
    assert "handwriting_ocr_raw" not in app.session_state
    assert "handwriting_braille" not in app.session_state
    assert "handwriting_png" not in app.session_state
    assert app.text_area(key="handwriting_editor").value == ""
    assert len(calls) == 1
