"""Local Streamlit drawing component and validated PNG-to-OpenCV bridge."""

from __future__ import annotations

import base64
import binascii
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


CANVAS_WIDTH = 1000
CANVAS_HEIGHT = 260
MAX_PNG_BYTES = 2 * 1024 * 1024
_PREFIX = "data:image/png;base64,"


def decode_canvas_payload(payload: object) -> np.ndarray | None:
    """Validate the browser message and decode an exact-size BGR canvas image."""
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("Dữ liệu bảng vẽ không hợp lệ. Hãy tải lại trang.")
    expected = {"version": 1, "width": CANVAS_WIDTH, "height": CANVAS_HEIGHT}
    if any(type(payload.get(key)) is not int or payload[key] != value
           for key, value in expected.items()):
        raise ValueError("Kích thước hoặc phiên bản bảng vẽ không hợp lệ.")
    count = payload.get("stroke_count")
    if type(count) is not int or count < 0:
        raise ValueError("Số nét vẽ không hợp lệ.")
    encoded = payload.get("image")
    if not isinstance(encoded, str) or not encoded.startswith(_PREFIX):
        raise ValueError("Bảng vẽ cần gửi ảnh PNG hợp lệ.")
    encoded = encoded[len(_PREFIX):]
    if len(encoded) > ((MAX_PNG_BYTES + 2) // 3) * 4:
        raise ValueError("Ảnh bảng vẽ quá lớn, tối đa 2 MiB.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Dữ liệu PNG của bảng vẽ bị lỗi.") from exc
    if len(raw) > MAX_PNG_BYTES or not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Định dạng hoặc dung lượng PNG của bảng vẽ không hợp lệ.")
    if (len(raw) < 24 or raw[12:16] != b"IHDR"
            or int.from_bytes(raw[16:20], "big") != CANVAS_WIDTH
            or int.from_bytes(raw[20:24], "big") != CANVAS_HEIGHT):
        raise ValueError("Ảnh bảng vẽ không đúng kích thước yêu cầu.")
    try:
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError("Không đọc được ảnh bảng vẽ.") from exc
    if image is None or image.shape != (CANVAS_HEIGHT, CANVAS_WIDTH, 3):
        raise ValueError("Ảnh bảng vẽ không đúng kích thước yêu cầu.")
    return image


def crop_drawing_for_ocr(image: np.ndarray, padding: int = 24) -> np.ndarray:
    """Crop unused white space, retaining a clean margin around black strokes."""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.shape != (CANVAS_HEIGHT, CANVAS_WIDTH, 3)):
        raise ValueError("Ảnh bảng vẽ không hợp lệ.")
    if type(padding) is not int or not 0 <= padding <= 100:
        raise ValueError("Lề ảnh bảng vẽ không hợp lệ.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = np.where(gray < 245, 255, 0).astype(np.uint8)
    if not np.any(ink):
        raise ValueError("Bảng vẽ đang trống. Hãy viết một từ hoặc một dòng chữ trước.")
    x, y, width, height = cv2.boundingRect(ink)
    return cv2.copyMakeBorder(image[y:y + height, x:x + width], padding, padding,
                              padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def drawing_touches_edge(image: np.ndarray, margin: int = 6) -> bool:
    """Flag ink near an edge; padding cannot restore strokes already clipped."""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.shape != (CANVAS_HEIGHT, CANVAS_WIDTH, 3)):
        raise ValueError("Ảnh bảng vẽ không hợp lệ.")
    if type(margin) is not int or not 1 <= margin <= CANVAS_HEIGHT // 2:
        raise ValueError("Khoảng kiểm tra viền không hợp lệ.")
    ink = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < 245
    return bool(np.any(ink[:margin]) or np.any(ink[-margin:])
                or np.any(ink[:, :margin]) or np.any(ink[:, -margin:]))


@lru_cache(maxsize=1)
def _component():
    """Declare the static component lazily so importing this module is cheap."""
    import streamlit.components.v1 as components

    return components.declare_component(
        "handwriting_canvas", path=str(Path(__file__).resolve().parent / "components" / "handwriting_canvas"))


def render_drawing_canvas(key: str = "handwriting_drawing") -> np.ndarray | None:
    """Show the drawing surface; return its latest pixels without running OCR."""
    import streamlit as st

    snapshot_key = "_snapshot_" + key
    saved = st.session_state.get(snapshot_key)
    if not isinstance(saved, dict):
        saved = None
    payload = _component()(initial_png=saved["image"] if saved else "",
                           initial_stroke_count=saved["stroke_count"] if saved else 0,
                           disabled=False, key=key, default=saved)
    image = decode_canvas_payload(payload)
    if image is not None:
        st.session_state[snapshot_key] = payload
    return image
