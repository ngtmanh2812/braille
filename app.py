"""Streamlit interface for image recognition and Latin-to-Braille conversion."""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from pipeline import PipelineConfig, run_pipeline
from preprocessing import PreprocessConfig
from braille_renderer import render_braille
from decoder import encode_text
from handwriting_ui import render_handwriting_tab


st.set_page_config(page_title="Braille Recognition", page_icon="⠃", layout="wide")
st.title("Braille Recognition")
st.caption("Nhận diện ảnh Braille, chuyển chữ Latin sang Braille hoặc đọc một dòng chữ viết tay.")

recognition_tab, conversion_tab, handwriting_tab = st.tabs([
    "Ảnh Braille → Chữ", "Chữ → Braille", "Chữ viết tay → Braille",
])

with recognition_tab:
    uploaded = st.file_uploader("Upload a JPG or PNG image", type=["jpg", "jpeg", "png"])
    method = st.selectbox("Threshold method", ["otsu", "global", "adaptive"], index=0)
    threshold_value = 127
    if method == "global":
        threshold_value = st.slider("Global threshold", 0, 255, 127)
    denoise = st.checkbox("Gaussian blur denoising", value=True)
    if uploaded is not None:
        raw = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            st.error("The uploaded file could not be decoded as an image. Please try another JPG or PNG.")
        else:
            try:
                result = run_pipeline(image, PipelineConfig(preprocessing=PreprocessConfig(
                    threshold_method=method, threshold_value=threshold_value, denoise=denoise,
                )))
            except ValueError as exc:
                st.error(f"Image processing failed: {exc}")
            else:
                left, middle, right = st.columns(3)
                left.image(cv2.cvtColor(result.debug_images["original"], cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
                middle.image(result.debug_images["binary"], caption="Binary dots", use_container_width=True)
                right.image(cv2.cvtColor(result.debug_images["annotated"], cv2.COLOR_BGR2RGB), caption="Detected dots and cells", use_container_width=True)
                st.subheader("Predicted text")
                st.code(result.text or "(no Braille cells detected)", language=None)
                st.write(f"Unicode Braille: {result.braille or '—'}")
                st.write(f"Detected {len(result.dots)} dots in {len(result.cells)} cells · {result.processing_time_ms:.2f} ms")
                st.download_button("Download TXT", data=(result.text + "\n").encode("utf-8"),
                                   file_name="braille_prediction.txt", mime="text/plain")

with conversion_tab:
    st.subheader("Chuyển chữ Latin thành Braille")
    text = st.text_input("Nhập chữ cần chuyển", value="abc", max_chars=100, key="latin_text",
                         help="Hỗ trợ chữ thường a–z, tối đa 100 ký tự. Chưa hỗ trợ dấu cách, chữ hoa, số hoặc tiếng Việt có dấu.")
    if not text:
        st.info("Nhập một hoặc nhiều chữ cái thường, ví dụ: abc, hello, braille.")
    else:
        try:
            braille = encode_text(text)
            image, _ = render_braille(text)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.write("Kết quả Unicode Braille:")
            st.code(braille, language=None)
            st.image(image, caption=f"Ảnh Braille của: {text}")
            st.caption("Ảnh tổng hợp chấm đen trên nền trắng. Bạn có thể tải PNG rồi upload ở tab nhận diện ảnh để thử lại.")
            success, png = cv2.imencode(".png", image)
            if success:
                st.download_button("Tải ảnh PNG", data=png.tobytes(), file_name="braille.png", mime="image/png")
            else:
                st.error("Không thể tạo tệp PNG từ ảnh Braille.")
            st.download_button("Tải Braille TXT", data=(braille + "\n").encode("utf-8"),
                               file_name="braille_unicode.txt", mime="text/plain")

with handwriting_tab:
    render_handwriting_tab()
