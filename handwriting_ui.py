"""Streamlit workflow for one-line handwriting OCR followed by Braille rendering.

Importing this UI module does not load OCR weights or machine-learning libraries.
The handwriting backend is responsible for keeping its own imports lightweight;
inference is requested only after the user presses the recognition button.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np
import streamlit as st


_PREFIX = "handwriting_"


def _clear_conversion() -> None:
    """Remove output that no longer matches the current editable text."""
    for key in ("converted_source", "normalized", "braille", "braille_image", "png"):
        st.session_state.pop(_PREFIX + key, None)


def _clear_recognition() -> None:
    """Invalidate recognition and conversion after the image or crop changes."""
    for key in ("ocr_raw", "elapsed_ms", "truncated", "alternatives", "alternatives_ms",
                "alternative_choice", "alternatives_truncated"):
        st.session_state.pop(_PREFIX + key, None)
    st.session_state[_PREFIX + "editor"] = ""
    _clear_conversion()


def _show_setup_guide(*, expanded: bool = False) -> None:
    """Show the optional CPU setup steps without running any installation."""
    with st.expander("Cài đặt chức năng nhận diện chữ viết tay", expanded=expanded):
        st.write("Chạy lần lượt các lệnh sau trong Terminal tại thư mục dự án. Cần kết nối Internet để tải thư viện và mô hình.")
        st.code(
            r""".\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-handwriting.txt
.\.venv\Scripts\python.exe scripts\download_handwriting_model.py""",
            language="powershell",
        )
        st.caption("Sau khi tải xong mô hình, tải lại trang để kiểm tra trạng thái.")


def _show_letter_setup_guide(*, expanded: bool = False) -> None:
    """Explain how to prepare and train the isolated-letter model explicitly."""
    with st.expander("Chuẩn bị mô hình một ký tự", expanded=expanded):
        st.write("Tải dữ liệu EMNIST Letters rồi huấn luyện mô hình một lần trong Terminal. Cần thư viện chữ viết tay đã cài; ứng dụng không tự tải dữ liệu hoặc chạy huấn luyện khi mở tab.")
        st.code(
            r""".\.venv\Scripts\python.exe scripts\download_emnist_letters.py
.\.venv\Scripts\python.exe scripts\train_handwriting_letters.py""",
            language="powershell",
        )
        st.caption("Sau khi huấn luyện hoàn tất, tải lại trang để nạp mô hình đã lưu.")


def _uploaded_input() -> tuple[np.ndarray, str] | None:
    """Get an uploaded image/crop while leaving OCR decisions to the caller."""
    from handwriting import prepare_handwriting_image

    uploaded = st.file_uploader(
        "Ảnh một từ hoặc một dòng chữ viết tay (JPG/PNG)",
        type=["jpg", "jpeg", "png"], key=_PREFIX + "upload")
    if uploaded is None:
        st.info("Chọn ảnh chữ viết tay rõ nét, nền sáng và chữ tối để bắt đầu.")
        return None
    contents = uploaded.getvalue()
    file_hash = hashlib.sha256(contents).hexdigest()[:16]
    try:
        image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_UNCHANGED) if contents else None
        if image is None:
            raise ValueError("Không đọc được ảnh. Vui lòng chọn một tệp JPG hoặc PNG hợp lệ.")
        image = cv2.cvtColor(prepare_handwriting_image(image), cv2.COLOR_RGB2BGR)
    except (ValueError, cv2.error) as exc:
        st.error(str(exc))
        return None

    height, width = image.shape[:2]
    crop_enabled = st.checkbox("Cắt vùng chứa dòng chữ", key=f"{_PREFIX}crop_{file_hash}")
    left, right, top, bottom = 0, width, 0, height
    if crop_enabled:
        st.caption("Điều chỉnh vùng cắt để chỉ giữ một dòng chữ, có đủ lề quanh nét chữ.")
        left, right = st.slider("Giới hạn ngang (pixel)", 0, width, (0, width),
                                key=f"{_PREFIX}crop_x_{file_hash}")
        top, bottom = st.slider("Giới hạn dọc (pixel)", 0, height, (0, height),
                                key=f"{_PREFIX}crop_y_{file_hash}")
    if right <= left or bottom <= top:
        st.error("Vùng cắt đang rỗng. Hãy tăng khoảng giới hạn ngang và dọc.")
        return None

    roi = image[top:bottom, left:right].copy()
    if crop_enabled:
        original_column, crop_column = st.columns(2)
        original_column.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", use_container_width=True)
        crop_column.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), caption="Vùng chữ dùng để nhận diện", use_container_width=True)
    else:
        st.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), caption="Ảnh chữ viết tay", use_container_width=True)
    return roi, f"upload:{file_hash}:{left}:{top}:{right}:{bottom}"


def _drawing_input() -> tuple[np.ndarray, str] | None:
    """Receive the canvas pixels, retaining word spacing when cropping margins."""
    from drawing_canvas import crop_drawing_for_ocr, drawing_touches_edge, render_drawing_canvas

    st.caption("Dùng chuột hoặc bút cảm ứng viết một từ hoặc một dòng. Viết chữ lớn, giữ khoảng cách giữa các từ và thả chuột sau mỗi nét.")
    try:
        image = render_drawing_canvas(key="handwriting_drawing")
        if image is None:
            st.info("Bảng vẽ đang trống. Hãy viết lên bảng để bắt đầu.")
            return None
        roi = crop_drawing_for_ocr(image)
    except ValueError as exc:
        st.info(str(exc))
        return None
    if drawing_touches_edge(image):
        st.warning("Một số nét chạm sát viền bảng và có thể bị cắt. Hãy viết cách mép một chút; thêm lề sau đó không khôi phục được nét đã mất.")
    signature = "drawing:" + hashlib.sha256(image.tobytes()).hexdigest()
    encoded, png = cv2.imencode(".png", image)
    if encoded:
        st.download_button("Tải ảnh nét vẽ PNG", data=png.tobytes(), file_name="net_viet_tay.png",
                           mime="image/png", key=_PREFIX + "download_drawing")
    return roi, signature


def render_handwriting_tab() -> None:
    """Render upload, optional cropping, explicit OCR, review, and Braille export."""
    # These modules expose only lightweight configuration/helpers at import time.
    # No call to the OCR backend is made until the recognition button is pressed.
    from braille_renderer import render_braille_line
    from handwriting import (
        DEFAULT_MODEL_DIR,
        HandwritingSetupError,
        model_files_ready,
        recognize_handwriting,
        recognize_handwriting_alternatives,
    )
    from latin_line import encode_latin_line, normalize_latin_line
    from handwritten_letters import (
        DEFAULT_LETTER_MODEL_DIR,
        LetterSetupError,
        active_letter_model_dir,
        letter_model_ready,
        recognize_letter,
    )

    st.subheader("Chữ viết tay → Braille")
    st.write("Tải ảnh hoặc viết trực tiếp một chữ cái, một từ hay một dòng chữ Latin. Sau khi nhận diện, kiểm tra và sửa văn bản trước khi tạo Braille.")
    st.caption("Hỗ trợ chữ Latin, khoảng trắng và dấu phẩy (,), chấm (.), apostrophe ('). Chữ hoa chuyển thành chữ thường; chưa có dấu báo chữ hoa, số, dấu câu khác hoặc tiếng Việt có dấu.")

    recognition_target = st.radio("Nội dung cần đọc", ["Một từ / dòng", "Một ký tự a–z"],
                                   horizontal=True, key=_PREFIX + "recognition_target")
    is_letter = recognition_target == "Một ký tự a–z"
    if is_letter:
        st.caption("Chọn chế độ này khi viết riêng một chữ như h. Mô hình ký tự được huấn luyện trên EMNIST Letters; không dùng chế độ này để đọc cả từ viết nối nét.")
        if active_letter_model_dir() != DEFAULT_LETTER_MODEL_DIR:
            st.caption("Đang dùng bản học bổ sung từ các mẫu đã gán nhãn; mô hình EMNIST ban đầu vẫn được giữ riêng để đối chiếu.")

    try:
        ready = (letter_model_ready(active_letter_model_dir()) if is_letter
                 else model_files_ready(DEFAULT_MODEL_DIR))
    except OSError:
        ready = False
    if not ready:
        if is_letter:
            st.warning("Chưa có mô hình một ký tự đã huấn luyện. Hãy chuẩn bị mô hình theo hướng dẫn bên dưới.")
            _show_letter_setup_guide()
        else:
            st.warning("Chưa có đủ tệp mô hình OCR. Hãy làm theo hướng dẫn cài đặt bên dưới để bật nhận diện chữ viết tay.")
            _show_setup_guide()

    input_mode = st.radio("Cách nhập chữ viết tay", ["Tải ảnh", "Viết trực tiếp"],
                          horizontal=True, key=_PREFIX + "input_mode")
    selected = _drawing_input() if input_mode == "Viết trực tiếp" else _uploaded_input()
    signature = f"{recognition_target}:{selected[1]}" if selected is not None else None
    if st.session_state.get(_PREFIX + "signature") != signature:
        _clear_recognition()
        st.session_state[_PREFIX + "signature"] = signature
    if selected is None:
        return
    roi, _ = selected

    if st.button("Nhận diện chữ viết tay", key=_PREFIX + "recognize", disabled=not ready):
        _clear_recognition()
        try:
            with st.spinner("Đang nhận diện chữ viết tay. Lần đầu có thể mất thêm thời gian để nạp mô hình…"):
                result = recognize_letter(roi) if is_letter else recognize_handwriting(roi)
            st.session_state[_PREFIX + "ocr_raw"] = result.text
            st.session_state[_PREFIX + "elapsed_ms"] = result.processing_time_ms
            st.session_state[_PREFIX + "truncated"] = False if is_letter else result.truncated
            st.session_state[_PREFIX + "editor"] = result.text
            if is_letter:
                st.session_state[_PREFIX + "alternatives"] = list(result.alternatives)
                st.session_state[_PREFIX + "alternatives_ms"] = result.processing_time_ms
                st.session_state[_PREFIX + "alternatives_truncated"] = False
        except (HandwritingSetupError, LetterSetupError) as exc:
            st.error(f"Chưa thể chạy OCR: {exc}")
            if is_letter:
                _show_letter_setup_guide(expanded=True)
            else:
                _show_setup_guide(expanded=True)
        except (ValueError, RuntimeError, cv2.error) as exc:
            st.error(f"Không nhận diện được vùng ảnh này: {exc}")

    if not is_letter and st.button("Đọc kỹ / xem gợi ý", key=_PREFIX + "reread", disabled=not ready):
        for key in ("alternatives", "alternatives_ms", "alternative_choice", "alternatives_truncated"):
            st.session_state.pop(_PREFIX + key, None)
        try:
            with st.spinner("Đang thử thêm cách cắt lề, độ dày nét và nhiều phương án đọc…"):
                alternatives = recognize_handwriting_alternatives(roi)
            st.session_state[_PREFIX + "alternatives"] = list(alternatives.texts)
            st.session_state[_PREFIX + "alternatives_ms"] = alternatives.processing_time_ms
            st.session_state[_PREFIX + "alternatives_truncated"] = alternatives.truncated
        except HandwritingSetupError as exc:
            st.error(f"Chưa thể chạy OCR: {exc}")
            _show_setup_guide(expanded=True)
        except (ValueError, RuntimeError, cv2.error) as exc:
            st.error(f"Chưa đọc được cách khác: {exc}")

    if st.session_state.get(_PREFIX + "alternatives"):
        st.caption("Đây là các gợi ý chưa được xác nhận. Bạn chọn cách đúng hoặc sửa trực tiếp; mô hình không tự sửa theo tên hay từ điển.")
        choice = st.selectbox("Các cách đọc gợi ý", st.session_state[_PREFIX + "alternatives"],
                              index=None, placeholder="Chọn cách đọc phù hợp…", key=_PREFIX + "alternative_choice")
        if st.button("Dùng cách đọc này", key=_PREFIX + "apply_alternative", disabled=choice is None):
            st.session_state[_PREFIX + "editor"] = choice
            _clear_conversion()
        st.caption(f"Thời gian tạo gợi ý: {st.session_state[_PREFIX + 'alternatives_ms']:.2f} ms")
        if st.session_state.get(_PREFIX + "alternatives_truncated"):
            st.warning("Một số gợi ý có thể bị cắt ở cuối. Hãy kiểm tra lại văn bản.")

    if _PREFIX + "ocr_raw" in st.session_state:
        raw_text = st.session_state[_PREFIX + "ocr_raw"]
        st.write("Văn bản OCR gốc")
        if raw_text.strip():
            st.code(raw_text, language=None)
        else:
            st.warning("OCR chưa tìm được văn bản. Bạn có thể cắt lại vùng chữ hoặc nhập văn bản đã đọc được bên dưới.")
        st.caption(f"Thời gian OCR: {st.session_state[_PREFIX + 'elapsed_ms']:.2f} ms")
        if st.session_state.get(_PREFIX + "truncated", False):
            st.warning("Kết quả OCR đã chạm giới hạn độ dài và có thể bị cắt. Hãy kiểm tra cuối dòng hoặc chọn ảnh chứa ít chữ hơn.")

    st.session_state.setdefault(_PREFIX + "editor", "")
    edited_text = st.text_area(
        "Kiểm tra hoặc sửa văn bản trước khi tạo Braille",
        key=_PREFIX + "editor",
        height=100,
        help="Một dòng chữ Latin, dấu cách và các dấu , . '. Dấu ' được hiểu là apostrophe trong từ, chưa xử lý dấu ngoặc kép/nháy theo ngữ cảnh. Tối đa 200 ký tự.",
    )
    if st.session_state.get(_PREFIX + "converted_source") != edited_text:
        _clear_conversion()

    if is_letter:
        with st.expander("Bổ sung mẫu chữ của bạn để huấn luyện"):
            st.write("Sửa ô văn bản phía trên thành đúng một chữ cái rồi lưu để xác nhận nhãn. Mẫu chỉ được thêm khi bạn nhấn nút.")
            confirmed_label = edited_text.strip()
            valid_label = len(confirmed_label) == 1 and confirmed_label in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if valid_label:
                st.caption(f"Nhãn sẽ lưu: {confirmed_label.lower()}")
            if st.button("Lưu mẫu chữ đã sửa", key=_PREFIX + "save_letter_sample", disabled=not valid_label):
                from letter_samples import save_letter_sample

                try:
                    save_letter_sample(roi, confirmed_label)
                    st.success("Đã lưu mẫu có nhãn. Mô hình chưa thay đổi; chạy lệnh học bổ sung bên dưới khi đã thu thập đủ mẫu.")
                except (OSError, ValueError) as exc:
                    st.error(f"Không lưu được mẫu: {exc}")
            st.code(r".\.venv\Scripts\python.exe scripts\adapt_handwriting_letters.py", language="powershell")
            st.caption("Nên thu thập nhiều cách viết cho mỗi chữ. Mẫu dùng học bổ sung không được coi là ảnh test độc lập.")

    if st.button("Xác nhận văn bản và tạo Braille", key=_PREFIX + "convert"):
        _clear_conversion()
        try:
            normalized = normalize_latin_line(edited_text)
            braille = encode_latin_line(normalized)
            braille_image, _ = render_braille_line(normalized)
            encoded, png = cv2.imencode(".png", braille_image)
            if not encoded:
                raise ValueError("Không thể tạo tệp ảnh PNG.")
        except (ValueError, cv2.error) as exc:
            st.error(f"Không thể tạo Braille: {exc}")
        else:
            st.session_state[_PREFIX + "converted_source"] = edited_text
            st.session_state[_PREFIX + "normalized"] = normalized
            st.session_state[_PREFIX + "braille"] = braille
            st.session_state[_PREFIX + "braille_image"] = braille_image
            st.session_state[_PREFIX + "png"] = png.tobytes()

    if _PREFIX + "braille" not in st.session_state:
        return

    normalized = st.session_state[_PREFIX + "normalized"]
    braille = st.session_state[_PREFIX + "braille"]
    st.write("Văn bản dùng để tạo Braille")
    st.code(normalized, language=None)
    st.caption("Đã chuyển chữ hoa thành chữ thường và gộp khoảng trắng liên tiếp. Khoảng trắng giữa các từ được giữ thành ô Braille trống; chưa có dấu báo chữ hoa.")
    st.write("Unicode Braille")
    st.code(braille, language=None)
    st.image(st.session_state[_PREFIX + "braille_image"], caption="Ảnh Braille chấm đen trên nền trắng", use_container_width=True)
    png_column, txt_column = st.columns(2)
    png_column.download_button(
        "Tải ảnh Braille PNG", data=st.session_state[_PREFIX + "png"],
        file_name="handwriting_braille.png", mime="image/png", key=_PREFIX + "download_png",
    )
    txt_column.download_button(
        "Tải TXT (văn bản và Braille)",
        data=f"Văn bản: {normalized}\nBraille: {braille}\n".encode("utf-8"),
        file_name="handwriting_braille.txt", mime="text/plain", key=_PREFIX + "download_txt",
    )
