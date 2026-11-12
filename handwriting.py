"""Optional local TrOCR reader for one word or one line of Latin handwriting.

Torch/Transformers are imported only when the reader is explicitly requested.
Model weights are downloaded by a separate setup script, never on image upload.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os
import time

import cv2
import numpy as np


MODEL_ID = "microsoft/trocr-small-handwritten"
MODEL_REVISION = "b4648cfa171985a6745f37ddd637e98c0da958ac"
ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "models" / "trocr-small-handwritten"
MODEL_FILES = ("config.json", "preprocessor_config.json", "pytorch_model.bin",
               "sentencepiece.bpe.model", "tokenizer_config.json", "special_tokens_map.json")


class HandwritingSetupError(RuntimeError):
    """The optional OCR dependencies or local model files are unavailable."""


@dataclass(frozen=True)
class HandwritingResult:
    """Raw model transcription, inference time, and output-limit status."""

    text: str
    processing_time_ms: float
    truncated: bool = False


@dataclass(frozen=True)
class HandwritingAlternatives:
    """Unverified model suggestions; ordering is not a calibrated probability."""

    texts: tuple[str, ...]
    processing_time_ms: float
    truncated: bool = False


def make_handwriting_variants(rgb_image: np.ndarray) -> list[np.ndarray]:
    """Keep the original and try two margin/stroke variants for optional rereading.

    Larger margins can help with tightly cropped mouse drawings; mild thickening
    is only an alternative because it may also join neighboring strokes.
    """
    if (not isinstance(rgb_image, np.ndarray) or rgb_image.dtype != np.uint8
            or rgb_image.ndim != 3 or rgb_image.shape[2] != 3 or rgb_image.size == 0):
        raise ValueError("Ảnh dùng để đọc kỹ cần là ảnh RGB 8 bit hợp lệ.")
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    if int(gray.max()) - int(gray.min()) < 5:
        raise ValueError("Ảnh trống hoặc không có nét chữ rõ.")
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if not np.any(mask):
        raise ValueError("Không tìm thấy vùng chữ để đọc kỹ.")
    x, y, width, height = cv2.boundingRect(mask)
    scale = min(96.0 / height, 4096.0 / width)
    resized = cv2.resize(rgb_image[y:y + height, x:x + width],
                         (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    thickened = cv2.erode(resized, np.ones((2, 2), dtype=np.uint8))
    variants = [rgb_image.copy()]
    for image in (resized, thickened):
        variants.append(cv2.copyMakeBorder(image, 48, 48, 48, 48,
                                           cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    return variants


def model_files_ready(model_dir: str | Path = DEFAULT_MODEL_DIR) -> bool:
    """Check presence of the pinned model's required local files."""
    folder = Path(model_dir)
    return all((folder / filename).is_file() and (folder / filename).stat().st_size > 0
               for filename in MODEL_FILES)


def prepare_handwriting_image(image: str | Path | np.ndarray) -> np.ndarray:
    """Load grayscale/BGR/BGRA uint8 input as RGB and reject blank/invalid crops."""
    if isinstance(image, (str, Path)):
        loaded = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        if loaded is None:
            raise ValueError("Không đọc được ảnh chữ viết tay. Hãy chọn JPG hoặc PNG hợp lệ.")
        image = loaded
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Ảnh đầu vào rỗng hoặc không hợp lệ.")
    if image.dtype != np.uint8:
        raise ValueError("Ảnh chữ viết tay cần ở định dạng 8 bit; hãy lưu lại thành JPG/PNG 8 bit.")
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        bgr = np.rint(image[:, :, :3] * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Ảnh cần có một kênh xám, ba kênh BGR hoặc bốn kênh BGRA.")
    height, width = rgb.shape[:2]
    if min(height, width) < 8:
        raise ValueError("Vùng chữ quá nhỏ. Hãy chọn ảnh có chiều rộng và cao ít nhất 8 pixel.")
    if height * width > 20_000_000:
        raise ValueError("Ảnh quá lớn. Hãy cắt sát một dòng chữ và giữ ảnh dưới 20 triệu pixel.")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if int(gray.max()) - int(gray.min()) < 5 or float(gray.std()) < 1.0:
        raise ValueError("Ảnh trống hoặc không có nét chữ rõ. Hãy chọn lại vùng chứa chữ.")
    return np.ascontiguousarray(rgb)


class HandwritingRecognizer:
    """CPU inference using Microsoft's pretrained single-line handwriting reader."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR):
        if not model_files_ready(model_dir):
            raise HandwritingSetupError(
                "Chưa có mô hình chữ viết tay. Chạy scripts/download_handwriting_model.py trước.")
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise HandwritingSetupError(
                "Thiếu thư viện chữ viết tay. Cài requirements-handwriting.txt trong .venv.") from exc
        try:
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
            self.processor = TrOCRProcessor.from_pretrained(
                str(model_dir), local_files_only=True, use_fast=False)
            self.model = VisionEncoderDecoderModel.from_pretrained(
                str(model_dir), local_files_only=True).to("cpu").eval()
        except (OSError, ValueError, RuntimeError) as exc:
            raise HandwritingSetupError(
                f"Không nạp được mô hình cục bộ. Hãy kiểm tra thư viện và tải lại mô hình. Chi tiết: {exc}") from exc
        self.torch = torch

    def recognize(self, rgb_image: np.ndarray) -> HandwritingResult:
        """Transcribe an RGB image; keep punctuation/case for user review."""
        from PIL import Image

        started = time.perf_counter()
        pixels = self.processor(images=Image.fromarray(rgb_image), return_tensors="pt").pixel_values
        with self.torch.inference_mode():
            ids = self.model.generate(pixels, max_new_tokens=128, num_beams=1, do_sample=False)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        truncated = ids.shape[1] >= 129 and int(ids[0, -1]) not in eos_ids
        if not text:
            raise ValueError("Mô hình chưa đọc được chữ. Hãy thử ảnh rõ và cắt sát một dòng.")
        return HandwritingResult(text, (time.perf_counter() - started) * 1000.0, truncated)

    def recognize_alternatives(self, rgb_image: np.ndarray) -> HandwritingAlternatives:
        """Search five beams per image variant without forcing a known word."""
        from PIL import Image

        started = time.perf_counter()
        texts: list[str] = []
        seen: set[str] = set()
        truncated = False
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        for variant in make_handwriting_variants(rgb_image):
            pixels = self.processor(images=Image.fromarray(variant), return_tensors="pt").pixel_values
            with self.torch.inference_mode():
                ids = self.model.generate(pixels, max_new_tokens=128, num_beams=5,
                                          num_return_sequences=5, do_sample=False, use_cache=True)
            for row, text in zip(ids, self.processor.batch_decode(ids, skip_special_tokens=True)):
                text = text.strip()
                if len(row) >= 129 and int(row[-1]) not in eos_ids:
                    truncated = True
                if text and text not in seen:
                    seen.add(text)
                    texts.append(text)
        if not texts:
            raise ValueError("Chưa tìm được cách đọc khác. Hãy viết lại rõ hơn hoặc sửa văn bản.")
        return HandwritingAlternatives(tuple(texts), (time.perf_counter() - started) * 1000.0, truncated)


@lru_cache(maxsize=1)
def get_recognizer(model_dir: str | Path = DEFAULT_MODEL_DIR) -> HandwritingRecognizer:
    """Reuse the local CPU reader across repeated requests in one process."""
    return HandwritingRecognizer(model_dir)


def recognize_handwriting(image: str | Path | np.ndarray,
                          model_dir: str | Path = DEFAULT_MODEL_DIR) -> HandwritingResult:
    """Validate an image and transcribe it without normalizing the raw text."""
    rgb = prepare_handwriting_image(image)
    return get_recognizer(model_dir).recognize(rgb)


def recognize_handwriting_alternatives(image: str | Path | np.ndarray,
                                       model_dir: str | Path = DEFAULT_MODEL_DIR) -> HandwritingAlternatives:
    """Return raw suggestions for explicit user selection, never autocorrection."""
    rgb = prepare_handwriting_image(image)
    return get_recognizer(model_dir).recognize_alternatives(rgb)
