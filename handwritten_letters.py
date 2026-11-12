"""Trainable isolated-letter recognizer; independent of the TrOCR line reader."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import os
import string
import time

import cv2
import numpy as np

from handwriting import prepare_handwriting_image


ALPHABET = string.ascii_lowercase
DEFAULT_LETTER_MODEL_DIR = Path(__file__).resolve().parent / "models" / "handwriting_letters"
PERSONAL_LETTER_MODEL_DIR = Path(__file__).resolve().parent / "models" / "handwriting_letters_personal"


class LetterSetupError(RuntimeError):
    """The locally trained letter model is not ready."""


@dataclass(frozen=True)
class LetterPrediction:
    text: str
    alternatives: tuple[str, ...]
    processing_time_ms: float


def preprocess_letter(image: str | Path | np.ndarray) -> np.ndarray:
    """Crop one dark glyph, fit its longest side to 20, and center on black 28×28.

    Output foreground is white. Keep every component so dotted i/j retain their
    detached marks. Input is one letter, never an entire cursive word.
    """
    rgb = prepare_handwriting_image(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if not np.any(mask):
        raise ValueError("Không tìm thấy nét chữ. Hãy viết một chữ cái a–z.")
    x, y, width, height = cv2.boundingRect(mask)
    scale = 20.0 / max(width, height)
    resized = cv2.resize(mask[y:y + height, x:x + width],
                         (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    normalized = np.zeros((28, 28), dtype=np.uint8)
    top, left = (28 - resized.shape[0]) // 2, (28 - resized.shape[1]) // 2
    normalized[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return normalized


def create_letter_model():
    """Build a small two-convolution CNN; importing this module needs no Torch."""
    import torch.nn as nn

    class LetterCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 26),
            )

        def forward(self, image):
            return self.classifier(self.features(image))

    return LetterCNN()


def letter_model_ready(model_dir: str | Path = DEFAULT_LETTER_MODEL_DIR) -> bool:
    folder = Path(model_dir)
    return all((folder / name).is_file() and (folder / name).stat().st_size > 0
               for name in ("model.pt", "metadata.json"))


def active_letter_model_dir() -> Path:
    """Prefer a completed personalized checkpoint while preserving the baseline."""
    if letter_model_ready(PERSONAL_LETTER_MODEL_DIR):
        try:
            meta = json.loads((PERSONAL_LETTER_MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
            if meta.get("personalized") is True:
                return PERSONAL_LETTER_MODEL_DIR
        except (OSError, ValueError):
            pass
    return DEFAULT_LETTER_MODEL_DIR


class LetterRecognizer:
    """CPU reader for a trained EMNIST Letters CNN; no language-model decoding."""

    def __init__(self, model_dir: str | Path = DEFAULT_LETTER_MODEL_DIR):
        if not letter_model_ready(model_dir):
            raise LetterSetupError("Chưa có mô hình một ký tự. Chạy scripts/train_handwriting_letters.py trước.")
        try:
            import torch
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
            meta = json.loads((Path(model_dir) / "metadata.json").read_text(encoding="utf-8"))
            if (meta.get("architecture") != "LetterCNN-v1" or meta.get("alphabet") != ALPHABET
                    or meta.get("input_size") != 28 or meta.get("foreground") != "white"
                    or meta.get("normalize") != "crop-longest20-center28"):
                raise ValueError("Thông tin mô hình không khớp bảng chữ cái hoặc kiến trúc.")
            self.model = create_letter_model()
            self.model.load_state_dict(torch.load(Path(model_dir) / "model.pt", map_location="cpu", weights_only=True))
            self.model.eval()
            self.torch = torch
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            raise LetterSetupError(f"Không nạp được mô hình một ký tự: {exc}") from exc

    def predict(self, normalized: np.ndarray) -> LetterPrediction:
        started = time.perf_counter()
        pixels = self.torch.from_numpy(normalized.astype(np.float32) / 255.0)[None, None]
        with self.torch.inference_mode():
            logits = self.model(pixels)
            indices = logits[0].topk(3).indices.tolist()
        letters = tuple(ALPHABET[index] for index in indices)
        return LetterPrediction(letters[0], letters, (time.perf_counter() - started) * 1000.0)


@lru_cache(maxsize=1)
def get_letter_recognizer(model_dir: str | Path = DEFAULT_LETTER_MODEL_DIR) -> LetterRecognizer:
    return LetterRecognizer(model_dir)


def recognize_letter(image: str | Path | np.ndarray,
                      model_dir: str | Path | None = None) -> LetterPrediction:
    """Recognize one glyph, returning three candidate letters for user review."""
    normalized = preprocess_letter(image)
    return get_letter_recognizer(model_dir if model_dir is not None else active_letter_model_dir()).predict(normalized)
