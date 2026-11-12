"""Store explicitly confirmed letter samples for later supervised adaptation."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import string
import threading

import cv2
import numpy as np

from handwriting import prepare_handwriting_image
from handwritten_letters import preprocess_letter


DEFAULT_SAMPLE_DIR = Path(__file__).resolve().parent / "data" / "handwriting" / "user_samples"
_WRITE_LOCK = threading.Lock()


def plan_personal_repeats(unique_samples: int, requested: int = 0) -> int:
    """Bound personal replay to 4,000 entries while accepting growing datasets."""
    if type(unique_samples) is not int or not 1 <= unique_samples <= 4000:
        raise ValueError("Cần từ 1 đến 4.000 mẫu riêng biệt cho một lần học bổ sung.")
    if type(requested) is not int or requested < 0:
        raise ValueError("Số lần lấy mẫu phải không âm; 0 để chọn tự động.")
    repeats = requested or min(480, 4000 // unique_samples)
    if repeats * unique_samples > 4000:
        raise ValueError("Tổng lượt mẫu cá nhân vượt 4.000. Giảm --repeats hoặc dùng 0 để chọn tự động.")
    return repeats


def save_letter_sample(image: str | Path | np.ndarray, label: str,
                        sample_dir: str | Path = DEFAULT_SAMPLE_DIR) -> Path:
    """Append one confirmed a-z sample; saving never trains or changes a model.

    Duplicate normalized images are deduplicated. Conflicting labels are rejected
    so a previous confirmed label cannot silently be overwritten.
    """
    if not isinstance(label, str) or len(label) != 1 or label not in string.ascii_letters:
        raise ValueError("Nhãn mẫu cần là đúng một chữ cái a–z.")
    label = label.lower()
    normalized = preprocess_letter(image)
    rgb = prepare_handwriting_image(image)
    digest = hashlib.sha256(normalized.tobytes()).hexdigest()
    relative = f"captured/{digest}.png"
    root = Path(sample_dir)
    manifest = root / "train_manifest.csv"
    with _WRITE_LOCK:
        rows = []
        fields = ["path", "label"]
        if manifest.exists():
            with manifest.open(encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                fields = reader.fieldnames or []
                if not {"path", "label"}.issubset(fields):
                    raise ValueError("Danh sách mẫu cần có cột path và label.")
                rows = list(reader)
            for row in rows:
                if row["path"] == relative:
                    if row["label"] != label:
                        raise ValueError("Mẫu này đã được gán nhãn khác. Hãy kiểm tra train_manifest.csv trước.")
                    if (root / relative).is_file():
                        return root / relative
        root.mkdir(parents=True, exist_ok=True)
        (root / "captured").mkdir(exist_ok=True)
        success, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("Không thể lưu ảnh mẫu chữ.")
        image_path = root / relative
        temporary_image = image_path.with_suffix(".png.tmp")
        temporary_image.write_bytes(png.tobytes())
        temporary_image.replace(image_path)
        if not any(row["path"] == relative for row in rows):
            rows.append({**{field: "" for field in fields}, "path": relative, "label": label})
        temporary_manifest = manifest.with_suffix(".csv.tmp")
        with temporary_manifest.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary_manifest.replace(manifest)
    return image_path
