"""Generate labeled synthetic black-dot-on-white Braille images.

Generated images are useful for software development and controlled experiments;
they do not represent the appearance of real embossed Braille paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow `python scripts/generate_samples.py` from any working directory.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from braille_renderer import render_braille


DEFAULT_TEXTS = ["abc", "hello", "world", "braille", "python", "vision", "sample", "dots",
                 "matrix", "recognize", "pattern", "computer", "student", "project", "letter",
                 "simple", "camera", "image", "science", "linear"]


def render_text(text: str, *, dot_spacing: int = 18, radius: int = 6,
                margin: int = 24, angle: float = 0, brightness: int = 0,
                noise_std: float = 0, seed: int = 0) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Render lowercase letters as filled black dots, with optional image variation."""
    text = text.lower()
    image, centers = render_braille(text, dot_spacing=dot_spacing, radius=radius, margin=margin)
    height, width = image.shape
    if angle:
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        mapped = cv2.transform(np.asarray(centers, dtype=np.float32).reshape(1, -1, 2), matrix)[0]
        centers = [(float(x), float(y)) for x, y in mapped]
    if brightness:
        image = np.clip(image.astype(np.int16) + int(brightness), 0, 255).astype(np.uint8)
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, noise_std, image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image, centers


def generate_dataset(output_dir: Path, texts: list[str] | None = None,
                     seed: int = 42, variations_per_text: int = 4) -> Path:
    """Create grouped development/test data and a CSV manifest without group leakage."""
    texts = texts or DEFAULT_TEXTS
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    # Split is made at source text/group level. All augmented versions of a source
    # sequence therefore stay together in development or test.
    test_ids = {text for index, text in enumerate(texts) if index % 5 == 0}
    for group_index, text in enumerate(texts):
        split = "test" if text in test_ids else "development"
        for variant in range(variations_per_text):
            angle = float(rng.uniform(-1.5, 1.5)) if variant else 0.0
            brightness = int(rng.integers(-18, 19)) if variant else 0
            noise_std = float(rng.choice([0, 2, 5, 8])) if variant else 0.0
            image, centers = render_text(text, angle=angle, brightness=brightness,
                                         noise_std=noise_std, seed=seed + group_index * 100 + variant)
            filename = f"{group_index:03d}_{text}_{variant:02d}.png"
            folder = output_dir / split
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / filename), image)
            rows.append({
                "path": str(Path(split) / filename), "text": text, "source_id": text,
                "split": split, "variant": variant, "angle_degrees": angle,
                "brightness_offset": brightness, "noise_std": noise_std,
                "dot_centers_json": json.dumps(centers),
            })
    manifest = output_dir / "labels.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variations", type=int, default=4)
    parser.add_argument("--texts", nargs="*", default=None)
    args = parser.parse_args()
    manifest = generate_dataset(args.output, args.texts, args.seed, args.variations)
    print(f"Generated labeled synthetic data: {manifest}")


if __name__ == "__main__":
    main()
