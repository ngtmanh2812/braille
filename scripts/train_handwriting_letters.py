"""Train a small CPU CNN on EMNIST Letters with a held-out official test split."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
import string
import struct
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from handwritten_letters import create_letter_model, preprocess_letter


DEFAULT_DATA = ROOT / "data" / "handwriting" / "emnist_letters"
DEFAULT_OUTPUT = ROOT / "models" / "handwriting_letters"
DEFAULT_REPORT = ROOT / "outputs" / "handwriting_letters_training"
ALPHABET = string.ascii_lowercase


def load_idx(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Read strict EMNIST Letters gzip/IDX data and map labels to 0..25."""
    expected = {"train": 124800, "test": 20800}[split]
    image_path = data_dir / f"emnist-letters-{split}-images-idx3-ubyte.gz"
    label_path = data_dir / f"emnist-letters-{split}-labels-idx1-ubyte.gz"
    with gzip.open(image_path, "rb") as file:
        image_bytes = file.read()
    with gzip.open(label_path, "rb") as file:
        label_bytes = file.read()
    if len(image_bytes) < 16 or len(label_bytes) < 8:
        raise ValueError(f"Truncated IDX file in {split}.")
    image_magic, count, rows, columns = struct.unpack(">IIII", image_bytes[:16])
    label_magic, label_count = struct.unpack(">II", label_bytes[:8])
    if (image_magic, count, rows, columns) != (2051, expected, 28, 28):
        raise ValueError(f"Unexpected image header in {split}: {(image_magic, count, rows, columns)}")
    if (label_magic, label_count) != (2049, expected):
        raise ValueError(f"Unexpected label header in {split}: {(label_magic, label_count)}")
    if len(image_bytes) != 16 + expected * 28 * 28 or len(label_bytes) != 8 + expected:
        raise ValueError(f"IDX data length does not match the header for {split}.")
    images = np.frombuffer(image_bytes, dtype=np.uint8, offset=16).reshape(expected, 28, 28)
    labels = np.frombuffer(label_bytes, dtype=np.uint8, offset=8).astype(np.int64) - 1
    if set(np.unique(labels)) != set(range(26)):
        raise ValueError(f"Expected all labels 1..26 in {split}.")
    counts = np.bincount(labels, minlength=26)
    if not np.all(counts == expected // 26):
        raise ValueError(f"EMNIST Letters {split} is not class balanced as expected.")
    return images, labels


def normalize_images(raw: np.ndarray, description: str) -> np.ndarray:
    """Correct IDX orientation and apply exactly the inference preprocessing."""
    normalized = np.empty_like(raw)
    for index, image in enumerate(raw):
        # EMNIST's stored orientation is transposed relative to normal reading.
        # Raw IDX has white ink/black paper; the public preprocessing accepts dark ink.
        normalized[index] = preprocess_letter(255 - image.T)
        if (index + 1) % 25000 == 0:
            print(f"Normalized {description}: {index + 1:,}/{len(raw):,}", flush=True)
    return normalized


class LetterDataset(Dataset):
    """Keep normalized images as uint8 to avoid a large permanent float copy."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: np.ndarray):
        self.images = images
        self.labels = labels
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        source_index = self.indices[index]
        return torch.from_numpy(self.images[source_index][None]), int(self.labels[source_index])


def augment_training_batch(images: torch.Tensor) -> torch.Tensor:
    """Apply mild rotation and translation only to training images, never labels."""
    count = len(images)
    angles = (torch.rand(count) * 2 - 1) * (10 * np.pi / 180)
    theta = torch.zeros(count, 2, 3)
    theta[:, 0, 0] = torch.cos(angles)
    theta[:, 0, 1] = -torch.sin(angles)
    theta[:, 1, 0] = torch.sin(angles)
    theta[:, 1, 1] = torch.cos(angles)
    theta[:, :, 2] = (torch.rand(count, 2) * 2 - 1) * 0.10
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    """Measure loss, top-1/top-3 accuracy, and confusion without augmentation."""
    model.eval()
    total_loss = 0.0
    correct = top3 = total = 0
    confusion = np.zeros((26, 26), dtype=np.int64)
    with torch.inference_mode():
        for images, labels in loader:
            images = images.float().div_(255.0)
            logits = model(images)
            total_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
            prediction = logits.argmax(1)
            correct += int((prediction == labels).sum())
            top3 += int((logits.topk(3, dim=1).indices == labels[:, None]).any(1).sum())
            total += len(labels)
            np.add.at(confusion, (labels.numpy(), prediction.numpy()), 1)
    if total == 0:
        raise ValueError("Evaluation split is empty.")
    tp = np.diag(confusion).astype(float)
    precision = np.divide(tp, confusion.sum(0), out=np.zeros(26), where=confusion.sum(0) != 0)
    recall = np.divide(tp, confusion.sum(1), out=np.zeros(26), where=confusion.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(26), where=precision + recall != 0)
    return {"images": total, "loss": total_loss / total, "accuracy": correct / total,
            "top3_accuracy": top3 / total, "macro_f1": float(f1.mean()),
            "confusion": confusion, "precision": precision, "recall": recall, "f1": f1}


def _public_metrics(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key not in ("confusion", "precision", "recall", "f1")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-train-per-class", type=int, default=0,
                        help="0 uses all available training samples after the validation split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_train_per_class < 0 or args.threads < 1:
        parser.error("Epochs, batch size, and threads must be positive; sample limit must be nonnegative.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)
    started_run = time.perf_counter()

    print("Loading official EMNIST train split (124,800 images)...", flush=True)
    raw_train, train_labels = load_idx(args.data, "train")
    rng = np.random.default_rng(args.seed)
    training_indices, validation_indices = [], []
    for class_index in range(26):
        indices = rng.permutation(np.flatnonzero(train_labels == class_index))
        validation_count = max(1, round(len(indices) * 0.10))
        validation_indices.extend(indices[:validation_count])
        candidates = indices[validation_count:]
        if args.max_train_per_class:
            candidates = candidates[:args.max_train_per_class]
        training_indices.extend(candidates)
    training_indices = np.asarray(training_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    if np.intersect1d(training_indices, validation_indices).size:
        raise RuntimeError("Training/validation leakage detected.")
    np.savez_compressed(args.report / "split_indices.npz", train=training_indices,
                        validation=validation_indices)
    normalized_train = normalize_images(raw_train, "train/validation")
    del raw_train
    generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(LetterDataset(normalized_train, train_labels, training_indices),
                                 batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    validation_loader = DataLoader(LetterDataset(normalized_train, train_labels, validation_indices),
                                   batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = create_letter_model().cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    best_accuracy, best_loss, best_epoch = -1.0, float("inf"), 0
    model_path = args.output / "model.pt"
    started_training = time.perf_counter()
    print(f"Train={len(training_indices):,}, validation={len(validation_indices):,}; CPU threads={args.threads}", flush=True)
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        loss_sum = 0.0
        correct = total = 0
        for batch_index, (images, labels) in enumerate(training_loader, start=1):
            images = augment_training_batch(images.float().div_(255.0))
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            correct += int((logits.detach().argmax(1) == labels).sum())
            total += len(labels)
            if batch_index % 250 == 0:
                print(f"Epoch {epoch}: {total:,}/{len(training_indices):,} training images", flush=True)
        validation = evaluate(model, validation_loader)
        elapsed = time.perf_counter() - started
        row = {"epoch": epoch, "train_loss": loss_sum / total, "train_accuracy": correct / total,
               "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
               "seconds": elapsed}
        history.append(row)
        print(f"Epoch {epoch}/{args.epochs}: train acc={row['train_accuracy']:.4f}, "
              f"val acc={row['validation_accuracy']:.4f}, val loss={row['validation_loss']:.4f}, "
              f"time={elapsed:.1f}s", flush=True)
        better = (validation["accuracy"] > best_accuracy or
                  (validation["accuracy"] == best_accuracy and validation["loss"] < best_loss))
        if better:
            best_accuracy, best_loss, best_epoch = validation["accuracy"], validation["loss"], epoch
            temporary = args.output / "model.pt.tmp"
            torch.save(model.state_dict(), temporary)
            temporary.replace(model_path)
        with (args.report / "history.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

    training_seconds = time.perf_counter() - started_training
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    validation = evaluate(model, validation_loader)
    print("Model selected using validation only. Evaluating official test once...", flush=True)
    started_test = time.perf_counter()
    raw_test, test_labels = load_idx(args.data, "test")
    normalized_test = normalize_images(raw_test, "official test")
    del raw_test
    test_loader = DataLoader(LetterDataset(normalized_test, test_labels, np.arange(len(test_labels))),
                             batch_size=args.batch_size, shuffle=False, num_workers=0)
    test = evaluate(model, test_loader)
    provenance_file = args.data / "provenance.json"
    provenance = json.loads(provenance_file.read_text(encoding="utf-8")) if provenance_file.exists() else None
    metadata = {
        "architecture": "LetterCNN-v1", "alphabet": ALPHABET, "input_size": 28,
        "foreground": "white", "normalize": "crop-longest20-center28",
        "trained_locally": True, "training_dataset": "EMNIST Letters", "dataset_provenance": provenance,
        "label_mapping": {letter: index for index, letter in enumerate(ALPHABET)},
        "orientation": "IDX image transposed once before preprocessing",
        "train_count": len(training_indices), "validation_count": len(validation_indices),
        "test_count": len(test_labels), "seed": args.seed, "epochs_run": args.epochs,
        "best_epoch": best_epoch, "selection": "highest validation accuracy, then lowest validation loss",
        "validation_accuracy": validation["accuracy"], "test_accuracy": test["accuracy"],
        "macro_f1": test["macro_f1"], "top3_accuracy": test["top3_accuracy"],
        "learning_rate": 0.001, "optimizer": "Adam", "batch_size": args.batch_size,
        "augmentation_training_only": {"rotation_degrees": [-10, 10], "translation_normalized": [-0.10, 0.10]},
        "training_seconds_including_epoch_validation": training_seconds,
        "test_seconds_including_preprocessing": time.perf_counter() - started_test,
        "total_run_seconds": time.perf_counter() - started_run,
        "max_train_per_class": args.max_train_per_class,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "weights_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "limitation": "Single EMNIST character classification; not word/line OCR or guaranteed accuracy on mouse drawings.",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.report / "metrics.json").write_text(json.dumps({"model": metadata,
        "validation": _public_metrics(validation), "official_test": _public_metrics(test)}, indent=2), encoding="utf-8")
    with (args.report / "confusion.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["reference/prediction", *ALPHABET])
        for index, letter in enumerate(ALPHABET):
            writer.writerow([letter, *test["confusion"][index].tolist()])
    with (args.report / "per_class.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["letter", "support", "precision", "recall", "f1"])
        for index, letter in enumerate(ALPHABET):
            writer.writerow([letter, int(test["confusion"][index].sum()),
                             test["precision"][index], test["recall"][index], test["f1"][index]])
    print(f"Official test: accuracy={test['accuracy']:.4f}, macro-F1={test['macro_f1']:.4f}, "
          f"top3={test['top3_accuracy']:.4f}; n={len(test_labels):,}", flush=True)
    print(f"Saved weights: {model_path}\nMetrics: {args.report / 'metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
