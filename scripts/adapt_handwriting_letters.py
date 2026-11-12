"""Adapt a letter CNN using explicitly labelled personal samples plus EMNIST.

Personal images participate only in training. The original validation split
selects the checkpoint; the official EMNIST test is evaluated once afterwards.
Correct classification of an enrolled personal image is a training-case result,
not independent evidence of recognition accuracy on a person's handwriting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import string
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from handwritten_letters import create_letter_model, preprocess_letter
from letter_samples import plan_personal_repeats
from train_handwriting_letters import (
    LetterDataset, _public_metrics, augment_training_batch, evaluate,
    load_idx, normalize_images,
)

ALPHABET = string.ascii_lowercase


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_personal_samples(manifest: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Read explicit labels and retain hashes of each unique original source."""
    base = manifest.resolve().parent
    with manifest.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"path", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("Personal manifest requires path,label columns.")
        rows = list(reader)
    if not rows:
        raise ValueError("Personal training manifest is empty.")
    images, labels, sources = [], [], []
    observed: set[str] = set()
    for row in rows:
        relative = Path(row["path"])
        label = row["label"].strip()
        if relative.is_absolute() or label not in ALPHABET or len(label) != 1:
            raise ValueError("Manifest paths must be relative; labels must be a single lowercase a-z.")
        path = (base / relative).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            raise ValueError(f"Personal image missing or outside the manifest directory: {relative}")
        digest = sha256_file(path)
        if digest in observed:
            raise ValueError("Duplicate personal source bytes in manifest. Use --repeats for intentional repetition.")
        observed.add(digest)
        image = preprocess_letter(path)
        images.append(image)
        labels.append(ALPHABET.index(label))
        sources.append({"path": relative.as_posix(), "label": label, "sha256": digest,
                        "normalized_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                        "role": "personal training sample; not an independent test sample"})
    return np.stack(images), np.asarray(labels, dtype=np.int64), sources


def load_original_splits(path: Path, labels: np.ndarray, baseline: dict) -> tuple[np.ndarray, np.ndarray]:
    """Require the exact original seeded partition, with all sources accounted for."""
    with np.load(path, allow_pickle=False) as saved:
        train = saved["train"]
        validation = saved["validation"]
    for name, indices in (("train", train), ("validation", validation)):
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"Invalid {name} indices.")
        if np.any(indices < 0) or np.any(indices >= len(labels)) or len(np.unique(indices)) != len(indices):
            raise ValueError(f"Invalid or repeated {name} source indices.")
    if len(train) != baseline["train_count"] or len(validation) != baseline["validation_count"]:
        raise ValueError("Saved split counts differ from baseline metadata.")
    if np.intersect1d(train, validation).size or len(train) + len(validation) != len(labels):
        raise ValueError("Training/validation overlap or incomplete original split.")
    rng = np.random.default_rng(baseline["seed"])
    expected_train, expected_validation = [], []
    for class_index in range(26):
        indices = rng.permutation(np.flatnonzero(labels == class_index))
        count = max(1, round(len(indices) * 0.10))
        expected_validation.extend(indices[:count])
        expected_train.extend(indices[count:])
    if not np.array_equal(train, expected_train) or not np.array_equal(validation, expected_validation):
        raise ValueError("Saved indices do not match the full original seeded EMNIST partition.")
    return train.astype(np.int64), validation.astype(np.int64)


def personal_training_predictions(model, images: np.ndarray, labels: np.ndarray) -> list[dict]:
    """Record enrolled-image performance without using it for checkpoint selection."""
    model.eval()
    with torch.inference_mode():
        indices = model(torch.from_numpy(images[:, None]).float().div_(255.0)).argmax(1).tolist()
    return [{"label": ALPHABET[int(label)], "prediction": ALPHABET[index],
             "correct": index == int(label), "evaluation_role": "enrolled training image"}
            for index, label in zip(indices, labels)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/handwriting/user_samples/train_manifest.csv")
    parser.add_argument("--data", type=Path, default=ROOT / "data/handwriting/emnist_letters")
    parser.add_argument("--base", type=Path, default=ROOT / "models/handwriting_letters")
    parser.add_argument("--splits", type=Path, default=ROOT / "outputs/handwriting_letters_training/split_indices.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "models/handwriting_letters_personal")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/handwriting_letters_personal")
    parser.add_argument("--repeats", type=int, default=0,
                        help="0 selects up to 480 repeats per source, at most 4,000 personal entries total")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.learning_rate <= 0 or args.repeats < 0:
        parser.error("Epochs, batch size, threads and learning rate must be positive; repeats must be nonnegative.")
    if args.output.resolve() == args.base.resolve():
        parser.error("Personalization output must not overwrite the baseline model directory.")
    baseline = json.loads((args.base / "metadata.json").read_text(encoding="utf-8"))
    if (baseline.get("architecture") != "LetterCNN-v1" or baseline.get("alphabet") != ALPHABET
            or baseline.get("normalize") != "crop-longest20-center28"):
        parser.error("Baseline architecture, alphabet, or preprocessing is incompatible.")
    base_weights = args.base / "model.pt"
    base_hash = sha256_file(base_weights)
    if base_hash != baseline.get("weights_sha256"):
        parser.error("Baseline weights do not match their recorded SHA256.")
    personal_images, personal_labels, sources = load_personal_samples(args.manifest)
    args.repeats = plan_personal_repeats(len(sources), args.repeats)
    personal_count = len(sources) * args.repeats
    if personal_count > 4000:
        parser.error("At most 4,000 repeated personal training entries are allowed per run.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)
    started_run = time.perf_counter()
    print("Loading original EMNIST train split and fixed validation partition...", flush=True)
    raw_train, labels = load_idx(args.data, "train")
    train_indices, validation_indices = load_original_splits(args.splits, labels, baseline)
    normalized = normalize_images(raw_train, "adaptation train/validation")
    del raw_train
    original_count = len(normalized)
    images = np.concatenate((normalized, personal_images))
    all_labels = np.concatenate((labels, personal_labels))
    del normalized
    personal_indices = np.repeat(np.arange(original_count, original_count + len(sources)), args.repeats)
    combined_indices = np.concatenate((train_indices, personal_indices))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(LetterDataset(images, all_labels, combined_indices),
                              batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    validation_loader = DataLoader(LetterDataset(images, all_labels, validation_indices),
                                   batch_size=args.batch_size, shuffle=False, num_workers=0)
    np.savez_compressed(args.report / "split_indices.npz", emnist_train=train_indices,
                        validation=validation_indices, personal_training=personal_indices)
    model = create_letter_model().cpu()
    model.load_state_dict(torch.load(base_weights, map_location="cpu", weights_only=True))
    base_validation = evaluate(model, validation_loader)
    enrolled_before = personal_training_predictions(model, personal_images, personal_labels)
    # The baseline is an explicit candidate: adaptation cannot win by reference to test or personal labels.
    best_accuracy, best_loss, best_epoch = base_validation["accuracy"], base_validation["loss"], 0
    model_path = args.output / "model.pt"
    torch.save(model.state_dict(), model_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    started_training = time.perf_counter()
    print(f"Original train={len(train_indices):,}; personal unique={len(sources)}, "
          f"repeated entries={personal_count}; fixed validation={len(validation_indices):,}", flush=True)
    print(f"Baseline validation accuracy={best_accuracy:.4f}; enrolled images before={enrolled_before}", flush=True)
    for epoch in range(1, args.epochs + 1):
        started_epoch = time.perf_counter()
        model.train()
        loss_sum = 0.0
        correct = total = 0
        for batch_index, (batch_images, batch_labels) in enumerate(train_loader, start=1):
            batch_images = augment_training_batch(batch_images.float().div_(255.0))
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_images)
            loss = F.cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_labels)
            correct += int((logits.detach().argmax(1) == batch_labels).sum())
            total += len(batch_labels)
            if batch_index % 250 == 0:
                print(f"Adapt epoch {epoch}: {total:,}/{len(combined_indices):,}", flush=True)
        validation = evaluate(model, validation_loader)
        row = {"epoch": epoch, "train_loss": loss_sum / total, "train_accuracy": correct / total,
               "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
               "seconds": time.perf_counter() - started_epoch}
        history.append(row)
        print(f"Adapt epoch {epoch}/{args.epochs}: train={row['train_accuracy']:.4f}, "
              f"val={row['validation_accuracy']:.4f}, loss={row['validation_loss']:.4f}, "
              f"time={row['seconds']:.1f}s", flush=True)
        if (validation["accuracy"] > best_accuracy or
                (validation["accuracy"] == best_accuracy and validation["loss"] < best_loss)):
            best_accuracy, best_loss, best_epoch = validation["accuracy"], validation["loss"], epoch
            temporary = args.output / "model.pt.tmp"
            torch.save(model.state_dict(), temporary)
            temporary.replace(model_path)
        with (args.report / "history.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    adaptation_seconds = time.perf_counter() - started_training
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    validation = evaluate(model, validation_loader)
    enrolled_after = personal_training_predictions(model, personal_images, personal_labels)
    print(f"Selected checkpoint epoch={best_epoch} by validation; now evaluating official test once.", flush=True)
    raw_test, test_labels = load_idx(args.data, "test")
    test_images = normalize_images(raw_test, "adaptation official test")
    del raw_test
    test_loader = DataLoader(LetterDataset(test_images, test_labels, np.arange(len(test_labels))),
                             batch_size=args.batch_size, shuffle=False, num_workers=0)
    test = evaluate(model, test_loader)

    metadata = {**baseline,
        "personalized": best_epoch > 0, "personalization_attempted": True,
        "selected_checkpoint": "baseline" if best_epoch == 0 else f"adaptation_epoch_{best_epoch}",
        "base_model": str(args.base), "base_weights_sha256": base_hash,
        "training_dataset": "EMNIST Letters plus explicitly labelled personal training images",
        "train_count": len(combined_indices), "emnist_train_count": len(train_indices),
        "personal_unique_sources": len(sources), "personal_repeated_entries": personal_count,
        "personal_repeats_per_source": args.repeats, "personal_sources": sources,
        "manifest_sha256": sha256_file(args.manifest), "original_split_sha256": sha256_file(args.splits),
        "validation_count": len(validation_indices), "test_count": len(test_labels),
        "seed": args.seed, "original_split_seed": baseline["seed"],
        "epochs_run": args.epochs, "best_epoch": best_epoch,
        "selection": "highest unchanged EMNIST validation accuracy, then lowest loss; baseline included as epoch 0",
        "baseline_validation_accuracy": base_validation["accuracy"],
        "validation_accuracy": validation["accuracy"], "test_accuracy": test["accuracy"],
        "macro_f1": test["macro_f1"], "top3_accuracy": test["top3_accuracy"],
        "learning_rate": args.learning_rate, "batch_size": args.batch_size,
        "adaptation_seconds_including_validation": adaptation_seconds,
        "total_run_seconds": time.perf_counter() - started_run,
        "weights_sha256": sha256_file(model_path), "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "limitation": "Enrolled personal-image results are training diagnostics; no claim of generalization to new handwriting.",
    }
    # Remove baseline-only timing fields rather than mislabel them as this run's timings.
    for key in ("training_seconds_including_epoch_validation", "test_seconds_including_preprocessing"):
        metadata.pop(key, None)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = {"model": metadata, "baseline_validation": _public_metrics(base_validation),
              "validation": _public_metrics(validation), "official_test": _public_metrics(test),
              "enrolled_training_images_before": enrolled_before,
              "enrolled_training_images_after": enrolled_after,
              "enrolled_image_note": "These are the same labelled images used for training, not independent test samples."}
    (args.report / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
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
    print(f"Official test accuracy={test['accuracy']:.4f}, macro-F1={test['macro_f1']:.4f}, "
          f"enrolled training predictions={enrolled_after}", flush=True)
    print(f"Personal model: {model_path}; report: {args.report / 'metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
