"""Evaluation metrics and reproducible threshold/denoising comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from decoder import LETTER_TO_BITS
from grid_alignment import GridConfig
from pipeline import PipelineConfig, run_pipeline
from preprocessing import PreprocessConfig


def levenshtein_distance(reference: str, hypothesis: str) -> int:
    """Compute edit distance using two rows of dynamic programming."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ref_char != hyp_char)))
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Return Levenshtein distance divided by reference length (0 for both empty)."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein_distance(reference, hypothesis) / len(reference)


def _dot_counts(predicted: list[tuple[float, float]], expected: list[tuple[float, float]],
                tolerance: float = 5.0) -> tuple[int, int, int]:
    remaining = set(range(len(expected)))
    true_positive = 0
    for x, y in predicted:
        candidates = [(float(np.hypot(x - expected[j][0], y - expected[j][1])), j)
                     for j in remaining]
        if candidates:
            distance, index = min(candidates)
            if distance <= tolerance:
                true_positive += 1
                remaining.remove(index)
    return true_positive, len(predicted) - true_positive, len(expected) - true_positive


def evaluate_manifest(manifest: Path, *, split: str | None = "test",
                      threshold_method: str = "otsu", denoise: bool = True,
                      grid_mode: str = "estimated") -> tuple[list[dict[str, object]], dict[str, float]]:
    """Evaluate labeled images from the generated CSV manifest."""
    manifest = Path(manifest)
    with manifest.open(newline="", encoding="utf-8") as file:
        entries = list(csv.DictReader(file))
    if split:
        entries = [row for row in entries if row["split"] == split]
    image_root = manifest.parent
    config = PipelineConfig(
        preprocessing=PreprocessConfig(threshold_method=threshold_method, denoise=denoise),
        grid=GridConfig(mode=grid_mode, dot_spacing=18.0 if grid_mode == "fixed" else None,
                        cell_stride=54.0 if grid_mode == "fixed" else None,
                        origin_x=24.0 if grid_mode == "fixed" else None,
                        origin_y=24.0 if grid_mode == "fixed" else None),
    )
    results: list[dict[str, object]] = []
    total_edits = total_chars = total_exact_cells = total_cells = 0
    all_tp = all_fp = all_fn = 0
    for row in entries:
        source_image = image_root / row["path"]
        expected_text = row["text"]
        result = run_pipeline(source_image, config)
        edits = levenshtein_distance(expected_text, result.text)
        gt_bits = [LETTER_TO_BITS[char] for char in expected_text]
        compared_cells = max(len(gt_bits), len(result.patterns))
        exact_cells = sum(i < len(gt_bits) and i < len(result.patterns)
                          and result.patterns[i] == gt_bits[i]
                          for i in range(compared_cells))
        expected_centers = [tuple(map(float, point)) for point in json.loads(row["dot_centers_json"])]
        predicted_centers = [dot.center for dot in result.dots]
        tp, fp, fn = _dot_counts(predicted_centers, expected_centers)
        total_edits += edits
        total_chars += len(expected_text)
        total_exact_cells += exact_cells
        total_cells += compared_cells
        all_tp += tp
        all_fp += fp
        all_fn += fn
        results.append({
            "path": row["path"], "source_id": row["source_id"], "split": row["split"],
            "ground_truth": expected_text, "prediction": result.text,
            "edit_distance": edits, "cer": character_error_rate(expected_text, result.text),
            "exact_cells": exact_cells, "cell_count": compared_cells,
            "cell_accuracy": exact_cells / compared_cells if compared_cells else 1.0,
            "dot_tp": tp, "dot_fp": fp, "dot_fn": fn,
            "processing_time_ms": result.processing_time_ms, "dots_detected": len(result.dots),
        })
    summary = {
        "images": float(len(results)),
        "cer": total_edits / total_chars if total_chars else 0.0,
        "cell_accuracy": total_exact_cells / total_cells if total_cells else 0.0,
        "dot_precision": all_tp / (all_tp + all_fp) if all_tp + all_fp else 1.0,
        "dot_recall": all_tp / (all_tp + all_fn) if all_tp + all_fn else 1.0,
        "mean_processing_time_ms": (sum(float(row["processing_time_ms"]) for row in results) / len(results)
                                     if results else 0.0),
    }
    return results, summary


def run_comparisons(manifest: Path, output_dir: Path, split: str = "test") -> list[dict[str, object]]:
    """Run the two required comparisons on the same labeled split."""
    cases = [
        ("global_threshold", "global", True),
        ("adaptive_threshold", "adaptive", True),
        ("otsu_no_denoise", "otsu", False),
        ("otsu_with_denoise", "otsu", True),
    ]
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for name, method, denoise in cases:
        rows, summary = evaluate_manifest(manifest, split=split,
                                          threshold_method=method, denoise=denoise)
        for row in rows:
            all_rows.append({"configuration": name, **row})
        summaries.append({"configuration": name, "threshold": method,
                          "denoise": denoise, **summary})
    output_dir.mkdir(parents=True, exist_ok=True)
    if all_rows:
        with (output_dir / "evaluation_rows.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    with (output_dir / "evaluation_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([str(row["configuration"]) for row in summaries], [float(row["cer"]) for row in summaries])
    ax.set_ylabel("Character error rate (CER)")
    ax.set_title(f"Synthetic {split} split: threshold and denoising configurations")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_cer.png", dpi=160)
    plt.close(fig)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="?", default=Path("data/synthetic/labels.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    summaries = run_comparisons(args.manifest, args.output, args.split)
    for summary in summaries:
        print(f"{summary['configuration']}: CER={summary['cer']:.4f}, "
              f"cell accuracy={summary['cell_accuracy']:.4f}, "
              f"dot P/R={summary['dot_precision']:.4f}/{summary['dot_recall']:.4f}, "
              f"time={summary['mean_processing_time_ms']:.2f} ms")
    print(f"CSV and plot written to: {args.output}")


if __name__ == "__main__":
    main()
