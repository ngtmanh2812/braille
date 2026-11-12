"""Collect labelled data explicitly without training inside the UI."""

import csv

import cv2
import numpy as np
import pytest

from letter_samples import plan_personal_repeats, save_letter_sample


def _image():
    image = np.full((60, 80), 255, dtype=np.uint8)
    cv2.line(image, (20, 10), (20, 50), 0, 4)
    cv2.line(image, (20, 30), (45, 30), 0, 4)
    cv2.line(image, (45, 30), (45, 50), 0, 4)
    return image


def test_confirmed_sample_is_saved_and_deduplicated_without_changing_model(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"existing checkpoint")
    first = save_letter_sample(_image(), "H", sample_dir=tmp_path)
    second = save_letter_sample(_image(), "h", sample_dir=tmp_path)
    assert first == second and first.is_file()
    assert cv2.imread(str(first)) is not None
    with (tmp_path / "train_manifest.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["label"] == "h"
    assert model.read_bytes() == b"existing checkpoint"


def test_conflicting_label_does_not_overwrite_original_manifest(tmp_path):
    save_letter_sample(_image(), "h", sample_dir=tmp_path)
    before = (tmp_path / "train_manifest.csv").read_bytes()
    with pytest.raises(ValueError):
        save_letter_sample(_image(), "k", sample_dir=tmp_path)
    assert (tmp_path / "train_manifest.csv").read_bytes() == before


@pytest.mark.parametrize("label", ["", "hh", "1", "é", "../h", None])
def test_bad_labels_cannot_create_training_entries(tmp_path, label):
    with pytest.raises(ValueError):
        save_letter_sample(_image(), label, sample_dir=tmp_path)
    assert not (tmp_path / "train_manifest.csv").exists()


def test_blank_input_cannot_be_enrolled(tmp_path):
    with pytest.raises(ValueError):
        save_letter_sample(np.full((50, 80), 255, dtype=np.uint8), "h", sample_dir=tmp_path)
    assert not (tmp_path / "train_manifest.csv").exists()


def test_replay_budget_scales_to_more_than_eight_personal_samples():
    assert plan_personal_repeats(1) == 480
    assert plan_personal_repeats(100) == 40
    assert plan_personal_repeats(260) * 260 <= 4000
    with pytest.raises(ValueError):
        plan_personal_repeats(100, 480)
