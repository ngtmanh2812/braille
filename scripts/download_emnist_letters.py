"""Download the EMNIST Letters IDX files from a hash-checked public mirror.

NIST publishes the full EMNIST archive. This script uses individually downloadable
mirror files to avoid fetching unrelated splits, and records that provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "handwriting" / "emnist_letters"
REPOSITORY = "Royc30ne/emnist-letters"
FILES = {
    "emnist-letters-train-images-idx3-ubyte.gz": "6288f418a917e007bf5fddb823a2dc7a5c6a35031401b93292a539cb12bd881f",
    "emnist-letters-train-labels-idx1-ubyte.gz": "9759af91ea6bccdf07a6040fd085861f2ec88e37f7c3f5760f6feb57106fbe5c",
    "emnist-letters-test-images-idx3-ubyte.gz": "50f0e93d75b99b463e9760b8345866f22587fa2c2fba472730a5ba09c693412d",
    "emnist-letters-test-labels-idx1-ubyte.gz": "924955c31fe7fd809b7303b5c0141282955f4b8390c74cdfd5d80435e0f4b4cb",
}


def sha256_file(path: Path) -> str:
    """Hash a downloaded file without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(output: Path, revision: str | None = None) -> Path:
    """Fetch four immutable, hash-checked files and save their exact source URLs."""
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "provenance.json"
    session = requests.Session()
    if revision is None and provenance_path.is_file():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        if previous.get("mirror_repository") == REPOSITORY:
            revision = previous.get("mirror_revision")
    if revision is None:
        metadata = session.get(f"https://huggingface.co/api/datasets/{REPOSITORY}", timeout=30)
        metadata.raise_for_status()
        revision = metadata.json()["sha"]
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("Expected an immutable 40-character dataset commit SHA.")

    records = []
    for filename, expected_hash in FILES.items():
        destination = output / filename
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{filename}"
        if destination.is_file() and sha256_file(destination) == expected_hash:
            print(f"Already verified: {filename}", flush=True)
        else:
            print(f"Downloading: {filename}", flush=True)
            temporary = output / (filename + ".download")
            with session.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with temporary.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file.write(chunk)
            observed = sha256_file(temporary)
            if observed != expected_hash:
                raise ValueError(f"SHA256 mismatch for {filename}: {observed}. The file was not installed.")
            temporary.replace(destination)
            print(f"Verified: {filename} ({destination.stat().st_size:,} bytes)", flush=True)
        records.append({"filename": filename, "url": url, "sha256": expected_hash,
                        "bytes": destination.stat().st_size})
    provenance = {
        "dataset": "EMNIST Letters", "original_publisher": "NIST",
        "original_source": "https://www.nist.gov/itl/products-and-services/emnist-dataset",
        "paper": "https://arxiv.org/abs/1702.05373",
        "download_source_kind": "third-party mirror of the IDX files, not a direct NIST download",
        "mirror_repository": REPOSITORY, "mirror_revision": revision,
        "mirror_card": f"https://huggingface.co/datasets/{REPOSITORY}",
        "files": records,
        "labels": "1..26; upper/lowercase forms merge into the same Latin letter",
        "expected_counts": {"train": 124800, "test": 20800},
        "note": "Pinned compressed-file hashes checked; no claim of independently comparing the entire NIST archive.",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--revision", help="Optional immutable Hugging Face dataset commit SHA")
    args = parser.parse_args()
    try:
        path = download_dataset(args.output, args.revision)
    except (OSError, ValueError, requests.RequestException, KeyError) as exc:
        parser.exit(1, f"Dataset download failed: {exc}\nRerun this command after checking the network.\n")
    print(f"Dataset ready. Provenance: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
