"""Download the pinned public Microsoft TrOCR model into the project's models/."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handwriting import DEFAULT_MODEL_DIR, MODEL_FILES, MODEL_ID, MODEL_REVISION, model_files_ready


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        parser.error("Install requirements-handwriting.txt in the project .venv first.")
    try:
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=args.output,
                          allow_patterns=list(MODEL_FILES) + ["generation_config.json", "README.md"],
                          max_workers=2)
    except Exception as exc:
        parser.exit(1, f"Download failed; you can rerun to resume: {exc}\n")
    if not model_files_ready(args.output):
        parser.exit(1, "Download incomplete. Rerun this command.\n")
    (args.output / "provenance.json").write_text(json.dumps({
        "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "source": f"https://huggingface.co/{MODEL_ID}",
        "training_dataset_reported_by_publisher": "IAM",
        "locally_trained": False,
    }, indent=2), encoding="utf-8")
    print(f"Model ready: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
