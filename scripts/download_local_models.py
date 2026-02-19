#!/usr/bin/env python3
"""Download local inference models to a scratch directory.

By default this script performs a dry run and prints what it would download.
To actually fetch weights, pass ``--execute`` and ``--scratch-dir``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


MODEL_IDS = [
    "Nanbeige/Nanbeige4.1-3B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "allenai/Olmo-3-7B-Instruct",
]

# Leave empty by default; set this when running on your cluster.
DEFAULT_SCRATCH_DIR = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download local vLLM model snapshots")
    parser.add_argument(
        "--scratch-dir",
        type=str,
        default=DEFAULT_SCRATCH_DIR,
        help="Scratch directory where model snapshots should be stored",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually download files (default is dry run)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.getenv("HF_TOKEN", ""),
        help="Optional Hugging Face token (defaults to HF_TOKEN env var)",
    )
    return parser.parse_args()


def _target_dir(root: Path, model_id: str) -> Path:
    return root / model_id.replace("/", "__")


def main() -> int:
    args = parse_args()

    print("Models:")
    for model_id in MODEL_IDS:
        print(f"  - {model_id}")

    if not args.scratch_dir:
        print("\nNo scratch directory configured. Set --scratch-dir to enable downloads.")
        return 0

    scratch_dir = Path(args.scratch_dir).expanduser()
    print(f"\nScratch directory: {scratch_dir}")

    if not args.execute:
        print("Dry run only. Re-run with --execute to download.")
        for model_id in MODEL_IDS:
            print(f"  would download -> {_target_dir(scratch_dir, model_id)}")
        return 0

    scratch_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    for model_id in MODEL_IDS:
        local_dir = _target_dir(scratch_dir, model_id)
        print(f"\nDownloading {model_id} -> {local_dir}")
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            token=args.hf_token or None,
            resume_download=True,
        )

    print("\nDownload complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
