from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.final5_store import migrate_augmented_dataset_in_place
from utils.constants import AUGMENTED_STORE_MANIFEST


def _discover_augmented_roots(root: Path) -> list[Path]:
    roots: list[Path] = []
    if not root.exists():
        return roots
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        for model_dir in sorted(path for path in run_dir.iterdir() if path.is_dir() and not path.name.startswith(".")):
            if (model_dir / "dataset_dict.json").exists() or (model_dir / AUGMENTED_STORE_MANIFEST).exists():
                roots.append(model_dir)
    return roots


def main() -> int:
    parser = argparse.ArgumentParser(description="Rewrite checked-in augmented datasets into the setting-record store.")
    parser.add_argument(
        "--root",
        default="datasets/augmented",
        help="Augmented datasets root containing <run>/<model>/ directories.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    migrated = 0
    skipped = 0
    for dataset_root in _discover_augmented_roots(root):
        if (dataset_root / AUGMENTED_STORE_MANIFEST).exists():
            skipped += 1
            print(f"skip {dataset_root}")
            continue
        migrate_augmented_dataset_in_place(dataset_root)
        migrated += 1
        print(f"migrated {dataset_root}")

    print(f"migrated={migrated} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
