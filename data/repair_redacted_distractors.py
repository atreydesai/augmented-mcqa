"""Repair augmented stores: replace Inspect char-split redaction strings.

The Inspect scorer summary truncates metadata fields >1 k chars by replacing
them with the literal string "Key removed from summary (> 1k)".  When stored as
a list the string is split character-by-character, producing corrupt
model_distractors / human_distractors / distractors values.

Fix (deterministic and verified 100% on good rows):
  model_distractors = options_randomized - {correct option}

Run:
    uv run python data/repair_redacted_distractors.py [--dry-run] [--augmented-root PATH]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# ── bootstrap path so we can import project modules ──────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Dataset, load_from_disk

from data.store import _is_redaction_string, _INSPECT_REDACTION_MARKER
from utils.constants import CHOICE_LABELS, SETTING_NAMES

AUGMENTED_RECORD_COLUMNS = (
    "id", "question_id", "dataset_type", "row_index", "sample_id",
    "question", "answer", "category", "options", "answer_index",
    "choices_human", "setting", "generation_strategy", "num_human",
    "num_model", "num_choices", "human_distractors", "model_distractors",
    "distractors", "options_randomized", "correct_answer_letter", "traces",
)


def _derive_model_distractors(row: dict) -> list[str]:
    """Derive model_distractors from options_randomized minus the correct answer."""
    opts = list(row.get("options_randomized") or [])
    letter = str(row.get("correct_answer_letter") or "")
    if not opts or letter not in CHOICE_LABELS[: len(opts)]:
        return []
    correct_idx = CHOICE_LABELS.index(letter)
    return [opts[i] for i in range(len(opts)) if i != correct_idx]


def _needs_repair(row: dict) -> bool:
    """Return True if any distractor field looks like the char-split redaction string."""
    for field in ("model_distractors", "human_distractors", "distractors"):
        val = list(row.get(field) or [])
        if _is_redaction_string(val):
            return True
    return False


def repair_dataset(ds: Dataset, *, setting: str) -> tuple[Dataset, int]:
    """Return a repaired copy of *ds* and the number of rows that were fixed."""
    repaired_rows: list[dict] = []
    n_fixed = 0
    for row in ds:
        payload = dict(row)
        if not _needs_repair(payload):
            repaired_rows.append(payload)
            continue

        opts = list(payload.get("options_randomized") or [])
        letter = str(payload.get("correct_answer_letter") or "")
        if not opts or letter not in CHOICE_LABELS[: len(opts)]:
            # Cannot repair: no usable options_randomized.
            raise RuntimeError(
                f"Cannot repair sample {payload.get('sample_id')!r} in setting "
                f"{setting!r}: options_randomized is missing or correct_answer_letter "
                "is invalid.  Manual intervention required."
            )

        model_d = _derive_model_distractors(payload)
        human_d = list(payload.get("human_distractors") or [])
        # human_distractors is typically [] for augment_model/augment_ablation
        if _is_redaction_string(human_d):
            human_d = []

        payload["model_distractors"] = model_d
        payload["human_distractors"] = human_d
        payload["distractors"] = [*human_d, *model_d]
        payload["num_model"] = len(model_d)
        payload["num_human"] = len(human_d)
        repaired_rows.append(payload)
        n_fixed += 1

    return Dataset.from_list(repaired_rows), n_fixed


def repair_store(augmented_root: Path, *, dry_run: bool = False) -> dict:
    """Walk every setting-split under *augmented_root* and patch corrupted rows.

    Returns a summary dict: {(gen_run, bench, setting): (n_fixed, total)}.
    """
    summary: dict[tuple[str, str, str], tuple[int, int]] = {}

    for gen_run_dir in sorted(augmented_root.iterdir()):
        if not gen_run_dir.is_dir():
            continue
        gen_run = gen_run_dir.name
        for model_dir in sorted(p for p in gen_run_dir.iterdir() if p.is_dir()):
            for bench_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                bench = bench_dir.name
                for setting_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
                    setting = setting_dir.name
                    if setting not in SETTING_NAMES:
                        continue

                    try:
                        ds = load_from_disk(str(setting_dir))
                    except Exception as exc:
                        print(f"  SKIP (load error) {setting_dir}: {exc}")
                        continue

                    n_bad = sum(1 for row in ds if _needs_repair(dict(row)))
                    total = len(ds)
                    key = (gen_run, bench, setting)
                    summary[key] = (n_bad, total)

                    if n_bad == 0:
                        continue

                    print(
                        f"  {'[DRY RUN] ' if dry_run else ''}"
                        f"Repairing {gen_run}/{bench}/{setting}: "
                        f"{n_bad}/{total} rows corrupted"
                    )

                    if dry_run:
                        continue

                    repaired_ds, n_fixed = repair_dataset(ds, setting=setting)
                    assert n_fixed == n_bad, f"Expected {n_bad} fixes, got {n_fixed}"

                    # Atomic write: tmp → rename
                    tmp = setting_dir.with_name(f".{setting_dir.name}.repair_tmp")
                    bak = setting_dir.with_name(f".{setting_dir.name}.repair_bak")
                    if tmp.exists():
                        shutil.rmtree(tmp)
                    repaired_ds.save_to_disk(str(tmp))
                    if setting_dir.exists():
                        setting_dir.rename(bak)
                    tmp.rename(setting_dir)
                    if bak.exists():
                        shutil.rmtree(bak, ignore_errors=True)

                    print(f"    ✓ Fixed {n_fixed} rows → {setting_dir}")

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Repair char-split Inspect redaction strings in augmented stores."
    )
    parser.add_argument(
        "--augmented-root",
        default="datasets/augmented",
        help="Root directory of augmented stores (default: datasets/augmented).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be changed without writing anything.",
    )
    args = parser.parse_args(argv)

    root = Path(args.augmented_root)
    if not root.exists():
        print(f"ERROR: augmented root not found: {root}", file=sys.stderr)
        return 1

    print(f"{'DRY RUN: ' if args.dry_run else ''}Scanning {root} ...\n")
    summary = repair_store(root, dry_run=args.dry_run)

    total_fixed = sum(n for n, _ in summary.values())
    total_rows = sum(t for _, t in summary.values())
    affected_slices = sum(1 for n, _ in summary.values() if n > 0)

    print(f"\n{'─' * 70}")
    if args.dry_run:
        print(f"DRY RUN complete. Would repair {total_fixed} rows across {affected_slices} slices.")
    else:
        print(f"Repair complete. Fixed {total_fixed}/{total_rows} rows across {affected_slices} slices.")

    if affected_slices > 0:
        print("\nAffected slices:")
        for (gen_run, bench, setting), (n_bad, total) in sorted(summary.items()):
            if n_bad > 0:
                pct = n_bad / total * 100
                print(f"  {gen_run}/{bench}/{setting}: {n_bad}/{total} ({pct:.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
