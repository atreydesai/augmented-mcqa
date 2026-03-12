from __future__ import annotations

from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, concatenate_datasets

from config import PROCESSED_DATASETS_DIR, RAW_DATASETS_DIR
from data.arc_processor import process_arc_for_experiments
from data.downloader import download_arc, download_gpqa, download_mmlu_all_configs, download_mmlu_pro
from data.gpqa_processor import process_gpqa_for_experiments
from data.mmlu_pro_processor import process_mmlu_pro


PROCESSED_SCHEMA_VERSION = "final5_processed_v3"
DEFAULT_PER_DATASET_LIMIT = 1000
EXCLUDED_QUESTION_IDS = {
    "gpqa": {
        "recDjE01bu72pPUU2",
        "recZSGUkn56v9kEp1",
        "recnGEpF1srQpaqWq",
        "recnTTKdBzfuoZ7w7",
    },
    "mmlu_pro": {"996"},
}


def _limit_dataset(ds: Dataset, limit: Optional[int]) -> Dataset:
    if limit is None:
        return ds
    return ds.select(range(min(limit, len(ds))))


def _add_or_replace_column(ds: Dataset, name: str, values: list) -> Dataset:
    if name in ds.column_names:
        ds = ds.remove_columns([name])
    return ds.add_column(name, values)


def _row_identifier(ds: Dataset, dataset_type: str, row_index: int) -> str:
    row = ds[row_index]
    if dataset_type == "mmlu_pro":
        value = row.get("question_id")
    else:
        value = row.get("id")
    return str(value).strip()


def _exclude_known_bad_questions(ds: Dataset, dataset_type: str) -> Dataset:
    excluded_ids = EXCLUDED_QUESTION_IDS.get(dataset_type)
    if not excluded_ids:
        return ds

    keep_indices = [
        row_index
        for row_index in range(len(ds))
        if _row_identifier(ds, dataset_type, row_index) not in excluded_ids
    ]
    if len(keep_indices) == len(ds):
        return ds
    return ds.select(keep_indices)


def _ensure_common_columns(ds: Dataset, dataset_type: str) -> Dataset:
    n = len(ds)
    defaults = {
        "question": "",
        "options": [],
        "answer": "",
        "answer_index": -1,
        "answer_letter": "",
        "category": "",
        "src": "",
        "subfield": "",
        "difficulty": "",
        "discipline": "",
        "labels": [],
        "choices_answer": [],
        "choices_human": [],
    }

    for col, default in defaults.items():
        if col not in ds.column_names:
            ds = ds.add_column(col, [default for _ in range(n)])

    ds = _add_or_replace_column(ds, "dataset_type", [dataset_type for _ in range(n)])
    ds = _add_or_replace_column(ds, "schema_version", [PROCESSED_SCHEMA_VERSION for _ in range(n)])
    return ds


def _process_mmlu_pro(limit: Optional[int]) -> Dataset:
    output_path = PROCESSED_DATASETS_DIR / "mmlu_pro_processed"
    processed = process_mmlu_pro(output_path=output_path, limit=None)

    if "train" in processed and "test" in processed:
        merged = concatenate_datasets([processed["train"], processed["test"]])
    elif "train" in processed:
        merged = processed["train"]
    elif "test" in processed:
        merged = processed["test"]
    else:
        raise ValueError("MMLU-Pro processor returned no train/test split")

    merged = _limit_dataset(merged, limit)
    merged = _exclude_known_bad_questions(merged, "mmlu_pro")
    return _ensure_common_columns(merged, "mmlu_pro")


def _process_arc_challenge(limit: Optional[int]) -> Dataset:
    ds = process_arc_for_experiments(
        difficulty="challenge",
        output_path=PROCESSED_DATASETS_DIR / "arc_processed" / "arc_challenge",
        limit=None,
    )
    ds = _limit_dataset(ds, limit)
    ds = _exclude_known_bad_questions(ds, "arc_challenge")
    return _ensure_common_columns(ds, "arc_challenge")


def _process_gpqa(limit: Optional[int]) -> Dataset:
    ds = process_gpqa_for_experiments(
        output_path=PROCESSED_DATASETS_DIR / "gpqa_processed",
        limit=None,
    )
    ds = _limit_dataset(ds, limit)
    ds = _exclude_known_bad_questions(ds, "gpqa")
    return _ensure_common_columns(ds, "gpqa")


def process_unified_dataset(limit: Optional[int] = None, output_path: Optional[Path] = None) -> DatasetDict:
    effective_limit = DEFAULT_PER_DATASET_LIMIT if limit is None else limit

    print("Processing MMLU-Pro with preserved MMLU filtering...")
    mmlu_pro = _process_mmlu_pro(effective_limit)

    print("Processing ARC-Challenge...")
    arc_challenge = _process_arc_challenge(effective_limit)

    print("Processing GPQA...")
    gpqa = _process_gpqa(effective_limit)

    unified = DatasetDict(
        {
            "arc_challenge": arc_challenge,
            "mmlu_pro": mmlu_pro,
            "gpqa": gpqa,
        }
    )

    target_path = Path(output_path or (PROCESSED_DATASETS_DIR / "unified_processed_v3"))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    unified.save_to_disk(str(target_path))

    print(f"Saved unified dataset to {target_path}")
    for split in unified.keys():
        print(f"  - {split}: {len(unified[split])} rows")

    return unified


def download_raw_datasets(
    *,
    dataset: str | None = None,
    download_all: bool = False,
    output_dir: Path | str | None = None,
) -> int:
    target_dir = Path(output_dir or RAW_DATASETS_DIR)

    datasets_to_download: list[str]
    if download_all:
        datasets_to_download = ["mmlu_pro", "mmlu", "arc", "gpqa"]
    elif dataset:
        datasets_to_download = [dataset]
    else:
        print("Must specify --dataset or --all")
        return 1

    for dataset_name in datasets_to_download:
        print(f"\n{'=' * 50}")
        print(f"Downloading: {dataset_name}")
        print("=" * 50)
        if dataset_name == "mmlu_pro":
            download_mmlu_pro(target_dir / "mmlu_pro")
        elif dataset_name == "mmlu":
            download_mmlu_all_configs(target_dir / "mmlu_all")
        elif dataset_name == "arc":
            download_arc(target_dir / "arc")
        elif dataset_name == "gpqa":
            download_gpqa(target_dir / "gpqa")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"\nDatasets saved to: {target_dir}")
    return 0


def prepare_data(
    *,
    step: str,
    dataset: str | None = None,
    download_all: bool = False,
    output_dir: Path | str | None = None,
    output_path: Path | str | None = None,
    limit: int | None = None,
) -> int:
    if step == "download":
        return download_raw_datasets(dataset=dataset, download_all=download_all, output_dir=output_dir)
    if step == "process":
        process_unified_dataset(limit=limit, output_path=Path(output_path) if output_path else None)
        return 0
    if step == "all":
        rc = download_raw_datasets(dataset=dataset, download_all=download_all, output_dir=output_dir)
        if rc != 0:
            return rc
        process_unified_dataset(limit=limit, output_path=Path(output_path) if output_path else None)
        return 0
    raise ValueError(f"Unknown step: {step}")
