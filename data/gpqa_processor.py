"""
GPQA dataset processor.

Maps Idavidrein/gpqa (subset=gpqa_main, split=train) to the repository's
unified schema for generation/evaluation workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset

from config import HF_TOKEN, PROCESSED_DATASETS_DIR


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_gpqa_dataset(
    split: str = "train",
    subset: str = "gpqa_main",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load GPQA and map fields into unified format.

    Mapping:
    - question <- Question
    - answer <- Correct Answer
    - choices_answer <- [Correct Answer]
    - choices_human <- [Incorrect Answer 1, Incorrect Answer 2, Incorrect Answer 3]
    - subfield <- Subdomain
    - category/src/difficulty/options/answer_index left null-safe
    """
    load_kwargs: Dict[str, Any] = {}
    if HF_TOKEN:
        load_kwargs["token"] = HF_TOKEN

    ds = load_dataset("Idavidrein/gpqa", subset, split=split, **load_kwargs)

    entries: List[Dict[str, Any]] = []
    for row in ds:
        question = _safe_text(row.get("Question"))
        answer = _safe_text(row.get("Correct Answer"))
        d1 = _safe_text(row.get("Incorrect Answer 1"))
        d2 = _safe_text(row.get("Incorrect Answer 2"))
        d3 = _safe_text(row.get("Incorrect Answer 3"))
        subfield = _safe_text(row.get("Subdomain"))

        if not question or not answer:
            continue
        if not d1 or not d2 or not d3:
            continue

        entry = {
            "id": _safe_text(row.get("Record ID")),
            "question": question,
            "options": [],
            "answer": answer,
            "answer_index": None,
            "answer_letter": "",
            "choices_answer": [answer],
            "choices_human": [d1, d2, d3],
            "category": "",
            "src": "",
            "subfield": subfield,
            "difficulty": "",
            "discipline": _safe_text(row.get("High-level domain")),
            "dataset_type": "gpqa",
        }
        entries.append(entry)

        if limit is not None and len(entries) >= limit:
            break

    return entries


def process_gpqa_for_experiments(
    split: str = "train",
    subset: str = "gpqa_main",
    limit: Optional[int] = None,
    output_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Dataset:
    """Process GPQA and save as Hugging Face Dataset on disk."""
    entries = load_gpqa_dataset(split=split, subset=subset, limit=limit)
    dataset = Dataset.from_list(entries)

    if output_path is None:
        if output_dir is None:
            output_dir = PROCESSED_DATASETS_DIR
        output_path = output_dir / "gpqa_processed"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    print(f"Saved {len(dataset)} GPQA rows to {output_path}")

    return dataset


def get_gpqa_stats(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple summary stats for mapped GPQA rows."""
    return {
        "total_entries": len(entries),
        "with_three_human_distractors": sum(
            1 for e in entries if len(e.get("choices_human", [])) == 3
        ),
        "with_nonempty_answer": sum(1 for e in entries if bool(e.get("answer"))),
        "with_nonempty_subfield": sum(1 for e in entries if bool(e.get("subfield"))),
    }
