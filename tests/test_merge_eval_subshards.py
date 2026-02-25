import json
from pathlib import Path

from datasets import Dataset, load_from_disk

from scripts.merge_eval_subshards import merge_config_root


def _write_partial(rows_path: Path, *, shard_idx: int, shard_total: int, question_indices: list[int]) -> None:
    rows_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for idx in question_indices:
        results.append(
            {
                "question_idx": idx,
                "question": f"Q{idx}",
                "is_correct": idx % 2 == 0,
                "prediction_type": "G" if idx % 2 == 0 else "H",
                "category": "cat",
            }
        )

    Dataset.from_list(results).save_to_disk(str(rows_path))

    summary_payload = {
        "config": {"name": "cfg", "entry_shards": shard_total, "entry_shard_index": shard_idx},
        "summary": {
            "total": len(results),
            "correct": sum(1 for r in results if r["is_correct"]),
            "accuracy": 0.0,
            "attempted_entries": len(results),
            "successful_entries": len(results),
            "failed_entries": 0,
            "entry_failure_count": 0,
            "behavioral_counts": {"G": 0, "H": 0, "M": 0, "?": 0},
            "accuracy_by_category": {"cat": 0.0},
        },
        "timing": {"start": f"2026-01-01T00:00:0{shard_idx}Z", "end": f"2026-01-01T00:00:1{shard_idx}Z"},
        "entry_failures": [],
        "rows_path": str(rows_path),
    }

    (rows_path.parent / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


def test_merge_subshards_creates_canonical_results(tmp_path):
    config_root = tmp_path / "results" / "gpt" / "Qwen_Qwen3-4B-Instruct-2507" / "full_question" / "mmlu_pro" / "model_from_scratch"

    _write_partial(
        config_root / "_partials" / "entry_shard_0_of_3" / "rows",
        shard_idx=0,
        shard_total=3,
        question_indices=[0, 3, 6],
    )
    _write_partial(
        config_root / "_partials" / "entry_shard_1_of_3" / "rows",
        shard_idx=1,
        shard_total=3,
        question_indices=[1, 4, 7],
    )
    _write_partial(
        config_root / "_partials" / "entry_shard_2_of_3" / "rows",
        shard_idx=2,
        shard_total=3,
        question_indices=[2, 5, 8],
    )

    out = merge_config_root(config_root, expected_entry_shards=3, strict=True)
    assert out["status"] == "merged"

    merged_summary_path = config_root / "summary.json"
    merged_rows_path = config_root / "rows"
    merged_meta_path = config_root / "merge_metadata.json"
    assert merged_summary_path.exists()
    assert merged_rows_path.exists()
    assert merged_meta_path.exists()

    merged_rows = load_from_disk(str(merged_rows_path))
    merged_indices = [int(row["question_idx"]) for row in merged_rows]
    assert merged_indices == list(range(9))
    merged_summary = json.loads(merged_summary_path.read_text(encoding="utf-8"))
    assert merged_summary["summary"]["total"] == 9


def test_merge_subshards_strict_detects_missing_shards(tmp_path):
    config_root = tmp_path / "results" / "gpt" / "Qwen_Qwen3-4B-Instruct-2507" / "full_question" / "gpqa" / "augment_model"

    _write_partial(
        config_root / "_partials" / "entry_shard_0_of_2" / "rows",
        shard_idx=0,
        shard_total=2,
        question_indices=[0, 2],
    )

    try:
        merge_config_root(config_root, expected_entry_shards=2, strict=True)
    except ValueError as exc:
        assert "missing entry shards" in str(exc)
    else:
        raise AssertionError("Expected strict merge failure")
