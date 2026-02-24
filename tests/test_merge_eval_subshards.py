import json
from pathlib import Path

from scripts.merge_eval_subshards import merge_config_root


def _write_partial(path: Path, *, shard_idx: int, shard_total: int, question_indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

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

    payload = {
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
        "results": results,
    }

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_merge_subshards_creates_canonical_results(tmp_path):
    config_root = tmp_path / "results" / "gpt" / "Qwen_Qwen3-4B-Instruct-2507" / "full_question" / "mmlu_pro" / "model_from_scratch"

    _write_partial(
        config_root / "_partials" / "entry_shard_0_of_3" / "results.json",
        shard_idx=0,
        shard_total=3,
        question_indices=[0, 3, 6],
    )
    _write_partial(
        config_root / "_partials" / "entry_shard_1_of_3" / "results.json",
        shard_idx=1,
        shard_total=3,
        question_indices=[1, 4, 7],
    )
    _write_partial(
        config_root / "_partials" / "entry_shard_2_of_3" / "results.json",
        shard_idx=2,
        shard_total=3,
        question_indices=[2, 5, 8],
    )

    out = merge_config_root(config_root, expected_entry_shards=3, strict=True)
    assert out["status"] == "merged"

    merged_path = config_root / "results.json"
    assert merged_path.exists()

    merged = json.loads(merged_path.read_text(encoding="utf-8"))
    merged_indices = [row["question_idx"] for row in merged["results"]]
    assert merged_indices == list(range(9))
    assert merged["summary"]["total"] == 9
    assert "merge_metadata" in merged


def test_merge_subshards_strict_detects_missing_shards(tmp_path):
    config_root = tmp_path / "results" / "gpt" / "Qwen_Qwen3-4B-Instruct-2507" / "full_question" / "gpqa" / "augment_model"

    _write_partial(
        config_root / "_partials" / "entry_shard_0_of_2" / "results.json",
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
