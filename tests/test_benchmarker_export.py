import json

import pytest
from datasets import Dataset, DatasetDict, load_from_disk

from data.benchmarker_export import export_benchmarker_items
from utils.constants import AUGMENTED_STORE_MANIFEST, AUGMENTED_STORE_SCHEMA_VERSION, SETTING_NAMES


def _choices(prefix: str, count: int) -> list[str]:
    return [f"{prefix}_{idx}" for idx in range(count)]


def _build_dataset(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / AUGMENTED_STORE_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": AUGMENTED_STORE_SCHEMA_VERSION,
                "storage_kind": "setting_records",
                "dataset_types": ["arc_challenge", "mmlu_pro", "gpqa"],
                "settings": list(SETTING_NAMES),
            }
        ),
        encoding="utf-8",
    )

    arc_row = {
        "id": "arc-1",
        "sample_id": "arc_challenge:arc-1",
        "question": "ARC question 1",
        "options": ["arc a", "arc b", "arc c", "arc d"],
        "answer_index": 2,
        "answer": "arc c",
    }
    mmlu_row = {
        "question_id": 101,
        "sample_id": "mmlu_pro:101",
        "question": "MMLU question 1",
        "options": _choices("mmlu_opt", 10),
        "answer": "correct text should not be used",
        "answer_index": 5,
    }
    gpqa_row = {
        "id": "gpqa-1",
        "sample_id": "gpqa:gpqa-1",
        "question": "GPQA question 1",
        "options": [],
        "answer": "gpqa gold",
        "choices_human": ["gpqa d1", "gpqa d2", "gpqa d3"],
    }

    setting_overrides = {
        "human_from_scratch": {
            "arc_challenge": (["h1", "h2", "h3", "h4"], "B"),
            "mmlu_pro": (["mh1", "mh2", "mh3", "mh4"], "A"),
            "gpqa": (["gh1", "gh2", "gh3", "gh4"], "D"),
        },
        "model_from_scratch": {
            "arc_challenge": (["m1", "m2", "m3", "m4"], "D"),
            "mmlu_pro": (["mm1", "mm2", "mm3", "mm4"], "C"),
            "gpqa": (["gm1", "gm2", "gm3", "gm4"], "B"),
        },
        "augment_human": {
            "arc_challenge": (_choices("ah", 10), "E"),
            "mmlu_pro": (_choices("mah", 10), "B"),
            "gpqa": (_choices("gah", 10), "A"),
        },
        "augment_model": {
            "arc_challenge": (_choices("am", 10), "F"),
            "mmlu_pro": (_choices("mam", 10), "H"),
            "gpqa": (_choices("gam", 10), "C"),
        },
        "augment_ablation": {
            "arc_challenge": (_choices("aa", 10), "G"),
            "mmlu_pro": (_choices("maa", 10), "J"),
            "gpqa": (_choices("gaa", 10), "I"),
        },
    }

    for setting in SETTING_NAMES:
        arc_choices, arc_answer = setting_overrides[setting]["arc_challenge"]
        mmlu_choices, mmlu_answer = setting_overrides[setting]["mmlu_pro"]
        gpqa_choices, gpqa_answer = setting_overrides[setting]["gpqa"]

        Dataset.from_list(
            [{**arc_row, "options_randomized": arc_choices, "correct_answer_letter": arc_answer}]
        ).save_to_disk(str(path / "arc_challenge" / setting))
        Dataset.from_list(
            [{**mmlu_row, "options_randomized": mmlu_choices, "correct_answer_letter": mmlu_answer}]
        ).save_to_disk(str(path / "mmlu_pro" / setting))
        Dataset.from_list(
            [{**gpqa_row, "options_randomized": gpqa_choices, "correct_answer_letter": gpqa_answer}]
        ).save_to_disk(str(path / "gpqa" / setting))


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_export_benchmarker_items_end_to_end(tmp_path):
    dataset_path = tmp_path / "augmented_fixture"
    output_root = tmp_path / "benchmarker_items"
    _build_dataset(dataset_path)

    summary_path = export_benchmarker_items(dataset_path, output_root)
    export_dir = output_root / dataset_path.name

    assert summary_path == export_dir / "export_summary.json"
    assert export_dir.exists()

    expected_files = {
        "original.jsonl",
        "human_from_scratch.jsonl",
        "model_from_scratch.jsonl",
        "augment_human.jsonl",
        "augment_model.jsonl",
        "augment_ablation.jsonl",
    }
    for split_name in ("arc_challenge", "mmlu_pro", "gpqa"):
        split_dir = export_dir / split_name
        assert split_dir.exists()
        assert {path.name for path in split_dir.iterdir()} == expected_files


def test_export_benchmarker_items_rejects_manifest_path(tmp_path):
    dataset_path = tmp_path / "augmented_fixture"
    output_root = tmp_path / "benchmarker_items"
    _build_dataset(dataset_path)

    with pytest.raises(ValueError, match="root directory, not a manifest file"):
        export_benchmarker_items(dataset_path / AUGMENTED_STORE_MANIFEST, output_root)


def test_export_benchmarker_items_rejects_unsupported_cache_layout(tmp_path):
    dataset_path = tmp_path / "legacy_fixture"
    DatasetDict(
        {
            "arc_challenge": Dataset.from_list([{"sample_id": "arc_challenge:arc-1"}]),
            "mmlu_pro": Dataset.from_list([]),
            "gpqa": Dataset.from_list([]),
        }
    ).save_to_disk(str(dataset_path))

    with pytest.raises(ValueError, match="unsupported augmented cache layout"):
        export_benchmarker_items(dataset_path, tmp_path / "out")


def test_original_arc_exports_use_options_order_and_answer_index(tmp_path):
    dataset_path = tmp_path / "augmented_fixture"
    _build_dataset(dataset_path)

    summary_path = export_benchmarker_items(dataset_path, tmp_path / "out")
    export_dir = summary_path.parent

    rows = _read_jsonl(export_dir / "arc_challenge" / "original.jsonl")
    assert rows == [
        {
            "question": "ARC question 1",
            "choices": ["arc a", "arc b", "arc c", "arc d"],
            "answer": "C",
        }
    ]


def test_original_mmlu_uses_answer_index_not_answer_text(tmp_path):
    dataset_path = tmp_path / "augmented_fixture"
    _build_dataset(dataset_path)

    summary_path = export_benchmarker_items(dataset_path, tmp_path / "out")
    rows = _read_jsonl(summary_path.parent / "mmlu_pro" / "original.jsonl")

    assert rows[0]["question"] == "MMLU question 1"
    assert rows[0]["choices"] == _choices("mmlu_opt", 10)
    assert rows[0]["answer"] == "F"
    assert rows[0]["answer"] != "correct text should not be used"


def test_original_gpqa_reconstructs_choices_and_answer_a(tmp_path):
    dataset_path = tmp_path / "augmented_fixture"
    _build_dataset(dataset_path)

    summary_path = export_benchmarker_items(dataset_path, tmp_path / "out")
    rows = _read_jsonl(summary_path.parent / "gpqa" / "original.jsonl")

    assert rows[0] == {
        "question": "GPQA question 1",
        "choices": ["gpqa gold", "gpqa d1", "gpqa d2", "gpqa d3"],
        "answer": "A",
    }


def test_generated_variant_uses_randomized_choices_and_correct_letter(tmp_path):
    dataset_path = tmp_path / "augmented_fixture"
    _build_dataset(dataset_path)

    summary_path = export_benchmarker_items(dataset_path, tmp_path / "out")
    rows = _read_jsonl(summary_path.parent / "arc_challenge" / "augment_model.jsonl")

    assert rows == [
        {
            "question": "ARC question 1",
            "choices": _choices("am", 10),
            "answer": "F",
        }
    ]


def test_generated_variant_rejects_multi_character_answer_letter(tmp_path):
    dataset_path = tmp_path / "augmented_fixture"
    _build_dataset(dataset_path)

    rows = load_from_disk(str(dataset_path / "arc_challenge" / "augment_model"))
    bad_row = dict(rows[0])
    bad_row["correct_answer_letter"] = "AB"
    Dataset.from_list([bad_row]).save_to_disk(str(dataset_path / "arc_challenge" / "augment_model"))

    summary_path = export_benchmarker_items(dataset_path, tmp_path / "out")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert _read_jsonl(summary_path.parent / "arc_challenge" / "augment_model.jsonl") == []
    assert summary["files"]["arc_challenge"]["augment_model"]["skipped_row_count"] == 1
    assert summary["files"]["arc_challenge"]["augment_model"]["skipped_rows"][0]["reason"] == (
        "invalid answer letter in correct_answer_letter"
    )


def test_missing_generated_rows_are_skipped_and_reported(tmp_path):
    dataset_path = tmp_path / "augmented_fixture"
    _build_dataset(dataset_path)

    summary_path = export_benchmarker_items(dataset_path, tmp_path / "out")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    mmlu_rows = _read_jsonl(summary_path.parent / "mmlu_pro" / "human_from_scratch.jsonl")
    gpqa_rows = _read_jsonl(summary_path.parent / "gpqa" / "augment_model.jsonl")

    assert len(mmlu_rows) == 1
    assert len(gpqa_rows) == 1

    mmlu_meta = summary["files"]["mmlu_pro"]["human_from_scratch"]
    gpqa_meta = summary["files"]["gpqa"]["augment_model"]

    assert mmlu_meta["rows_written"] == 1
    assert mmlu_meta["skipped_row_count"] == 0
    assert mmlu_meta["skipped_rows"] == []

    assert gpqa_meta["rows_written"] == 1
    assert gpqa_meta["skipped_row_count"] == 0
    assert gpqa_meta["skipped_rows"] == []


def test_exported_jsonl_lines_have_only_expected_keys(tmp_path):
    dataset_path = tmp_path / "augmented_fixture"
    _build_dataset(dataset_path)

    summary_path = export_benchmarker_items(dataset_path, tmp_path / "out")
    export_dir = summary_path.parent

    for path in export_dir.glob("*/*.jsonl"):
        rows = _read_jsonl(path)
        for row in rows:
            assert set(row.keys()) == {"question", "choices", "answer"}
