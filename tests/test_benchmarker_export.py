import json

import pytest
from datasets import Dataset, DatasetDict, load_from_disk

from data.benchmarker_export import export_benchmarker_items
from utils.constants import (
    AUGMENTED_STORE_MANIFEST,
    AUGMENTED_STORE_SCHEMA_VERSION,
    EVALUATED_STORE_MANIFEST,
    EVALUATED_STORE_SCHEMA_VERSION,
    SETTING_NAMES,
)


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

    for setting in setting_overrides:
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


def _write_collected_group(
    root,
    *,
    generation_run_name: str,
    generation_model: str,
    evaluation_model: str,
    rows_by_setting: dict[str, list[dict[str, object]]],
):
    root.mkdir(parents=True, exist_ok=True)
    (root / EVALUATED_STORE_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": EVALUATED_STORE_SCHEMA_VERSION,
                "storage_kind": "evaluated_setting_mode_records",
                "dataset_types": ["arc_challenge"],
                "settings": list(SETTING_NAMES),
                "modes": ["full_question"],
                "generation_run_name": generation_run_name,
                "generation_model": generation_model,
                "evaluation_model": evaluation_model,
                "source_results_root": "",
                "collection_state_file": "collected_state.json",
            }
        ),
        encoding="utf-8",
    )
    for setting in SETTING_NAMES:
        rows = rows_by_setting.get(setting, [])
        Dataset.from_list(rows).save_to_disk(str(root / "arc_challenge" / setting / "full_question"))


def test_export_benchmarker_items_end_to_end(tmp_path):
    dataset_path = tmp_path / "augmented_fixture"
    output_root = tmp_path / "benchmarker_items"
    _build_dataset(dataset_path)

    summary_path = export_benchmarker_items(dataset_path, output_root)
    export_dir = output_root / dataset_path.name

    assert summary_path == export_dir / "export_summary.json"
    assert export_dir.exists()

    expected_files = {"original.jsonl", *(f"{setting}.jsonl" for setting in SETTING_NAMES)}
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


def test_export_benchmarker_items_from_collected_serializes_rows_without_shared_support_filter(tmp_path):
    collected_root = tmp_path / "collected"

    common_row = {
        "id": "arc-1",
        "sample_id": "arc_challenge:arc-1",
        "question": "ARC question 1",
        "options": ["arc a", "arc b", "arc c", "arc d"],
        "answer_index": 2,
        "answer": "arc c",
        "choices_human": ["h1", "h2", "h3"],
        "options_randomized": ["arc c", "h1", "h2", "h3"],
        "correct_answer_letter": "A",
        "evaluation_status": "success",
    }
    source_only_row = {
        "id": "arc-2",
        "sample_id": "arc_challenge:arc-2",
        "question": "ARC question 2",
        "options": ["d1", "d2", "d3", "d4"],
        "answer_index": 1,
        "answer": "d2",
        "choices_human": ["x1", "x2", "x3"],
        "options_randomized": ["d2", "x1", "x2", "x3"],
        "correct_answer_letter": "A",
        "evaluation_status": "success",
    }

    source_group = collected_root / "run_a" / "gen_a" / "eval_a"
    peer_group = collected_root / "run_b" / "gen_b" / "eval_a"
    source_rows = {setting: [dict(common_row), dict(source_only_row)] for setting in SETTING_NAMES}
    peer_rows = {setting: [dict(common_row)] for setting in SETTING_NAMES}

    _write_collected_group(
        source_group,
        generation_run_name="run_a",
        generation_model="gen_a",
        evaluation_model="eval_a",
        rows_by_setting=source_rows,
    )
    _write_collected_group(
        peer_group,
        generation_run_name="run_b",
        generation_model="gen_b",
        evaluation_model="eval_a",
        rows_by_setting=peer_rows,
    )

    summary_path = export_benchmarker_items(source_group, tmp_path / "out")
    export_dir = summary_path.parent
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert export_dir.name == "run_a__gen_a"
    assert len(_read_jsonl(export_dir / "arc_challenge" / "original.jsonl")) == 2
    assert len(_read_jsonl(export_dir / "arc_challenge" / "augment_model.jsonl")) == 2
    assert summary["source_kind"] == "evaluated"
    assert summary["files"]["arc_challenge"]["augment_model"]["rows_written"] == 2
    assert summary["files"]["arc_challenge"]["augment_model"]["skipped_row_count"] == 0


def test_export_benchmarker_items_from_collected_only_skips_rows_that_are_structurally_invalid(tmp_path):
    collected_root = tmp_path / "collected"

    common_row = {
        "id": "arc-1",
        "sample_id": "arc_challenge:arc-1",
        "question": "ARC question 1",
        "options": ["arc a", "arc b", "arc c", "arc d"],
        "answer_index": 2,
        "answer": "arc c",
        "choices_human": ["h1", "h2", "h3"],
        "options_randomized": ["arc c", "h1", "h2", "h3"],
        "correct_answer_letter": "A",
        "evaluation_status": "success",
    }
    bad_setting_row = {
        "id": "arc-2",
        "sample_id": "arc_challenge:arc-2",
        "question": "ARC question 2",
        "options": ["b1", "b2", "b3", "b4"],
        "answer_index": 0,
        "answer": "b1",
        "choices_human": ["x1", "x2", "x3"],
        "options_randomized": ["b1", "x1", "x2", "x3"],
        "correct_answer_letter": "A",
        "evaluation_status": "success",
    }

    source_group = collected_root / "run_a" / "gen_a" / "eval_a"
    peer_group = collected_root / "run_b" / "gen_b" / "eval_a"

    source_rows = {setting: [dict(common_row), dict(bad_setting_row)] for setting in SETTING_NAMES}
    source_rows["augment_model"] = [
        dict(common_row),
        {
            **dict(bad_setting_row),
            "options_randomized": [],
            "correct_answer_letter": "",
        },
    ]
    peer_rows = {setting: [dict(common_row), dict(bad_setting_row)] for setting in SETTING_NAMES}

    _write_collected_group(
        source_group,
        generation_run_name="run_a",
        generation_model="gen_a",
        evaluation_model="eval_a",
        rows_by_setting=source_rows,
    )
    _write_collected_group(
        peer_group,
        generation_run_name="run_b",
        generation_model="gen_b",
        evaluation_model="eval_a",
        rows_by_setting=peer_rows,
    )

    summary_path = export_benchmarker_items(source_group, tmp_path / "out")
    export_dir = summary_path.parent
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(_read_jsonl(export_dir / "arc_challenge" / "original.jsonl")) == 2
    assert len(_read_jsonl(export_dir / "arc_challenge" / "human_from_scratch.jsonl")) == 2
    assert len(_read_jsonl(export_dir / "arc_challenge" / "model_from_scratch.jsonl")) == 2
    assert len(_read_jsonl(export_dir / "arc_challenge" / "augment_human.jsonl")) == 2
    assert len(_read_jsonl(export_dir / "arc_challenge" / "augment_model.jsonl")) == 1
    assert len(_read_jsonl(export_dir / "arc_challenge" / "augment_ablation.jsonl")) == 2
    assert summary["files"]["arc_challenge"]["original"]["skipped_row_count"] == 0
    assert summary["files"]["arc_challenge"]["augment_model"]["skipped_row_count"] == 1
