import json

from datasets import Dataset, DatasetDict

from data.benchmarker_export import export_benchmarker_items


def _choices(prefix: str, count: int) -> list[str]:
    return [f"{prefix}_{idx}" for idx in range(count)]


def _build_dataset(path):
    arc = Dataset.from_list(
        [
            {
                "id": "arc-1",
                "question": "ARC question 1",
                "options": ["arc a", "arc b", "arc c", "arc d"],
                "answer_index": 2,
                "answer_letter": "3",
                "human_from_scratch_options_randomized": ["h1", "h2", "h3", "h4"],
                "human_from_scratch_correct_answer_letter": "B",
                "model_from_scratch_options_randomized": ["m1", "m2", "m3", "m4"],
                "model_from_scratch_correct_answer_letter": "D",
                "augment_human_options_randomized": _choices("ah", 10),
                "augment_human_correct_answer_letter": "E",
                "augment_model_options_randomized": _choices("am", 10),
                "augment_model_correct_answer_letter": "F",
                "augment_ablation_options_randomized": _choices("aa", 10),
                "augment_ablation_correct_answer_letter": "G",
            }
        ]
    )

    mmlu = Dataset.from_list(
        [
            {
                "question_id": 101,
                "question": "MMLU question 1",
                "options": _choices("mmlu_opt", 10),
                "answer": "correct text should not be used",
                "answer_index": 5,
                "human_from_scratch_options_randomized": ["mh1", "mh2", "mh3", "mh4"],
                "human_from_scratch_correct_answer_letter": "A",
                "model_from_scratch_options_randomized": ["mm1", "mm2", "mm3", "mm4"],
                "model_from_scratch_correct_answer_letter": "C",
                "augment_human_options_randomized": _choices("mah", 10),
                "augment_human_correct_answer_letter": "B",
                "augment_model_options_randomized": _choices("mam", 10),
                "augment_model_correct_answer_letter": "H",
                "augment_ablation_options_randomized": _choices("maa", 10),
                "augment_ablation_correct_answer_letter": "J",
            },
            {
                "question_id": 102,
                "question": "MMLU question 2",
                "options": _choices("mmlu2_opt", 10),
                "answer": "another text answer",
                "answer_index": 1,
                "human_from_scratch_options_randomized": [],
                "human_from_scratch_correct_answer_letter": "",
                "model_from_scratch_options_randomized": [],
                "model_from_scratch_correct_answer_letter": "",
                "augment_human_options_randomized": [],
                "augment_human_correct_answer_letter": "",
                "augment_model_options_randomized": [],
                "augment_model_correct_answer_letter": "",
                "augment_ablation_options_randomized": [],
                "augment_ablation_correct_answer_letter": "",
            },
        ]
    )

    gpqa = Dataset.from_list(
        [
            {
                "id": "gpqa-1",
                "question": "GPQA question 1",
                "options": [],
                "answer": "gpqa gold",
                "choices_human": ["gpqa d1", "gpqa d2", "gpqa d3"],
                "human_from_scratch_options_randomized": ["gh1", "gh2", "gh3", "gh4"],
                "human_from_scratch_correct_answer_letter": "D",
                "model_from_scratch_options_randomized": ["gm1", "gm2", "gm3", "gm4"],
                "model_from_scratch_correct_answer_letter": "B",
                "augment_human_options_randomized": _choices("gah", 10),
                "augment_human_correct_answer_letter": "A",
                "augment_model_options_randomized": _choices("gam", 10),
                "augment_model_correct_answer_letter": "C",
                "augment_ablation_options_randomized": _choices("gaa", 10),
                "augment_ablation_correct_answer_letter": "I",
            },
            {
                "id": "gpqa-2",
                "question": "GPQA question 2",
                "options": [],
                "answer": "gpqa gold 2",
                "choices_human": ["gpqa2 d1", "gpqa2 d2", "gpqa2 d3"],
                "human_from_scratch_options_randomized": [],
                "human_from_scratch_correct_answer_letter": "",
                "model_from_scratch_options_randomized": [],
                "model_from_scratch_correct_answer_letter": "",
                "augment_human_options_randomized": [],
                "augment_human_correct_answer_letter": "",
                "augment_model_options_randomized": [],
                "augment_model_correct_answer_letter": "",
                "augment_ablation_options_randomized": [],
                "augment_ablation_correct_answer_letter": "",
            },
        ]
    )

    dataset = DatasetDict(
        {
            "arc_challenge": arc,
            "mmlu_pro": mmlu,
            "gpqa": gpqa,
        }
    )
    dataset.save_to_disk(str(path))


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_export_benchmarker_items_end_to_end(tmp_path):
    dataset_path = tmp_path / "final5_fixture"
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


def test_original_arc_exports_use_options_order_and_answer_index(tmp_path):
    dataset_path = tmp_path / "final5_fixture"
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
    dataset_path = tmp_path / "final5_fixture"
    _build_dataset(dataset_path)

    summary_path = export_benchmarker_items(dataset_path, tmp_path / "out")
    rows = _read_jsonl(summary_path.parent / "mmlu_pro" / "original.jsonl")

    assert rows[0]["question"] == "MMLU question 1"
    assert rows[0]["choices"] == _choices("mmlu_opt", 10)
    assert rows[0]["answer"] == "F"
    assert rows[0]["answer"] != "correct text should not be used"


def test_original_gpqa_reconstructs_choices_and_answer_a(tmp_path):
    dataset_path = tmp_path / "final5_fixture"
    _build_dataset(dataset_path)

    summary_path = export_benchmarker_items(dataset_path, tmp_path / "out")
    rows = _read_jsonl(summary_path.parent / "gpqa" / "original.jsonl")

    assert rows[0] == {
        "question": "GPQA question 1",
        "choices": ["gpqa gold", "gpqa d1", "gpqa d2", "gpqa d3"],
        "answer": "A",
    }


def test_generated_variant_uses_randomized_choices_and_correct_letter(tmp_path):
    dataset_path = tmp_path / "final5_fixture"
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


def test_missing_generated_rows_are_skipped_and_reported(tmp_path):
    dataset_path = tmp_path / "final5_fixture"
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
    assert mmlu_meta["skipped_row_count"] == 1
    assert mmlu_meta["skipped_rows"] == [
        {
            "row_index": 1,
            "identifier": "102",
            "reason": "missing choices in human_from_scratch_options_randomized",
        }
    ]

    assert gpqa_meta["rows_written"] == 1
    assert gpqa_meta["skipped_row_count"] == 1
    assert gpqa_meta["skipped_rows"] == [
        {
            "row_index": 1,
            "identifier": "gpqa-2",
            "reason": "missing choices in augment_model_options_randomized",
        }
    ]


def test_exported_jsonl_lines_have_only_expected_keys(tmp_path):
    dataset_path = tmp_path / "final5_fixture"
    _build_dataset(dataset_path)

    summary_path = export_benchmarker_items(dataset_path, tmp_path / "out")
    export_dir = summary_path.parent

    for path in export_dir.glob("*/*.jsonl"):
        rows = _read_jsonl(path)
        for row in rows:
            assert set(row.keys()) == {"question", "choices", "answer"}
