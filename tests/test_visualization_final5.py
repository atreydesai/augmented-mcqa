import json
from pathlib import Path

from datasets import Dataset

from analysis.visualize import SETTING_RANDOM_BASELINES, collect_final5_results, plot_final5_pairwise
from data.final5_store import EVALUATED_RECORD_COLUMNS
from utils.constants import EVALUATED_STORE_MANIFEST, SETTING_SPECS
from utils.modeling import safe_name


GEN_RUN = "gen_test_run"
GEN_MODEL = "openai/gpt-5.2-2025-12-11"
EVAL_MODEL_A = "vllm/Qwen/Qwen3-4B-Instruct-2507"
EVAL_MODEL_B = "vllm/allenai/Olmo-3-7B-Instruct"


def _evaluated_row(
    *,
    sample_id: str,
    row_index: int,
    dataset_type: str,
    setting: str,
    status: str,
    correct: bool | None,
    prediction: str = "",
    prediction_type: str = "",
) -> dict[str, object]:
    row = {
        "id": sample_id,
        "question_id": sample_id,
        "dataset_type": dataset_type,
        "row_index": row_index,
        "sample_id": sample_id,
        "question": f"Question {row_index}",
        "answer": "correct",
        "category": "cat",
        "options": ["wrong", "correct"],
        "answer_index": 1,
        "choices_human": ["wrong"],
        "setting": setting,
        "generation_strategy": setting,
        "status": "success",
        "num_human": 1,
        "num_model": 0,
        "num_choices": 2,
        "human_distractors": ["wrong"],
        "model_distractors": [],
        "distractors": ["wrong"],
        "options_randomized": ["wrong", "correct"],
        "correct_answer_letter": "B",
        "traces": {"source": "augmented"},
        "evaluation_status": status,
        "evaluation_is_correct": correct,
        "evaluation_score": None if correct is None else float(correct),
        "evaluation_prediction": prediction,
        "evaluation_prediction_type": prediction_type,
        "evaluation_raw_output": '{"answer": "B"}' if prediction else "",
        "evaluation_prompt": "Answer the MCQ." if prediction else "",
        "evaluation_question_idx": row_index,
        "evaluation_log_path": f"/tmp/{sample_id}.eval" if prediction else "",
    }
    assert set(row) == set(EVALUATED_RECORD_COLUMNS)
    return row


def _write_evaluated_group(
    root: Path,
    *,
    gen_run: str = GEN_RUN,
    gen_model: str = GEN_MODEL,
    eval_model: str,
    rows_by_split: dict[tuple[str, str, str], list[dict[str, object]]],
):
    group_root = root / safe_name(gen_run) / safe_name(gen_model) / safe_name(eval_model)
    dataset_types = sorted({dataset for dataset, _setting, _mode in rows_by_split})
    settings = sorted({setting for _dataset, setting, _mode in rows_by_split})
    modes = sorted({mode for _dataset, _setting, mode in rows_by_split})
    manifest = {
        "schema_version": "evaluated_mcqa_setting_mode_records_v1",
        "storage_kind": "evaluated_setting_mode_records",
        "dataset_types": dataset_types,
        "settings": settings,
        "modes": modes,
        "generation_run_name": gen_run,
        "generation_model": gen_model,
        "evaluation_model": eval_model,
        "source_results_root": "/tmp/inspect",
    }
    group_root.mkdir(parents=True, exist_ok=True)
    (group_root / EVALUATED_STORE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    for (dataset, setting, mode), rows in rows_by_split.items():
        path = group_root / dataset / setting / mode
        path.parent.mkdir(parents=True, exist_ok=True)
        Dataset.from_list(rows).save_to_disk(str(path))


def test_collect_final5_results_reads_materialized_evaluated_datasets(tmp_path):
    root = tmp_path / "evaluated"
    _write_evaluated_group(
        root,
        eval_model=EVAL_MODEL_A,
        rows_by_split={
            ("arc_challenge", "human_from_scratch", "full_question"): [
                _evaluated_row(
                    sample_id="arc:0",
                    row_index=0,
                    dataset_type="arc_challenge",
                    setting="human_from_scratch",
                    status="success",
                    correct=True,
                    prediction="B",
                    prediction_type="G",
                ),
                _evaluated_row(
                    sample_id="arc:1",
                    row_index=1,
                    dataset_type="arc_challenge",
                    setting="human_from_scratch",
                    status="success",
                    correct=False,
                    prediction="A",
                    prediction_type="H",
                ),
            ],
            ("gpqa", "model_from_scratch", "choices_only"): [
                _evaluated_row(
                    sample_id="gpqa:0",
                    row_index=0,
                    dataset_type="gpqa",
                    setting="model_from_scratch",
                    status="success",
                    correct=True,
                    prediction="B",
                    prediction_type="G",
                )
            ],
        },
    )

    df = collect_final5_results(root)
    assert set(df["setting"]) == {"human_from_scratch", "model_from_scratch"}
    assert set(df["dataset"]) == {"arc_challenge", "gpqa"}
    assert set(df["status"]) == {"complete", "missing"}

    hfs_row = df[
        (df["eval_model"] == EVAL_MODEL_A)
        & (df["setting"] == "human_from_scratch")
        & (df["mode"] == "full_question")
    ].iloc[0]
    assert int(hfs_row["observed_total"]) == 2
    assert int(hfs_row["expected_total"]) == 2
    assert abs(float(hfs_row["accuracy"]) - 0.5) < 1e-9
    assert hfs_row["random_baseline"] == 1.0 / SETTING_SPECS["human_from_scratch"]["num_choices"]
    assert SETTING_RANDOM_BASELINES["human_from_scratch"] == 1.0 / SETTING_SPECS["human_from_scratch"]["num_choices"]


def test_collect_final5_results_marks_partial_and_missing_rows(tmp_path):
    root = tmp_path / "evaluated"
    _write_evaluated_group(
        root,
        eval_model=EVAL_MODEL_A,
        rows_by_split={
            ("arc_challenge", "human_from_scratch", "full_question"): [
                _evaluated_row(
                    sample_id="arc:0",
                    row_index=0,
                    dataset_type="arc_challenge",
                    setting="human_from_scratch",
                    status="success",
                    correct=True,
                    prediction="B",
                    prediction_type="G",
                ),
                _evaluated_row(
                    sample_id="arc:1",
                    row_index=1,
                    dataset_type="arc_challenge",
                    setting="human_from_scratch",
                    status="success",
                    correct=False,
                    prediction="A",
                    prediction_type="H",
                ),
            ]
        },
    )
    _write_evaluated_group(
        root,
        eval_model=EVAL_MODEL_B,
        rows_by_split={
            ("arc_challenge", "human_from_scratch", "full_question"): [
                _evaluated_row(
                    sample_id="arc:0",
                    row_index=0,
                    dataset_type="arc_challenge",
                    setting="human_from_scratch",
                    status="success",
                    correct=True,
                    prediction="B",
                    prediction_type="G",
                ),
                _evaluated_row(
                    sample_id="arc:1",
                    row_index=1,
                    dataset_type="arc_challenge",
                    setting="human_from_scratch",
                    status="missing",
                    correct=None,
                ),
            ]
        },
    )

    df = collect_final5_results(root)
    rows = df[
        (df["setting"] == "human_from_scratch")
        & (df["mode"] == "full_question")
        & (df["dataset"] == "arc_challenge")
    ].set_index("eval_model")

    complete_row = rows.loc[EVAL_MODEL_A]
    assert complete_row["status"] == "complete"
    assert int(complete_row["observed_total"]) == 2
    assert int(complete_row["expected_total"]) == 2

    partial_row = rows.loc[EVAL_MODEL_B]
    assert partial_row["status"] == "partial"
    assert int(partial_row["observed_total"]) == 1
    assert int(partial_row["expected_total"]) == 2
    assert int(partial_row["missing_samples"]) == 1
    assert abs(float(partial_row["coverage_fraction"]) - 0.5) < 1e-9

    missing_row = rows.loc["vllm/meta-llama/Llama-3.1-8B-Instruct"]
    assert missing_row["status"] == "missing"
    assert int(missing_row["observed_total"]) == 0
    assert int(missing_row["expected_total"]) == 2


def test_plot_final5_pairwise_writes_pairwise_distribution_and_failure_outputs(tmp_path):
    root = tmp_path / "evaluated"
    common_rows = {
        ("arc_challenge", "human_from_scratch", "full_question"): [
            _evaluated_row(
                sample_id="arc:0",
                row_index=0,
                dataset_type="arc_challenge",
                setting="human_from_scratch",
                status="success",
                correct=True,
                prediction="B",
                prediction_type="G",
            ),
            _evaluated_row(
                sample_id="arc:1",
                row_index=1,
                dataset_type="arc_challenge",
                setting="human_from_scratch",
                status="success",
                correct=False,
                prediction="A",
                prediction_type="H",
            ),
        ],
        ("arc_challenge", "model_from_scratch", "full_question"): [
            _evaluated_row(
                sample_id="arc:0",
                row_index=0,
                dataset_type="arc_challenge",
                setting="model_from_scratch",
                status="success",
                correct=False,
                prediction="A",
                prediction_type="M",
            ),
            _evaluated_row(
                sample_id="arc:1",
                row_index=1,
                dataset_type="arc_challenge",
                setting="model_from_scratch",
                status="missing",
                correct=None,
            ),
        ],
    }
    _write_evaluated_group(root, eval_model=EVAL_MODEL_A, rows_by_split=common_rows)

    output_dir = tmp_path / "plots"
    outputs = plot_final5_pairwise(root, output_dir, include_tables=True)
    output_names = {path.name for path in outputs}

    assert any(name.startswith("pairwise_") and name.endswith(".png") for name in output_names)
    assert "final5_results_summary.csv" in output_names
    assert "final5_missing_or_partial.csv" in output_names
    assert "final5_failed_questions.csv" in output_names
    assert "final5_missing_questions.csv" in output_names
    assert any(name.startswith("prediction_type_distribution_") and name.endswith(".png") for name in output_names)
    assert any(name.startswith("prediction_distribution_") and name.endswith(".png") for name in output_names)
