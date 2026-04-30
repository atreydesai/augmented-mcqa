from argparse import Namespace
import os
from pathlib import Path

import json

from datasets import Dataset, load_from_disk
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, scorer
from inspect_ai.solver import solver

import cli.app as app_main
from data.store import (
    AUGMENTED_RECORD_COLUMNS,
    EVALUATED_RECORD_COLUMNS,
    materialize_evaluated_datasets,
)
from utils.constants import AUGMENTED_STORE_MANIFEST, COLLECTED_STATE_FILENAME
from utils.modeling import safe_name


GEN_RUN_NAME = "gen_test_run"
GEN_MODEL = "openai/gpt-5.2-2025-12-11"
EVAL_MODEL = "vllm/Qwen/Qwen3-4B-Instruct-2507"
SAMPLE_ID = "arc_challenge:test-0"


@solver
def _solver():
    async def solve(state, generate):  # noqa: ANN001
        state.output.completion = '{"answer": "B"}'
        return state

    return solve


@scorer(metrics=[])
def _scorer():
    async def score(state, target):  # noqa: ANN001
        return Score(
            value=1.0,
            metadata={
                "sample_id": SAMPLE_ID,
                "dataset_type": "arc_challenge",
                "question_idx": 0,
                "setting": "human_from_scratch",
                "mode": "full_question",
                "prediction": "B",
                "prediction_type": "H",
                "prompt": "Answer the MCQ.",
                "raw_output": '{"answer": "B"}',
                "gold_answer_letter": "B",
                "gold_index": 1,
                "selected_human_distractors": ["coal"],
                "selected_model_distractors": [],
                "human_option_indices": [0],
                "model_option_indices": [],
                "category": "science",
            },
        )

    return score


def _write_eval_log(root: Path):
    xdg_data_home = root.parent / ".xdg"
    xdg_data_home.mkdir(parents=True, exist_ok=True)
    original_xdg_data_home = os.environ.get("XDG_DATA_HOME")
    os.environ["XDG_DATA_HOME"] = str(xdg_data_home)
    try:
        eval(
            Task(
                name="eval_hfs_full",
                dataset=MemoryDataset(
                    [
                        Sample(
                            input="Which resource is renewable?",
                            choices=["coal", "trees"],
                            target="B",
                            id=SAMPLE_ID,
                            metadata={
                                "sample_id": SAMPLE_ID,
                                "dataset_type": "arc_challenge",
                                "row_index": 0,
                                "question": "Which resource is renewable?",
                                "category": "science",
                                "setting": "human_from_scratch",
                                "mode": "full_question",
                                "gold_answer": "trees",
                                "gold_index": 1,
                                "selected_human_distractors": ["coal"],
                                "selected_model_distractors": [],
                                "human_option_indices": [0],
                                "model_option_indices": [],
                                "evaluation": {
                                    "prompt": "Answer the MCQ.",
                                    "raw_output": '{"answer": "B"}',
                                    "prediction": "B",
                                },
                            },
                        )
                    ]
                ),
                solver=_solver(),
                scorer=_scorer(),
                metadata={
                    "kind": "evaluation",
                    "generation_run_name": GEN_RUN_NAME,
                    "generation_model": GEN_MODEL,
                    "evaluation_model": EVAL_MODEL,
                    "setting": "human_from_scratch",
                    "mode": "full_question",
                    "question_start": 0,
                    "question_end": 1,
                    "shard_count": 1,
                    "shard_index": 0,
                    "shard_strategy": "contiguous",
                    "slice_ref": "slice-0",
                },
            ),
            log_dir=str(root),
            display="none",
        )
    finally:
        if original_xdg_data_home is None:
            os.environ.pop("XDG_DATA_HOME", None)
        else:
            os.environ["XDG_DATA_HOME"] = original_xdg_data_home


def _write_augmented_predecessor(root: Path):
    row = {
        "id": SAMPLE_ID,
        "question_id": SAMPLE_ID,
        "dataset_type": "arc_challenge",
        "row_index": 0,
        "sample_id": SAMPLE_ID,
        "question": "Which resource is renewable?",
        "answer": "trees",
        "category": "science",
        "options": ["coal", "trees"],
        "answer_index": 1,
        "choices_human": ["coal"],
        "setting": "human_from_scratch",
        "generation_strategy": "human_from_scratch",
        "num_human": 1,
        "num_model": 0,
        "num_choices": 2,
        "human_distractors": ["coal"],
        "model_distractors": [],
        "distractors": ["coal"],
        "options_randomized": ["coal", "trees"],
        "correct_answer_letter": "B",
        "traces": {"source": "augmented"},
    }
    assert set(row) == set(AUGMENTED_RECORD_COLUMNS)
    store_root = root / safe_name(GEN_RUN_NAME) / safe_name(GEN_MODEL)
    store_path = store_root / "arc_challenge" / "human_from_scratch"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    (store_root / AUGMENTED_STORE_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": "augmented_mcqa_setting_records_v2",
                "storage_kind": "setting_records",
                "dataset_types": ["arc_challenge"],
                "settings": ["human_from_scratch"],
            }
        ),
        encoding="utf-8",
    )
    Dataset.from_list([row]).save_to_disk(str(store_path))


def _write_support_manifest(root: Path, *, sample_ids: list[str] | None = None):
    manifest_root = root / safe_name(GEN_RUN_NAME) / safe_name(GEN_MODEL)
    manifest_root.mkdir(parents=True, exist_ok=True)
    (manifest_root / "support_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "augmented_mcqa_support_set_v1",
                "storage_kind": "generation_support_set",
                "generation_run_name": GEN_RUN_NAME,
                "generation_model": GEN_MODEL,
                "dataset_types": ["arc_challenge"],
                "settings": ["human_from_scratch", "model_from_scratch", "augment_human", "augment_model", "augment_ablation"],
                "eligible_sample_ids_by_dataset": {"arc_challenge": list(sample_ids or [SAMPLE_ID])},
                "candidate_counts_by_dataset": {"arc_challenge": 1},
                "eligible_counts_by_dataset": {"arc_challenge": len(sample_ids or [SAMPLE_ID])},
                "excluded_counts_by_dataset": {"arc_challenge": 0},
                "excluded_counts_by_setting": {
                    "arc_challenge": {
                        "human_from_scratch": 0,
                        "model_from_scratch": 0,
                        "augment_human": 0,
                        "augment_model": 0,
                        "augment_ablation": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_materialize_evaluated_datasets_preserves_augmented_columns_and_adds_eval_fields(tmp_path):
    eval_root = tmp_path / "inspect"
    augmented_root = tmp_path / "augmented"
    support_root = tmp_path / "support_sets"
    output_root = tmp_path / "evaluated"
    _write_eval_log(eval_root)
    _write_augmented_predecessor(augmented_root)
    _write_support_manifest(support_root)

    outputs = materialize_evaluated_datasets(eval_root, output_root, augmented_root=augmented_root, support_root=support_root)
    assert len(outputs) == 1

    dataset_path = (
        output_root
        / safe_name(GEN_RUN_NAME)
        / safe_name(GEN_MODEL)
        / safe_name(EVAL_MODEL)
        / "arc_challenge"
        / "human_from_scratch"
        / "full_question"
    )
    rows = list(load_from_disk(str(dataset_path)))
    assert len(rows) == 1
    row = dict(rows[0])

    for column in AUGMENTED_RECORD_COLUMNS:
        assert column in row
    for column in EVALUATED_RECORD_COLUMNS:
        assert column in row

    assert row["traces"] == {"source": "augmented"}
    assert row["evaluation_used_random_fallback"] is False
    assert row["evaluation_is_correct"] is True
    assert row["evaluation_prediction"] == "B"
    assert row["evaluation_prompt"] == "Answer the MCQ."
    assert row["evaluation_raw_output"] == '{"answer": "B"}'
    assert row["evaluation_question_idx"] == 0
    assert row["evaluation_log_path"].endswith(".eval")


def test_materialize_evaluated_datasets_falls_back_when_augmented_row_is_missing(tmp_path):
    eval_root = tmp_path / "inspect"
    support_root = tmp_path / "support_sets"
    output_root = tmp_path / "evaluated"
    _write_eval_log(eval_root)
    _write_support_manifest(support_root)

    outputs = materialize_evaluated_datasets(
        eval_root,
        output_root,
        augmented_root=tmp_path / "augmented",
        support_root=support_root,
    )
    assert len(outputs) == 1

    dataset_path = (
        output_root
        / safe_name(GEN_RUN_NAME)
        / safe_name(GEN_MODEL)
        / safe_name(EVAL_MODEL)
        / "arc_challenge"
        / "human_from_scratch"
        / "full_question"
    )
    rows = list(load_from_disk(str(dataset_path)))
    assert len(rows) == 1
    row = dict(rows[0])

    assert row["sample_id"] == SAMPLE_ID
    assert row["evaluation_used_random_fallback"] is False
    assert row["evaluation_is_correct"] is True
    assert row["evaluation_prediction"] == "B"


def test_materialize_evaluated_fallback_prefers_full_input_over_truncated_metadata_question(tmp_path, monkeypatch):
    support_root = tmp_path / "support_sets"
    output_root = tmp_path / "evaluated"
    _write_support_manifest(support_root)
    full_question = "This question continues with the important final clause asking what the court should decide."
    truncated_question = "This question continues with..."
    payload = {
        "status": "success",
        "metadata": {
            "kind": "evaluation",
            "generation_run_name": GEN_RUN_NAME,
            "generation_model": GEN_MODEL,
            "evaluation_model": EVAL_MODEL,
            "setting": "human_from_scratch",
            "mode": "full_question",
        },
        "summaries": [
            {
                "id": SAMPLE_ID,
                "input": full_question,
                "choices": ["coal", "trees"],
                "target": "B",
                "metadata": {
                    "sample_id": SAMPLE_ID,
                    "dataset_type": "arc_challenge",
                    "row_index": 0,
                    "question": truncated_question,
                    "gold_answer": "trees",
                    "choices_human": ["coal"],
                },
                "scores": {
                    "augmented_mcqa_eval": {
                        "value": 1.0,
                        "metadata": {
                            "sample_id": SAMPLE_ID,
                            "dataset_type": "arc_challenge",
                            "question_idx": 0,
                            "setting": "human_from_scratch",
                            "mode": "full_question",
                            "prediction": "B",
                            "gold_answer_letter": "B",
                            "gold_index": 1,
                            "selected_human_distractors": ["coal"],
                            "selected_model_distractors": [],
                            "status": "success",
                        },
                    }
                },
            }
        ],
    }
    monkeypatch.setattr(
        "data.store.iter_log_payloads",
        lambda *args, **kwargs: [(tmp_path / "sample.eval", payload)],
    )

    outputs = materialize_evaluated_datasets(
        tmp_path / "inspect",
        output_root,
        augmented_root=tmp_path / "augmented",
        support_root=support_root,
    )

    assert len(outputs) == 1
    dataset_path = (
        output_root
        / safe_name(GEN_RUN_NAME)
        / safe_name(GEN_MODEL)
        / safe_name(EVAL_MODEL)
        / "arc_challenge"
        / "human_from_scratch"
        / "full_question"
    )
    rows = list(load_from_disk(str(dataset_path)))
    assert len(rows) == 1
    assert rows[0]["question"] == full_question


def test_materialize_evaluated_datasets_can_write_missing_rows_without_eval_logs(tmp_path):
    eval_root = tmp_path / "inspect"
    augmented_root = tmp_path / "augmented"
    support_root = tmp_path / "support_sets"
    output_root = tmp_path / "collected"
    _write_augmented_predecessor(augmented_root)
    _write_support_manifest(support_root)

    outputs = materialize_evaluated_datasets(
        eval_root,
        output_root,
        augmented_root=augmented_root / safe_name(GEN_RUN_NAME) / safe_name(GEN_MODEL),
        support_root=support_root,
        expected_dataset_types=["arc_challenge"],
        expected_settings=["human_from_scratch"],
        expected_modes=["full_question"],
        generation_run_name=GEN_RUN_NAME,
        generation_model=GEN_MODEL,
        evaluation_model=EVAL_MODEL,
    )

    assert len(outputs) == 1
    dataset_path = (
        output_root
        / safe_name(GEN_RUN_NAME)
        / safe_name(GEN_MODEL)
        / safe_name(EVAL_MODEL)
        / "arc_challenge"
        / "human_from_scratch"
        / "full_question"
    )
    rows = list(load_from_disk(str(dataset_path)))
    assert len(rows) == 1
    row = dict(rows[0])

    assert row["sample_id"] == SAMPLE_ID
    assert row["evaluation_status"] == "missing"
    assert row["evaluation_used_random_fallback"] is True
    assert row["evaluation_prediction"] in {"A", "B"}
    assert row["evaluation_log_path"] == ""


def test_materialize_evaluated_datasets_does_not_return_unrelated_group_from_shared_output_root(tmp_path):
    eval_root = tmp_path / "inspect"
    augmented_root = tmp_path / "augmented"
    support_root = tmp_path / "support_sets"
    output_root = tmp_path / "collected"
    unrelated_group = output_root / "other-run" / "other-model" / "other-eval"
    unrelated_group.mkdir(parents=True, exist_ok=True)
    (unrelated_group / "evaluated_manifest.json").write_text("{}", encoding="utf-8")
    os.utime(unrelated_group / "evaluated_manifest.json", (4_000_000_000, 4_000_000_000))
    _write_eval_log(eval_root)
    _write_augmented_predecessor(augmented_root)
    _write_support_manifest(support_root)

    outputs = materialize_evaluated_datasets(eval_root, output_root, augmented_root=augmented_root, support_root=support_root)

    assert outputs == [output_root / safe_name(GEN_RUN_NAME) / safe_name(GEN_MODEL) / safe_name(EVAL_MODEL)]


def test_run_analyze_requires_existing_collected_root(tmp_path):
    collected_root = tmp_path / "collected"
    plots_root = tmp_path / "plots"
    table_output = plots_root / "tables" / "summary.csv"

    args = Namespace(
        output_dir=str(plots_root),
        table_output=str(table_output),
        skip_tables=False,
        collected_root=str(collected_root),
    )

    rc = app_main._run_analyze(args)

    assert rc == 1
    assert not collected_root.exists()
    assert not table_output.exists()


def test_materialize_evaluated_datasets_prefers_augmented_eval_score_when_multiple_scores(tmp_path, monkeypatch):
    augmented_root = tmp_path / "augmented"
    support_root = tmp_path / "support_sets"
    output_root = tmp_path / "evaluated"
    _write_augmented_predecessor(augmented_root)
    _write_support_manifest(support_root)

    payload = {
        "status": "success",
        "metadata": {
            "kind": "evaluation",
            "generation_run_name": GEN_RUN_NAME,
            "generation_model": GEN_MODEL,
            "evaluation_model": EVAL_MODEL,
            "setting": "human_from_scratch",
            "mode": "full_question",
        },
        "summaries": [
            {
                "id": SAMPLE_ID,
                "metadata": {
                    "sample_id": SAMPLE_ID,
                    "dataset_type": "arc_challenge",
                    "row_index": 0,
                    "evaluation": {
                        "prompt": "Answer the MCQ.",
                        "raw_output": '{"answer": "B"}',
                        "prediction": "B",
                    },
                },
                "scores": {
                    "other_metric": {
                        "value": 0.0,
                        "metadata": {
                            "sample_id": "wrong",
                            "dataset_type": "",
                            "question_idx": 99,
                            "setting": "augment_model",
                            "mode": "choices_only",
                            "prediction": "A",
                            "prediction_type": "M",
                            "gold_answer_letter": "A",
                        },
                    },
                    "augmented_mcqa_eval": {
                        "value": 1.0,
                        "metadata": {
                            "sample_id": SAMPLE_ID,
                            "dataset_type": "arc_challenge",
                            "question_idx": 0,
                            "setting": "human_from_scratch",
                            "mode": "full_question",
                            "prediction": "B",
                            "prediction_type": "H",
                            "prompt": "Answer the MCQ.",
                            "raw_output": '{"answer": "B"}',
                            "gold_answer_letter": "B",
                            "category": "science",
                            "status": "success",
                        },
                    },
                },
            }
        ],
    }

    monkeypatch.setattr(
        "data.store.iter_log_payloads",
        lambda *args, **kwargs: [(tmp_path / "sample.eval", payload)],
    )

    outputs = materialize_evaluated_datasets(
        tmp_path / "inspect",
        output_root,
        augmented_root=augmented_root,
        support_root=support_root,
    )

    assert len(outputs) == 1
    dataset_path = (
        output_root
        / safe_name(GEN_RUN_NAME)
        / safe_name(GEN_MODEL)
        / safe_name(EVAL_MODEL)
        / "arc_challenge"
        / "human_from_scratch"
        / "full_question"
    )
    rows = list(load_from_disk(str(dataset_path)))
    assert len(rows) == 1
    row = dict(rows[0])
    assert row["sample_id"] == SAMPLE_ID
    assert row["evaluation_is_correct"] is True
    assert row["evaluation_prediction"] == "B"


def test_run_collect_evaluated_writes_collected_state(tmp_path):
    eval_root = tmp_path / "inspect"
    augmented_root = tmp_path / "augmented"
    support_root = tmp_path / "support_sets"
    collected_root = tmp_path / "collected"
    _write_eval_log(eval_root)
    _write_augmented_predecessor(augmented_root)
    _write_support_manifest(support_root)

    rc = app_main.main(
        [
            "build-collected-dataset",
            "--run-name",
            "eval-run",
            "--generator-run-name",
            GEN_RUN_NAME,
            "--generator-model",
            GEN_MODEL,
            "--model",
            EVAL_MODEL,
            "--evaluation-log-root",
            str(eval_root),
            "--augmented-dataset",
            str(augmented_root / safe_name(GEN_RUN_NAME) / safe_name(GEN_MODEL)),
            "--collected-root",
            str(collected_root),
            "--support-root",
            str(support_root),
            "--dataset-types",
            "arc_challenge",
            "--settings",
            "human_from_scratch",
            "--modes",
            "full_question",
        ]
    )

    assert rc == 0
    group_root = collected_root / safe_name(GEN_RUN_NAME) / safe_name(GEN_MODEL) / safe_name(EVAL_MODEL)
    state_path = group_root / COLLECTED_STATE_FILENAME
    assert state_path.exists()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["generation_run_name"] == GEN_RUN_NAME
    assert payload["generation_model"] == GEN_MODEL
    assert payload["evaluation_model"] == EVAL_MODEL
    assert payload["slice_count"] == 1
    assert sum(payload["status_counts"].values()) == 1


def test_materialize_collected_evaluation_writes_local_planned_state(tmp_path):
    eval_root = tmp_path / "inspect"
    augmented_root = tmp_path / "augmented"
    support_root = tmp_path / "support_sets"
    collected_root = tmp_path / "collected"
    _write_eval_log(eval_root)
    _write_augmented_predecessor(augmented_root)
    _write_support_manifest(support_root)

    outputs = app_main._materialize_collected_evaluation(
        run_name="eval-run",
        evaluation_log_root=eval_root,
        collected_root=collected_root,
        augmented_dataset=augmented_root,
        support_root=support_root,
        generation_run_name=GEN_RUN_NAME,
        generation_model=GEN_MODEL,
        evaluation_model=EVAL_MODEL,
        dataset_types=["arc_challenge"],
        settings=["human_from_scratch"],
        modes=["full_question"],
        planned_tasks=[
            {
                "slice_ref": "slice-0",
                "stage": "evaluate",
                "run_name": "eval-run",
                "model": EVAL_MODEL,
                "dataset_type": "arc_challenge",
                "dataset_types": ["arc_challenge"],
                "setting": "human_from_scratch",
                "mode": "full_question",
                "task_slug": "eval-task",
                "question_start": 0,
                "question_end": 1,
                "state_dependency_refs": [],
                "submit_dependency_refs": [],
                "force": False,
                "generation_run_name": GEN_RUN_NAME,
                "generation_model": GEN_MODEL,
            }
        ],
    )

    assert len(outputs) == 1
    payload = json.loads((outputs[0] / COLLECTED_STATE_FILENAME).read_text(encoding="utf-8"))
    assert payload["status_counts"] == {"current": 1}
    assert payload["slice_count"] == 1
    assert payload["slices"][0]["slice_ref"] == "slice-0"
    assert payload["slices"][0]["dataset_type"] == "arc_challenge"
    assert payload["slices"][0]["setting"] == "human_from_scratch"
    assert payload["slices"][0]["mode"] == "full_question"


def test_run_evaluate_materializes_collected_dataset(tmp_path, monkeypatch):
    collected_root = tmp_path / "collected"
    augmented_root = tmp_path / "augmented" / safe_name(GEN_RUN_NAME) / safe_name(GEN_MODEL)
    augmented_root.mkdir(parents=True, exist_ok=True)
    log_root = tmp_path / "logs"
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        app_main,
        "_resolved_generation_artifacts",
        lambda **kwargs: (GEN_MODEL, tmp_path / "generation-logs", augmented_root),
    )
    monkeypatch.setattr(
        app_main,
        "_combined_support_ids_for_run",
        lambda **kwargs: {"arc_challenge": {SAMPLE_ID}},
    )
    monkeypatch.setattr(
        app_main,
        "_combined_support_ids_for_run",
        lambda **kwargs: {"arc_challenge": {SAMPLE_ID}},
    )
    monkeypatch.setattr(
        app_main,
        "_combined_support_ids_for_run",
        lambda **kwargs: {"arc_challenge": {SAMPLE_ID}},
    )
    monkeypatch.setattr(
        app_main,
        "inspect_eval",
        lambda tasks, model, log_dir, args: ["ok"],
    )
    monkeypatch.setattr(app_main, "evaluation_logs_completed", lambda logs: True)
    monkeypatch.setattr(
        "tasks.build_evaluation_tasks",
        lambda **kwargs: [
            Namespace(
                metadata={
                    "run_name": "eval-run",
                    "generation_run_name": GEN_RUN_NAME,
                    "generation_model": GEN_MODEL,
                    "evaluation_model": EVAL_MODEL,
                    "setting": "human_from_scratch",
                    "mode": "full_question",
                    "dataset_types": ["arc_challenge"],
                    "question_start": 0,
                    "limit": 1,
                },
                dataset=[object()],
            )
        ],
    )

    def fake_materialize(**kwargs):
        observed.update(kwargs)
        return [collected_root / safe_name(GEN_RUN_NAME) / safe_name(GEN_MODEL) / safe_name(EVAL_MODEL)]

    monkeypatch.setattr(app_main, "_materialize_collected_evaluation", fake_materialize)

    args = Namespace(
        dataset_types="arc_challenge",
        default_dataset_types=["arc_challenge"],
        settings="human_from_scratch",
        modes="full_question",
        model=EVAL_MODEL,
        backend=None,
        processed_dataset=str(tmp_path / "processed"),
        generator_run_name=GEN_RUN_NAME,
        generator_model=GEN_MODEL,
        generator_backend=None,
        generation_log_dir=None,
        generation_log_root=str(log_root),
        cache_root=str(tmp_path / "augmented"),
        augmented_dataset=None,
        rebuild_cache=False,
        log_root=str(log_root),
        question_start=0,
        shard_count=1,
        shard_index=0,
        shard_strategy="contiguous",
        limit=1,
        run_name="eval-run",
        collected_root=str(collected_root),
        support_root=str(tmp_path / "support_sets"),
        collect_evaluated=True,
    )

    assert app_main._run_evaluate(args) == 0
    assert observed["run_name"] == "eval-run"
    assert observed["evaluation_log_root"] == log_root / safe_name("eval-run") / safe_name(GEN_RUN_NAME) / safe_name(GEN_MODEL) / safe_name(EVAL_MODEL)
    assert observed["collected_root"] == collected_root
    assert observed["augmented_dataset"] == augmented_root
    assert observed["generation_run_name"] == GEN_RUN_NAME
    assert observed["generation_model"] == GEN_MODEL
    assert observed["evaluation_model"] == EVAL_MODEL
    assert observed["planned_tasks"][0]["setting"] == "human_from_scratch"
    assert observed["planned_tasks"][0]["mode"] == "full_question"
    assert observed["planned_tasks"][0]["slice_ref"]


def test_run_evaluate_still_materializes_collected_dataset_when_incomplete(tmp_path, monkeypatch):
    collected_root = tmp_path / "collected"
    augmented_root = tmp_path / "augmented" / safe_name(GEN_RUN_NAME) / safe_name(GEN_MODEL)
    augmented_root.mkdir(parents=True, exist_ok=True)
    log_root = tmp_path / "logs"
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        app_main,
        "_resolved_generation_artifacts",
        lambda **kwargs: (GEN_MODEL, tmp_path / "generation-logs", augmented_root),
    )
    monkeypatch.setattr(
        app_main,
        "_combined_support_ids_for_run",
        lambda **kwargs: {"arc_challenge": {SAMPLE_ID}},
    )
    monkeypatch.setattr(
        app_main,
        "inspect_eval",
        lambda tasks, model, log_dir, args: ["partial"],
    )
    monkeypatch.setattr(app_main, "evaluation_logs_completed", lambda logs: False)
    monkeypatch.setattr(
        "tasks.build_evaluation_tasks",
        lambda **kwargs: [
            Namespace(
                metadata={
                    "run_name": "eval-run",
                    "generation_run_name": GEN_RUN_NAME,
                    "generation_model": GEN_MODEL,
                    "evaluation_model": EVAL_MODEL,
                    "setting": "human_from_scratch",
                    "mode": "full_question",
                    "dataset_types": ["arc_challenge"],
                    "question_start": 0,
                    "limit": 1,
                },
                dataset=[object()],
            )
        ],
    )

    def fake_materialize(**kwargs):
        observed.update(kwargs)
        return [collected_root / safe_name(GEN_RUN_NAME) / safe_name(GEN_MODEL) / safe_name(EVAL_MODEL)]

    monkeypatch.setattr(app_main, "_materialize_collected_evaluation", fake_materialize)

    args = Namespace(
        dataset_types="arc_challenge",
        default_dataset_types=["arc_challenge"],
        settings="human_from_scratch",
        modes="full_question",
        model=EVAL_MODEL,
        backend=None,
        processed_dataset=str(tmp_path / "processed"),
        generator_run_name=GEN_RUN_NAME,
        generator_model=GEN_MODEL,
        generator_backend=None,
        generation_log_dir=None,
        generation_log_root=str(log_root),
        cache_root=str(tmp_path / "augmented"),
        augmented_dataset=None,
        rebuild_cache=False,
        log_root=str(log_root),
        question_start=0,
        shard_count=1,
        shard_index=0,
        shard_strategy="contiguous",
        limit=1,
        run_name="eval-run",
        collected_root=str(collected_root),
        support_root=str(tmp_path / "support_sets"),
        collect_evaluated=True,
    )

    assert app_main._run_evaluate(args) == 1
    assert observed["evaluation_log_root"] == log_root / safe_name("eval-run") / safe_name(GEN_RUN_NAME) / safe_name(GEN_MODEL) / safe_name(EVAL_MODEL)


def test_run_evaluate_with_explicit_augmented_dataset_does_not_rematerialize_store(tmp_path, monkeypatch):
    collected_root = tmp_path / "collected"
    augmented_root = tmp_path / "explicit-augmented"
    augmented_root.mkdir(parents=True, exist_ok=True)
    log_root = tmp_path / "logs"

    def fail_ensure(**kwargs):
        raise AssertionError("explicit augmented datasets should not be rematerialized")

    monkeypatch.setattr(app_main, "ensure_augmented_dataset", fail_ensure)
    monkeypatch.setattr(
        app_main,
        "_combined_support_ids_for_run",
        lambda **kwargs: {"arc_challenge": {SAMPLE_ID}},
    )
    monkeypatch.setattr(
        app_main,
        "inspect_eval",
        lambda tasks, model, log_dir, args: ["ok"],
    )
    monkeypatch.setattr(app_main, "evaluation_logs_completed", lambda logs: True)
    monkeypatch.setattr(
        "tasks.build_evaluation_tasks",
        lambda **kwargs: [
            Namespace(
                metadata={
                    "run_name": "eval-run",
                    "generation_run_name": GEN_RUN_NAME,
                    "generation_model": GEN_MODEL,
                    "evaluation_model": EVAL_MODEL,
                    "setting": "human_from_scratch",
                    "mode": "full_question",
                    "dataset_types": ["arc_challenge"],
                    "question_start": 0,
                    "limit": 1,
                },
                dataset=[object()],
            )
        ],
    )
    monkeypatch.setattr(
        app_main,
        "_materialize_collected_evaluation",
        lambda **kwargs: [collected_root / safe_name(GEN_RUN_NAME) / safe_name(GEN_MODEL) / safe_name(EVAL_MODEL)],
    )

    args = Namespace(
        dataset_types="arc_challenge",
        default_dataset_types=["arc_challenge"],
        settings="human_from_scratch",
        modes="full_question",
        model=EVAL_MODEL,
        backend=None,
        processed_dataset=str(tmp_path / "processed"),
        generator_run_name=GEN_RUN_NAME,
        generator_model=GEN_MODEL,
        generator_backend=None,
        generation_log_dir=None,
        generation_log_root=str(log_root),
        cache_root=str(tmp_path / "augmented"),
        augmented_dataset=str(augmented_root),
        rebuild_cache=False,
        log_root=str(log_root),
        question_start=0,
        shard_count=1,
        shard_index=0,
        shard_strategy="contiguous",
        limit=1,
        run_name="eval-run",
        collected_root=str(collected_root),
        support_root=str(tmp_path / "support_sets"),
        collect_evaluated=False,
    )

    assert app_main._run_evaluate(args) == 0
