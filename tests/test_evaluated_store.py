from argparse import Namespace
from pathlib import Path

import json

from datasets import Dataset, load_from_disk
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, scorer
from inspect_ai.solver import solver

import main
from data.final5_store import (
    AUGMENTED_RECORD_COLUMNS,
    EVALUATED_RECORD_COLUMNS,
    materialize_evaluated_datasets,
)
from utils.constants import AUGMENTED_STORE_MANIFEST
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
        "status": "success",
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


def test_materialize_evaluated_datasets_preserves_augmented_columns_and_adds_eval_fields(tmp_path):
    eval_root = tmp_path / "inspect"
    augmented_root = tmp_path / "augmented"
    output_root = tmp_path / "evaluated"
    _write_eval_log(eval_root)
    _write_augmented_predecessor(augmented_root)

    outputs = materialize_evaluated_datasets(eval_root, output_root, augmented_root=augmented_root)
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

    assert row["status"] == "success"
    assert row["traces"] == {"source": "augmented"}
    assert row["evaluation_is_correct"] is True
    assert row["evaluation_prediction"] == "B"
    assert row["evaluation_prompt"] == "Answer the MCQ."
    assert row["evaluation_raw_output"] == '{"answer": "B"}'
    assert row["evaluation_question_idx"] == 0
    assert row["evaluation_log_path"].endswith(".eval")


def test_run_analyze_materializes_evaluated_dataset_before_plotting(tmp_path):
    eval_root = tmp_path / "inspect"
    augmented_root = tmp_path / "augmented"
    evaluated_root = tmp_path / "evaluated"
    plots_root = tmp_path / "plots"
    table_output = plots_root / "tables" / "summary.csv"
    _write_eval_log(eval_root)
    _write_augmented_predecessor(augmented_root)

    args = Namespace(
        results_root=str(eval_root),
        output_dir=str(plots_root),
        table_output=str(table_output),
        skip_tables=False,
        evaluated_output_root=str(evaluated_root),
        skip_evaluated_dataset=False,
    )

    previous_root = main.DEFAULT_AUGMENTED_CACHE_ROOT
    main.DEFAULT_AUGMENTED_CACHE_ROOT = augmented_root
    try:
        rc = main._run_analyze(args)
    finally:
        main.DEFAULT_AUGMENTED_CACHE_ROOT = previous_root

    assert rc == 0
    assert (evaluated_root / safe_name(GEN_RUN_NAME) / safe_name(GEN_MODEL) / safe_name(EVAL_MODEL)).exists()
    assert table_output.exists()
    assert any(path.name.startswith("pairwise_") for path in plots_root.iterdir() if path.is_file())
