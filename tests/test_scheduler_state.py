import os
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, scorer
from inspect_ai.solver import solver

from utils.scheduler_state import (
    STATUS_CURRENT,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_PLANNED,
    STATUS_STALE,
    build_scheduler_state,
    collect_slice_attempts,
    render_scheduler_dashboard,
)


@solver
def _solver():
    async def solve(state, generate):  # noqa: ANN001
        state.output.completion = "done"
        return state

    return solve


@scorer(metrics=[])
def _score_with(value: float):
    async def score(state, target):  # noqa: ANN001
        return Score(value=value)

    return score


def _write_eval_log(root: Path, *, slice_ref: str, score_value: float, kind: str = "evaluation"):
    root.mkdir(parents=True, exist_ok=True)
    xdg_data_home = root / "xdg"
    xdg_data_home.mkdir(parents=True, exist_ok=True)
    with patch.dict(
        os.environ,
        {
            "INSPECT_TRACE_FILE": str(root / "trace.log"),
            "XDG_DATA_HOME": str(xdg_data_home),
        },
    ):
        eval(
            Task(
                name="final5_eval_test",
                dataset=MemoryDataset([Sample(input="Q", choices=["A", "B"], target="A", id="arc:0")]),
                solver=_solver(),
                scorer=_score_with(score_value),
                metadata={"kind": kind, "slice_ref": slice_ref},
            ),
            log_dir=str(root),
            display="none",
        )


def test_build_scheduler_state_marks_pending_failed_current_and_stale():
    manifests = [
        {
            "stage": "generate",
            "run_name": "run1",
            "submission_id": "sub-a",
            "submission_created_at": "2026-03-11T10:00:00+00:00",
            "_path": "/tmp/sub-a/manifest.json",
            "tasks": [
                {
                    "slice_ref": "dep",
                    "stage": "generate",
                    "model": "vllm/model-a",
                    "dataset_type": "arc_challenge",
                    "strategy": "model_from_scratch",
                    "task_slug": "dep-task",
                    "question_start": 0,
                    "question_end": 10,
                    "state_dependency_refs": [],
                    "submit_dependency_refs": [],
                    "submitted_at": "2026-03-11T10:00:00+00:00",
                    "force": False,
                },
                {
                    "slice_ref": "child",
                    "stage": "evaluate",
                    "model": "vllm/model-b",
                    "dataset_type": "arc_challenge",
                    "setting": "model_from_scratch",
                    "mode": "full_question",
                    "task_slug": "child-task",
                    "question_start": 0,
                    "question_end": 10,
                    "state_dependency_refs": ["dep"],
                    "submit_dependency_refs": [],
                    "submitted_at": "2026-03-11T10:00:00+00:00",
                    "force": False,
                },
                {
                    "slice_ref": "failed",
                    "stage": "generate",
                    "model": "vllm/model-a",
                    "dataset_type": "gpqa",
                    "strategy": "augment_ablation",
                    "task_slug": "failed-task",
                    "question_start": 0,
                    "question_end": 5,
                    "state_dependency_refs": [],
                    "submit_dependency_refs": [],
                    "submitted_at": "2026-03-11T10:00:00+00:00",
                    "force": False,
                },
            ],
        },
        {
            "stage": "generate",
            "run_name": "run1",
            "submission_id": "sub-b",
            "submission_created_at": "2026-03-11T12:00:00+00:00",
            "_path": "/tmp/sub-b/manifest.json",
            "tasks": [
                {
                    "slice_ref": "dep",
                    "stage": "generate",
                    "model": "vllm/model-a",
                    "dataset_type": "arc_challenge",
                    "strategy": "model_from_scratch",
                    "task_slug": "dep-task",
                    "question_start": 0,
                    "question_end": 10,
                    "state_dependency_refs": [],
                    "submit_dependency_refs": [],
                    "submitted_at": "2026-03-11T12:00:00+00:00",
                    "force": True,
                },
                {
                    "slice_ref": "pending",
                    "stage": "generate",
                    "model": "vllm/model-a",
                    "dataset_type": "mmlu_pro",
                    "strategy": "augment_human",
                    "task_slug": "pending-task",
                    "question_start": 0,
                    "question_end": 5,
                    "state_dependency_refs": [],
                    "submit_dependency_refs": [],
                    "submitted_at": "2026-03-11T12:00:00+00:00",
                    "force": False,
                },
                {
                    "slice_ref": "planned",
                    "stage": "generate",
                    "model": "vllm/model-a",
                    "dataset_type": "arc_challenge",
                    "strategy": "augment_model",
                    "task_slug": "planned-task",
                    "question_start": 10,
                    "question_end": 15,
                    "state_dependency_refs": [],
                    "submit_dependency_refs": [],
                    "submitted_at": "",
                    "force": False,
                }
            ],
        },
    ]
    attempts = {
        "dep": [{"slice_ref": "dep", "status": "success", "completed_at": "2026-03-11T11:00:00+00:00"}],
        "child": [{"slice_ref": "child", "status": "success", "completed_at": "2026-03-11T11:30:00+00:00"}],
        "failed": [{"slice_ref": "failed", "status": "failed", "completed_at": "2026-03-11T10:30:00+00:00"}],
    }

    state = build_scheduler_state(manifests=manifests, attempts_by_slice=attempts)
    by_ref = {entry["slice_ref"]: entry for entry in state["slices"]}

    assert by_ref["dep"]["status"] == STATUS_PENDING
    assert by_ref["child"]["status"] == STATUS_STALE
    assert by_ref["failed"]["status"] == STATUS_FAILED
    assert by_ref["pending"]["status"] == STATUS_PENDING
    assert by_ref["planned"]["status"] == STATUS_PLANNED


def test_render_scheduler_dashboard_contains_statuses():
    state = {
        "stage": "evaluate",
        "run_name": "run1",
        "submission_count": 1,
        "slice_count": 2,
        "generated_at": "2026-03-11T12:00:00+00:00",
        "slices": [
            {
                "slice_ref": "a",
                "stage": "evaluate",
                "model": "vllm/model-a",
                "dataset_type": "arc_challenge",
                "setting": "model_from_scratch",
                "mode": "full_question",
                "question_start": 0,
                "question_end": 10,
                "task_slug": "task-a",
                "status": STATUS_CURRENT,
                "latest_attempt": {"status": "success"},
            },
            {
                "slice_ref": "b",
                "stage": "evaluate",
                "model": "vllm/model-a",
                "dataset_type": "arc_challenge",
                "setting": "augment_model",
                "mode": "choices_only",
                "question_start": 0,
                "question_end": 10,
                "task_slug": "task-b",
                "status": STATUS_STALE,
                "latest_attempt": {"status": "success"},
            },
        ],
    }

    html = render_scheduler_dashboard(state)
    assert "evaluate scheduler status" in html
    assert "task-a" in html
    assert STATUS_CURRENT in html
    assert STATUS_STALE in html


def test_render_scheduler_dashboard_uses_generation_strategies_as_columns():
    state = {
        "stage": "generate",
        "run_name": "run1",
        "submission_count": 1,
        "slice_count": 2,
        "generated_at": "2026-03-11T12:00:00+00:00",
        "slices": [
            {
                "slice_ref": "a",
                "stage": "generate",
                "model": "together/model-a",
                "dataset_type": "arc_challenge",
                "strategy": "model_from_scratch",
                "question_start": 0,
                "question_end": 10,
                "task_slug": "task-a",
                "status": STATUS_CURRENT,
                "latest_attempt": {"status": "success"},
            },
            {
                "slice_ref": "b",
                "stage": "generate",
                "model": "together/model-a",
                "dataset_type": "arc_challenge",
                "strategy": "augment_model",
                "question_start": 0,
                "question_end": 10,
                "task_slug": "task-b",
                "status": STATUS_PLANNED,
                "latest_attempt": {},
            },
        ],
    }

    html = render_scheduler_dashboard(state)
    assert "<th>model_from_scratch</th>" in html
    assert "<th>augment_model</th>" in html
    assert "task-a" in html
    assert "task-b" in html
    assert "None:None" not in html


def test_collect_slice_attempts_treats_completed_evaluation_logs_as_success(tmp_path):
    root = tmp_path / "eval-logs"
    _write_eval_log(root, slice_ref="evaluation|run1|model|arc|setting|mode|0|1", score_value=0.0)

    attempts = collect_slice_attempts(root, kind="evaluation")
    assert attempts["evaluation|run1|model|arc|setting|mode|0|1"][-1]["status"] == "success"


def test_collect_slice_attempts_treats_completed_generation_logs_as_success(tmp_path):
    root = tmp_path / "gen-logs"
    _write_eval_log(
        root,
        slice_ref="generation|run1|model|arc|setting|0|1",
        score_value=1.0,
        kind="generation",
    )

    attempts = collect_slice_attempts(root, kind="generation")
    assert attempts["generation|run1|model|arc|setting|0|1"][-1]["status"] == "success"


def test_collect_slice_attempts_skips_incomplete_archives(tmp_path):
    root = tmp_path / "partial-logs"
    root.mkdir()
    (root / "partial.eval").write_bytes(b"not-a-complete-archive")

    attempts = collect_slice_attempts(root, kind="evaluation")
    assert attempts == {}


def test_collect_slice_attempts_treats_errored_logs_as_failed(monkeypatch):
    fake_summary = {
        "metadata": {"slice_ref": "generation|run1|model|arc|setting|0|1"},
        "status": "error",
        "completed_at": "2026-03-11T12:00:00+00:00",
        "score_values": [1.0],
        "sample_statuses": ["success"],
        "summary_count": 1,
    }

    monkeypatch.setattr(
        "utils.scheduler_state.iter_log_summaries",
        lambda _path, *, kind=None: iter([(Path("/tmp/fake.eval"), fake_summary)]),
    )

    attempts = collect_slice_attempts("/tmp/unused", kind="generation")
    assert attempts["generation|run1|model|arc|setting|0|1"][-1]["status"] == "failed"
    assert attempts["generation|run1|model|arc|setting|0|1"][-1]["completed_at"] == "2026-03-11T12:00:00+00:00"


def test_collect_slice_attempts_treats_generation_sample_errors_as_failed(monkeypatch):
    fake_summary = {
        "metadata": {"slice_ref": "generation|run1|model|arc|setting|0|1"},
        "status": "success",
        "completed_at": "2026-03-11T12:00:00+00:00",
        "score_values": [1.0],
        "sample_statuses": ["error"],
        "summary_count": 1,
    }

    monkeypatch.setattr(
        "utils.scheduler_state.iter_log_summaries",
        lambda _path, *, kind=None: iter([(Path("/tmp/fake.eval"), fake_summary)]),
    )

    attempts = collect_slice_attempts("/tmp/unused", kind="generation")
    assert attempts["generation|run1|model|arc|setting|0|1"][-1]["status"] == "failed"
