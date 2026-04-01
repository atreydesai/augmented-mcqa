import os
from pathlib import Path
from unittest.mock import patch

from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, scorer
from inspect_ai.solver import solver

from utils.scheduler_state import (
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_PLANNED,
    STATUS_STALE,
    build_scheduler_state,
    build_scheduler_state_from_tasks,
    collect_slice_attempts,
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
                name="augmented_mcqa_eval_test",
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


def test_build_scheduler_state_does_not_mark_dead_submitted_jobs_pending():
    manifests = [
        {
            "stage": "evaluate",
            "run_name": "run1",
            "submission_id": "sub-a",
            "submission_created_at": "2026-03-11T10:00:00+00:00",
            "_path": "/tmp/sub-a/manifest.json",
            "tasks": [
                {
                    "slice_ref": "dead-job",
                    "stage": "evaluate",
                    "model": "vllm/model-a",
                    "dataset_type": "mmlu_pro",
                    "setting": "augment_ablation",
                    "mode": "full_question",
                    "task_slug": "dead-job-task",
                    "question_start": 0,
                    "question_end": 10,
                    "state_dependency_refs": [],
                    "submit_dependency_refs": [],
                    "submitted_at": "2026-03-11T10:00:00+00:00",
                    "submitted_job_id": "12345",
                    "force": False,
                },
            ],
        },
    ]

    state = build_scheduler_state(manifests=manifests, attempts_by_slice={}, live_job_ids=set())
    by_ref = {entry["slice_ref"]: entry for entry in state["slices"]}

    assert by_ref["dead-job"]["status"] == STATUS_PLANNED


def test_build_scheduler_state_from_tasks_tracks_local_plans_without_manifests():
    state = build_scheduler_state_from_tasks(
        stage="evaluate",
        run_name="eval-run",
        planned_tasks=[
            {
                "slice_ref": "slice-current",
                "stage": "evaluate",
                "model": "vllm/model-a",
                "dataset_type": "arc_challenge",
                "setting": "human_from_scratch",
                "mode": "full_question",
                "task_slug": "slice-current-task",
                "question_start": 0,
                "question_end": 1,
                "generation_run_name": "gen-run",
                "generation_model": "vllm/model-g",
                "state_dependency_refs": [],
                "submit_dependency_refs": [],
                "force": False,
            },
            {
                "slice_ref": "slice-planned",
                "stage": "evaluate",
                "model": "vllm/model-a",
                "dataset_type": "arc_challenge",
                "setting": "human_from_scratch",
                "mode": "choices_only",
                "task_slug": "slice-planned-task",
                "question_start": 1,
                "question_end": 2,
                "generation_run_name": "gen-run",
                "generation_model": "vllm/model-g",
                "state_dependency_refs": [],
                "submit_dependency_refs": [],
                "force": False,
            },
        ],
        attempts_by_slice={
            "slice-current": [
                {
                    "slice_ref": "slice-current",
                    "status": "success",
                    "completed_at": "2026-03-11T11:00:00+00:00",
                }
            ]
        },
    )
    by_ref = {entry["slice_ref"]: entry for entry in state["slices"]}

    assert by_ref["slice-current"]["status"] == "current"
    assert by_ref["slice-current"]["generation_run_name"] == "gen-run"
    assert by_ref["slice-current"]["setting"] == "human_from_scratch"
    assert by_ref["slice-planned"]["status"] == STATUS_PLANNED
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
