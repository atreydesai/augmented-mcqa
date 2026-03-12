import json
from subprocess import CompletedProcess

from datasets import Dataset, DatasetDict
from inspect_ai.dataset import MemoryDataset, Sample

import main as app_main
from utils.constants import DEFAULT_LOCAL_EVALUATION_MODELS, DEFAULT_LOCAL_GENERATION_MODELS
from utils.modeling import resolve_model_name
from utils.scheduler_state import evaluation_slice_ref, generation_slice_ref


def _processed_dataset(path, *, counts=None):
    counts = counts or {"arc_challenge": 1, "mmlu_pro": 1, "gpqa": 1}

    def _rows(dataset_type: str, count: int):
        rows = []
        for index in range(count):
            if dataset_type == "mmlu_pro":
                rows.append({"question_id": 100 + index, "question": f"{dataset_type} {index}", "answer": "A"})
            else:
                rows.append({"id": f"{dataset_type}-{index}", "question": f"{dataset_type} {index}", "answer": "A"})
        return rows

    DatasetDict(
        {
            "arc_challenge": Dataset.from_list(_rows("arc_challenge", counts.get("arc_challenge", 0))),
            "mmlu_pro": Dataset.from_list(_rows("mmlu_pro", counts.get("mmlu_pro", 0))),
            "gpqa": Dataset.from_list(_rows("gpqa", counts.get("gpqa", 0))),
        }
    ).save_to_disk(str(path))


def _manifest_path(bundle_dir):
    return next(bundle_dir.glob("submissions/*/manifest.json"))


def _submit_path(bundle_dir):
    return next(bundle_dir.glob("submissions/*/submit_all.sh"))


def _read_manifest(bundle_dir):
    return json.loads(_manifest_path(bundle_dir).read_text(encoding="utf-8"))


def test_submit_generate_cluster_write_only_writes_strategy_slice_manifest(tmp_path):
    dataset_path = tmp_path / "processed"
    bundle_dir = tmp_path / "bundle"
    _processed_dataset(dataset_path)

    rc = app_main.main(
        [
            "submit-generate-cluster",
            "--run-name",
            "cluster-gen",
            "--processed-dataset",
            str(dataset_path),
            "--output-dir",
            str(bundle_dir),
            "--write-only",
            "--render-status",
        ]
    )

    assert rc == 0
    manifest = _read_manifest(bundle_dir)
    assert manifest["stage"] == "generate"
    assert manifest["task_count"] == len(DEFAULT_LOCAL_GENERATION_MODELS) * 3 * 4
    assert {task["dataset_type"] for task in manifest["tasks"]} == {"arc_challenge", "mmlu_pro", "gpqa"}
    assert {task["strategy"] for task in manifest["tasks"]} == {
        "model_from_scratch",
        "augment_human",
        "augment_model",
        "augment_ablation",
    }
    assert {task["resource_class"] for task in manifest["tasks"]} == {"local"}
    first_task = manifest["tasks"][0]
    assert "bootstrap_stdout" not in first_task
    assert "bootstrap_stderr" not in first_task
    assert first_task["task_stdout"].endswith(".out")
    assert first_task["task_stderr"].endswith(".err")
    submit_text = _submit_path(bundle_dir).read_text(encoding="utf-8")
    assert 'task["task_stdout"]' in submit_text
    assert 'task["bootstrap_stdout"]' not in submit_text
    assert "${{" not in submit_text
    assert "{{}}" not in submit_text
    local_wrapper_text = next(bundle_dir.glob("submissions/*/run_local_task.sbatch")).read_text(encoding="utf-8")
    api_wrapper_text = next(bundle_dir.glob("submissions/*/run_api_task.sbatch")).read_text(encoding="utf-8")
    assert "${{" not in local_wrapper_text
    assert "${{" not in api_wrapper_text
    assert (bundle_dir / "scheduler_state.json").exists()
    assert (bundle_dir / "scheduler_status.html").exists()
    assert next(bundle_dir.glob("submissions/*/run_local_task.sbatch")).exists()
    assert next(bundle_dir.glob("submissions/*/run_api_task.sbatch")).exists()


def test_submit_generate_cluster_write_only_can_be_replanned(tmp_path):
    dataset_path = tmp_path / "processed"
    bundle_dir = tmp_path / "bundle"
    _processed_dataset(dataset_path, counts={"arc_challenge": 1, "mmlu_pro": 0, "gpqa": 0})

    argv = [
        "submit-generate-cluster",
        "--run-name",
        "cluster-gen",
        "--processed-dataset",
        str(dataset_path),
        "--dataset-types",
        "arc_challenge",
        "--models",
        "Qwen/Qwen3-4B-Instruct-2507",
        "--generation-strategies",
        "model_from_scratch",
        "--output-dir",
        str(bundle_dir),
        "--write-only",
    ]

    assert app_main.main(argv) == 0
    assert app_main.main(argv) == 0
    assert len(list(bundle_dir.glob("submissions/*/manifest.json"))) == 2


def test_submit_generate_cluster_noop_write_only_can_refresh_status_outputs(tmp_path, monkeypatch):
    dataset_path = tmp_path / "processed"
    bundle_dir = tmp_path / "bundle"
    _processed_dataset(dataset_path, counts={"arc_challenge": 1, "mmlu_pro": 0, "gpqa": 0})
    model = resolve_model_name("Qwen/Qwen3-4B-Instruct-2507", None)

    initial = [
        "submit-generate-cluster",
        "--run-name",
        "cluster-gen",
        "--processed-dataset",
        str(dataset_path),
        "--dataset-types",
        "arc_challenge",
        "--models",
        "Qwen/Qwen3-4B-Instruct-2507",
        "--generation-strategies",
        "model_from_scratch",
        "--output-dir",
        str(bundle_dir),
        "--write-only",
        "--render-status",
    ]
    assert app_main.main(initial) == 0

    def fake_state(*, stage, run_name, output_dir=None):
        return {
            "slices": [
                {
                    "slice_ref": generation_slice_ref(
                        run_name="cluster-gen",
                        model=model,
                        dataset_type="arc_challenge",
                        strategy="model_from_scratch",
                        question_start=0,
                        question_end=1,
                    ),
                    "status": "current",
                }
            ]
        }

    monkeypatch.setattr(app_main, "_current_stage_state", fake_state)

    refresh = [
        "submit-generate-cluster",
        "--run-name",
        "cluster-gen",
        "--processed-dataset",
        str(dataset_path),
        "--dataset-types",
        "arc_challenge",
        "--models",
        "Qwen/Qwen3-4B-Instruct-2507",
        "--generation-strategies",
        "model_from_scratch",
        "--output-dir",
        str(bundle_dir),
        "--write-only",
        "--render-status",
    ]
    assert app_main.main(refresh) == 0
    assert len(list(bundle_dir.glob("submissions/*/manifest.json"))) == 1
    assert (bundle_dir / "scheduler_state.json").exists()
    assert (bundle_dir / "scheduler_status.html").exists()


def test_submit_generate_cluster_relative_output_dir_writes_absolute_runtime_paths(tmp_path, monkeypatch):
    dataset_path = tmp_path / "processed"
    bundle_dir = tmp_path / "bundle"
    _processed_dataset(dataset_path, counts={"arc_challenge": 1, "mmlu_pro": 0, "gpqa": 0})
    monkeypatch.chdir(tmp_path)

    rc = app_main.main(
        [
            "submit-generate-cluster",
            "--run-name",
            "cluster-gen",
            "--processed-dataset",
            str(dataset_path),
            "--dataset-types",
            "arc_challenge",
            "--models",
            "Qwen/Qwen3-4B-Instruct-2507",
            "--generation-strategies",
            "model_from_scratch",
            "--output-dir",
            "bundle",
            "--write-only",
        ]
    )

    assert rc == 0
    manifest_path = _manifest_path(bundle_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task = manifest["tasks"][0]
    assert task["task_stdout"].startswith("/")
    assert task["task_stderr"].startswith("/")

    submit_text = _submit_path(bundle_dir).read_text(encoding="utf-8")
    assert str(manifest_path.resolve()) in submit_text
    assert str((bundle_dir / "submissions" / manifest["submission_id"] / "run_api_task.sbatch").resolve()) in submit_text


def test_submit_generate_cluster_mixed_local_and_api_models_split_resource_classes(tmp_path):
    dataset_path = tmp_path / "processed"
    bundle_dir = tmp_path / "bundle"
    _processed_dataset(dataset_path, counts={"arc_challenge": 1, "mmlu_pro": 0, "gpqa": 0})

    rc = app_main.main(
        [
            "submit-generate-cluster",
            "--run-name",
            "cluster-gen",
            "--processed-dataset",
            str(dataset_path),
            "--dataset-types",
            "arc_challenge",
            "--models",
            "Qwen/Qwen3-4B-Instruct-2507,gpt-5.2-2025-12-11",
            "--generation-strategies",
            "model_from_scratch",
            "--output-dir",
            str(bundle_dir),
            "--write-only",
        ]
    )

    assert rc == 0
    manifest = _read_manifest(bundle_dir)
    assert manifest["task_count"] == 2
    classes = {task["model"]: task["resource_class"] for task in manifest["tasks"]}
    assert classes[resolve_model_name("Qwen/Qwen3-4B-Instruct-2507", None)] == "local"
    assert classes[resolve_model_name("gpt-5.2-2025-12-11", None)] == "api"


def test_submit_generate_cluster_questions_per_job_chunks_and_wires_dependencies(tmp_path):
    dataset_path = tmp_path / "processed"
    bundle_dir = tmp_path / "bundle"
    _processed_dataset(dataset_path, counts={"arc_challenge": 4, "mmlu_pro": 0, "gpqa": 0})
    model = resolve_model_name("Qwen/Qwen3-4B-Instruct-2507", None)

    rc = app_main.main(
        [
            "submit-generate-cluster",
            "--run-name",
            "cluster-gen",
            "--processed-dataset",
            str(dataset_path),
            "--dataset-types",
            "arc_challenge",
            "--models",
            "Qwen/Qwen3-4B-Instruct-2507",
            "--generation-strategies",
            "model_from_scratch,augment_model",
            "--questions-per-job",
            "2",
            "--output-dir",
            str(bundle_dir),
            "--write-only",
        ]
    )

    assert rc == 0
    manifest = _read_manifest(bundle_dir)
    assert manifest["task_count"] == 4

    tasks = {task["slice_ref"]: task for task in manifest["tasks"]}
    expected_pairs = [(0, 2), (2, 4)]
    for start, end in expected_pairs:
        model_ref = generation_slice_ref(
            run_name="cluster-gen",
            model=model,
            dataset_type="arc_challenge",
            strategy="model_from_scratch",
            question_start=start,
            question_end=end,
        )
        augment_ref = generation_slice_ref(
            run_name="cluster-gen",
            model=model,
            dataset_type="arc_challenge",
            strategy="augment_model",
            question_start=start,
            question_end=end,
        )
        assert tasks[model_ref]["submit_dependency_refs"] == []
        assert tasks[augment_ref]["state_dependency_refs"] == [model_ref]
        assert tasks[augment_ref]["submit_dependency_refs"] == [model_ref]
        for task_ref in (model_ref, augment_ref):
            argv = tasks[task_ref]["argv"]
            assert "--augmented-dataset" in argv
            cache_target = argv[argv.index("--augmented-dataset") + 1]
            assert f"/_cluster_slices/arc_challenge/" in cache_target
            assert f"/{start}-{end}" in cache_target


def test_submit_generate_cluster_submit_script_uses_afterany_for_concurrency_caps(tmp_path):
    dataset_path = tmp_path / "processed"
    bundle_dir = tmp_path / "bundle"
    _processed_dataset(dataset_path, counts={"arc_challenge": 2, "mmlu_pro": 0, "gpqa": 0})

    rc = app_main.main(
        [
            "submit-generate-cluster",
            "--run-name",
            "cluster-gen",
            "--processed-dataset",
            str(dataset_path),
            "--dataset-types",
            "arc_challenge",
            "--models",
            "Qwen/Qwen3-4B-Instruct-2507",
            "--generation-strategies",
            "model_from_scratch,augment_model",
            "--questions-per-job",
            "1",
            "--gpu-count",
            "1",
            "--output-dir",
            str(bundle_dir),
            "--write-only",
        ]
    )

    assert rc == 0
    submit_text = _submit_path(bundle_dir).read_text(encoding="utf-8")
    assert "afterok:" in submit_text
    assert "afterany:" in submit_text


def test_submit_generate_cluster_limit_caps_per_dataset_before_chunking(tmp_path):
    dataset_path = tmp_path / "processed"
    bundle_dir = tmp_path / "bundle"
    _processed_dataset(dataset_path, counts={"arc_challenge": 5, "mmlu_pro": 0, "gpqa": 0})
    model = resolve_model_name("Qwen/Qwen3-4B-Instruct-2507", None)

    rc = app_main.main(
        [
            "submit-generate-cluster",
            "--run-name",
            "cluster-gen",
            "--processed-dataset",
            str(dataset_path),
            "--dataset-types",
            "arc_challenge",
            "--models",
            "Qwen/Qwen3-4B-Instruct-2507",
            "--generation-strategies",
            "model_from_scratch",
            "--limit",
            "3",
            "--questions-per-job",
            "2",
            "--output-dir",
            str(bundle_dir),
            "--write-only",
        ]
    )

    assert rc == 0
    manifest = _read_manifest(bundle_dir)
    assert manifest["task_count"] == 2
    refs = {
        generation_slice_ref(
            run_name="cluster-gen",
            model=model,
            dataset_type="arc_challenge",
            strategy="model_from_scratch",
            question_start=0,
            question_end=2,
        ),
        generation_slice_ref(
            run_name="cluster-gen",
            model=model,
            dataset_type="arc_challenge",
            strategy="model_from_scratch",
            question_start=2,
            question_end=3,
        ),
    }
    assert {task["slice_ref"] for task in manifest["tasks"]} == refs
    limits = sorted(int(task["argv"][task["argv"].index("--limit") + 1]) for task in manifest["tasks"])
    assert limits == [1, 2]


def test_submit_evaluate_cluster_requires_current_generation_prerequisite(tmp_path, capsys):
    dataset_path = tmp_path / "processed"
    bundle_dir = tmp_path / "bundle"
    _processed_dataset(dataset_path)

    rc = app_main.main(
        [
            "submit-evaluate-cluster",
            "--run-name",
            "cluster-eval",
            "--generator-run-name",
            "gen-run",
            "--generator-model",
            "gpt-5.2-2025-12-11",
            "--processed-dataset",
            str(dataset_path),
            "--output-dir",
            str(bundle_dir),
            "--write-only",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "Missing current generation prerequisite" in captured.out


def test_submit_evaluate_cluster_requires_generation_for_human_from_scratch(tmp_path, capsys):
    dataset_path = tmp_path / "processed"
    bundle_dir = tmp_path / "bundle"
    _processed_dataset(dataset_path, counts={"arc_challenge": 1, "mmlu_pro": 0, "gpqa": 0})

    rc = app_main.main(
        [
            "submit-evaluate-cluster",
            "--run-name",
            "cluster-eval",
            "--generator-run-name",
            "gen-run",
            "--generator-model",
            "gpt-5.2-2025-12-11",
            "--processed-dataset",
            str(dataset_path),
            "--dataset-types",
            "arc_challenge",
            "--settings",
            "human_from_scratch",
            "--output-dir",
            str(bundle_dir),
            "--write-only",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "human_from_scratch" in captured.out


def test_submit_evaluate_cluster_writes_setting_mode_chunk_tasks_when_generation_is_current(tmp_path, monkeypatch):
    dataset_path = tmp_path / "processed"
    bundle_dir = tmp_path / "bundle"
    _processed_dataset(dataset_path, counts={"arc_challenge": 3, "mmlu_pro": 0, "gpqa": 0})
    generator_model = resolve_model_name("gpt-5.2-2025-12-11", None)
    local_eval_model = resolve_model_name(DEFAULT_LOCAL_EVALUATION_MODELS[0], None)
    api_eval_model = resolve_model_name("gpt-5.2-2025-12-11", None)

    def fake_state(*, stage, run_name, output_dir=None):
        if stage == "evaluate":
            return {"slices": []}
        refs = []
        for start, end in ((0, 2), (2, 3)):
            refs.append(
                {
                    "slice_ref": generation_slice_ref(
                        run_name="gen-run",
                        model=generator_model,
                        dataset_type="arc_challenge",
                        strategy="model_from_scratch",
                        question_start=start,
                        question_end=end,
                    ),
                    "status": "current",
                }
            )
        return {"slices": refs}

    monkeypatch.setattr(app_main, "_current_stage_state", fake_state)

    rc = app_main.main(
        [
            "submit-evaluate-cluster",
            "--run-name",
            "cluster-eval",
            "--generator-run-name",
            "gen-run",
            "--generator-model",
            "gpt-5.2-2025-12-11",
            "--processed-dataset",
            str(dataset_path),
            "--dataset-types",
            "arc_challenge",
            "--models",
            "Qwen/Qwen3-4B-Instruct-2507,gpt-5.2-2025-12-11",
            "--settings",
            "model_from_scratch",
            "--modes",
            "full_question,choices_only",
            "--questions-per-job",
            "2",
            "--output-dir",
            str(bundle_dir),
            "--write-only",
        ]
    )

    assert rc == 0
    manifest = _read_manifest(bundle_dir)
    assert manifest["task_count"] == 8
    assert {task["resource_class"] for task in manifest["tasks"]} == {"local", "api"}
    assert {task["setting"] for task in manifest["tasks"]} == {"model_from_scratch"}
    assert {task["mode"] for task in manifest["tasks"]} == {"full_question", "choices_only"}

    task_by_model = {}
    for task in manifest["tasks"]:
        task_by_model.setdefault(task["model"], []).append(task)
        assert "--settings" in task["argv"]
        assert "--modes" in task["argv"]
        assert task["submit_dependency_refs"] == []
        assert len(task["state_dependency_refs"]) == 1

    assert {task["resource_class"] for task in task_by_model[local_eval_model]} == {"local"}
    assert {task["resource_class"] for task in task_by_model[api_eval_model]} == {"api"}


def test_submit_evaluate_cluster_limit_caps_per_dataset_before_chunking(tmp_path, monkeypatch):
    dataset_path = tmp_path / "processed"
    bundle_dir = tmp_path / "bundle"
    _processed_dataset(dataset_path, counts={"arc_challenge": 3, "mmlu_pro": 0, "gpqa": 0})
    generator_model = resolve_model_name("gpt-5.2-2025-12-11", None)

    def fake_state(*, stage, run_name, output_dir=None):
        if stage == "evaluate":
            return {"slices": []}
        return {
            "slices": [
                {
                    "slice_ref": generation_slice_ref(
                        run_name="gen-run",
                        model=generator_model,
                        dataset_type="arc_challenge",
                        strategy="model_from_scratch",
                        question_start=0,
                        question_end=2,
                    ),
                    "status": "current",
                },
                {
                    "slice_ref": generation_slice_ref(
                        run_name="gen-run",
                        model=generator_model,
                        dataset_type="arc_challenge",
                        strategy="model_from_scratch",
                        question_start=2,
                        question_end=3,
                    ),
                    "status": "current",
                },
            ]
        }

    monkeypatch.setattr(app_main, "_current_stage_state", fake_state)

    rc = app_main.main(
        [
            "submit-evaluate-cluster",
            "--run-name",
            "cluster-eval",
            "--generator-run-name",
            "gen-run",
            "--generator-model",
            "gpt-5.2-2025-12-11",
            "--processed-dataset",
            str(dataset_path),
            "--dataset-types",
            "arc_challenge",
            "--models",
            "Qwen/Qwen3-4B-Instruct-2507",
            "--settings",
            "model_from_scratch",
            "--modes",
            "full_question",
            "--limit",
            "2",
            "--questions-per-job",
            "2",
            "--output-dir",
            str(bundle_dir),
            "--write-only",
        ]
    )

    assert rc == 0
    manifest = _read_manifest(bundle_dir)
    assert manifest["task_count"] == 1
    task = manifest["tasks"][0]
    assert task["question_start"] == 0
    assert task["question_end"] == 2
    assert task["argv"][task["argv"].index("--limit") + 1] == "2"


def test_submit_evaluate_cluster_allows_human_from_scratch_when_any_generation_slice_is_current(tmp_path, monkeypatch):
    dataset_path = tmp_path / "processed"
    bundle_dir = tmp_path / "bundle"
    _processed_dataset(dataset_path, counts={"arc_challenge": 2, "mmlu_pro": 0, "gpqa": 0})
    generator_model = resolve_model_name("gpt-5.2-2025-12-11", None)

    def fake_state(*, stage, run_name, output_dir=None):
        if stage == "evaluate":
            return {"slices": []}
        return {
            "slices": [
                {
                    "slice_ref": generation_slice_ref(
                        run_name="gen-run",
                        model=generator_model,
                        dataset_type="arc_challenge",
                        strategy="augment_human",
                        question_start=0,
                        question_end=2,
                    ),
                    "status": "current",
                }
            ]
        }

    monkeypatch.setattr(app_main, "_current_stage_state", fake_state)

    rc = app_main.main(
        [
            "submit-evaluate-cluster",
            "--run-name",
            "cluster-eval",
            "--generator-run-name",
            "gen-run",
            "--generator-model",
            "gpt-5.2-2025-12-11",
            "--processed-dataset",
            str(dataset_path),
            "--dataset-types",
            "arc_challenge",
            "--models",
            "Qwen/Qwen3-4B-Instruct-2507",
            "--settings",
            "human_from_scratch",
            "--modes",
            "full_question",
            "--output-dir",
            str(bundle_dir),
            "--write-only",
        ]
    )

    assert rc == 0
    manifest = _read_manifest(bundle_dir)
    assert manifest["task_count"] == 1
    task = manifest["tasks"][0]
    assert task["setting"] == "human_from_scratch"
    assert task["state_dependency_refs"] == [
        generation_slice_ref(
            run_name="gen-run",
            model=generator_model,
            dataset_type="arc_challenge",
            strategy="augment_human",
            question_start=0,
            question_end=2,
        )
    ]


def test_submit_generate_cluster_allows_augment_model_after_failed_prerequisite_when_rows_remain(tmp_path, monkeypatch):
    dataset_path = tmp_path / "processed"
    bundle_dir = tmp_path / "bundle"
    _processed_dataset(dataset_path, counts={"arc_challenge": 2, "mmlu_pro": 0, "gpqa": 0})
    model = resolve_model_name("Qwen/Qwen3-4B-Instruct-2507", None)
    model_ref = generation_slice_ref(
        run_name="cluster-gen",
        model=model,
        dataset_type="arc_challenge",
        strategy="model_from_scratch",
        question_start=0,
        question_end=2,
    )
    augment_ref = generation_slice_ref(
        run_name="cluster-gen",
        model=model,
        dataset_type="arc_challenge",
        strategy="augment_model",
        question_start=0,
        question_end=2,
    )

    def fake_state(*, stage, run_name, output_dir=None):
        if stage == "generate":
            return {"slices": [{"slice_ref": model_ref, "status": "failed"}]}
        return {"slices": []}

    monkeypatch.setattr(app_main, "_current_stage_state", fake_state)
    monkeypatch.setattr(
        app_main,
        "build_generation_dataset",
        lambda *args, **kwargs: MemoryDataset([Sample(input="Q1", target="", id="arc_challenge:arc-1")]),
    )

    rc = app_main.main(
        [
            "submit-generate-cluster",
            "--run-name",
            "cluster-gen",
            "--processed-dataset",
            str(dataset_path),
            "--dataset-types",
            "arc_challenge",
            "--models",
            "Qwen/Qwen3-4B-Instruct-2507",
            "--generation-strategies",
            "augment_model",
            "--questions-per-job",
            "2",
            "--output-dir",
            str(bundle_dir),
            "--write-only",
        ]
    )

    assert rc == 0
    manifest = _read_manifest(bundle_dir)
    assert manifest["task_count"] == 1
    task = manifest["tasks"][0]
    assert task["slice_ref"] == augment_ref
    assert task["state_dependency_refs"] == [model_ref]
    assert task["submit_dependency_refs"] == []


def test_submit_evaluate_cluster_allows_failed_generation_prerequisite_when_rows_remain(tmp_path, monkeypatch):
    dataset_path = tmp_path / "processed"
    bundle_dir = tmp_path / "bundle"
    _processed_dataset(dataset_path, counts={"arc_challenge": 2, "mmlu_pro": 0, "gpqa": 0})
    generator_model = resolve_model_name("gpt-5.2-2025-12-11", None)
    generation_ref = generation_slice_ref(
        run_name="gen-run",
        model=generator_model,
        dataset_type="arc_challenge",
        strategy="model_from_scratch",
        question_start=0,
        question_end=2,
    )
    eval_ref = evaluation_slice_ref(
        run_name="cluster-eval",
        model=resolve_model_name("Qwen/Qwen3-4B-Instruct-2507", None),
        dataset_type="arc_challenge",
        setting="model_from_scratch",
        mode="full_question",
        question_start=0,
        question_end=2,
    )

    def fake_state(*, stage, run_name, output_dir=None):
        if stage == "evaluate":
            return {"slices": []}
        return {"slices": [{"slice_ref": generation_ref, "status": "failed"}]}

    monkeypatch.setattr(app_main, "_current_stage_state", fake_state)
    monkeypatch.setattr(app_main, "ensure_augmented_dataset", lambda **kwargs: kwargs["output_path"])
    monkeypatch.setattr(
        app_main,
        "build_evaluation_dataset",
        lambda *args, **kwargs: MemoryDataset(
            [Sample(input="Q1", choices=["A", "B"], target="A", id="arc_challenge:arc-1")]
        ),
    )

    rc = app_main.main(
        [
            "submit-evaluate-cluster",
            "--run-name",
            "cluster-eval",
            "--generator-run-name",
            "gen-run",
            "--generator-model",
            "gpt-5.2-2025-12-11",
            "--processed-dataset",
            str(dataset_path),
            "--dataset-types",
            "arc_challenge",
            "--models",
            "Qwen/Qwen3-4B-Instruct-2507",
            "--settings",
            "model_from_scratch",
            "--modes",
            "full_question",
            "--questions-per-job",
            "2",
            "--output-dir",
            str(bundle_dir),
            "--write-only",
        ]
    )

    assert rc == 0
    manifest = _read_manifest(bundle_dir)
    assert manifest["task_count"] == 1
    assert manifest["tasks"][0]["slice_ref"] == eval_ref


def test_submit_evaluate_cluster_skips_failed_generation_chunk_when_no_rows_remain(tmp_path, monkeypatch):
    dataset_path = tmp_path / "processed"
    bundle_dir = tmp_path / "bundle"
    _processed_dataset(dataset_path, counts={"arc_challenge": 2, "mmlu_pro": 0, "gpqa": 0})
    generator_model = resolve_model_name("gpt-5.2-2025-12-11", None)
    generation_ref = generation_slice_ref(
        run_name="gen-run",
        model=generator_model,
        dataset_type="arc_challenge",
        strategy="model_from_scratch",
        question_start=0,
        question_end=2,
    )

    def fake_state(*, stage, run_name, output_dir=None):
        if stage == "evaluate":
            return {"slices": []}
        return {"slices": [{"slice_ref": generation_ref, "status": "failed"}]}

    monkeypatch.setattr(app_main, "_current_stage_state", fake_state)
    monkeypatch.setattr(app_main, "ensure_augmented_dataset", lambda **kwargs: kwargs["output_path"])
    monkeypatch.setattr(app_main, "build_evaluation_dataset", lambda *args, **kwargs: MemoryDataset([]))

    rc = app_main.main(
        [
            "submit-evaluate-cluster",
            "--run-name",
            "cluster-eval",
            "--generator-run-name",
            "gen-run",
            "--generator-model",
            "gpt-5.2-2025-12-11",
            "--processed-dataset",
            str(dataset_path),
            "--dataset-types",
            "arc_challenge",
            "--models",
            "Qwen/Qwen3-4B-Instruct-2507",
            "--settings",
            "model_from_scratch",
            "--modes",
            "full_question",
            "--questions-per-job",
            "2",
            "--output-dir",
            str(bundle_dir),
            "--write-only",
            "--render-status",
        ]
    )

    assert rc == 0
    assert list(bundle_dir.glob("submissions/*/manifest.json")) == []
    assert (bundle_dir / "scheduler_state.json").exists()
    assert (bundle_dir / "scheduler_status.html").exists()


def test_submit_generate_cluster_submit_calls_master_script_once(tmp_path, monkeypatch):
    dataset_path = tmp_path / "processed"
    bundle_dir = tmp_path / "bundle"
    _processed_dataset(dataset_path, counts={"arc_challenge": 1, "mmlu_pro": 0, "gpqa": 0})
    calls = []

    def fake_submit(paths):
        calls.append(paths.submit_path)
        return CompletedProcess(args=["bash"], returncode=0, stdout="submitted\n", stderr="")

    monkeypatch.setattr(app_main, "submit_bundle", fake_submit)

    rc = app_main.main(
        [
            "submit-generate-cluster",
            "--run-name",
            "cluster-gen",
            "--processed-dataset",
            str(dataset_path),
            "--dataset-types",
            "arc_challenge",
            "--models",
            "Qwen/Qwen3-4B-Instruct-2507",
            "--generation-strategies",
            "model_from_scratch",
            "--output-dir",
            str(bundle_dir),
        ]
    )

    assert rc == 0
    assert len(calls) == 1
    assert calls[0].name == "submit_all.sh"
