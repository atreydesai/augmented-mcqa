import json
from subprocess import CompletedProcess

from datasets import Dataset, DatasetDict

import main as app_main


def _processed_dataset(path):
    DatasetDict(
        {
            "arc_challenge": Dataset.from_list([{"id": "arc-1", "question": "ARC?", "answer": "A"}]),
            "mmlu_pro": Dataset.from_list([{"question_id": 101, "question": "MMLU?", "answer": "B"}]),
            "gpqa": Dataset.from_list([{"id": "gpqa-1", "question": "GPQA?", "answer": "C"}]),
        }
    ).save_to_disk(str(path))


def _read_manifest(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_submit_generate_cluster_write_only_writes_nine_model_dataset_tasks(tmp_path):
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
        ]
    )

    assert rc == 0
    manifest = _read_manifest(bundle_dir / "manifest.json")
    assert manifest["stage"] == "generate"
    assert manifest["task_count"] == 9
    assert {task["dataset_type"] for task in manifest["tasks"]} == {"arc_challenge", "mmlu_pro", "gpqa"}
    assert len({task["model"] for task in manifest["tasks"]}) == 3
    assert (bundle_dir / "submit_all.sh").exists()
    assert next(bundle_dir.glob("*.sbatch")).exists()


def test_submit_generate_cluster_dataset_subset_reduces_manifest(tmp_path):
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
            "--dataset-types",
            "gpqa",
            "--output-dir",
            str(bundle_dir),
            "--write-only",
        ]
    )

    assert rc == 0
    manifest = _read_manifest(bundle_dir / "manifest.json")
    assert manifest["task_count"] == 3
    assert {task["dataset_type"] for task in manifest["tasks"]} == {"gpqa"}


def test_submit_generate_cluster_without_gpu_count_renders_uncapped_array(tmp_path):
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
        ]
    )

    assert rc == 0
    sbatch_text = next(bundle_dir.glob("*.sbatch")).read_text(encoding="utf-8")
    assert "#SBATCH --array=0-8" in sbatch_text
    assert "%4" not in sbatch_text


def test_submit_evaluate_cluster_renders_array_cap_and_dataset_scoped_tasks(tmp_path):
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
            "--gpu-count",
            "4",
            "--output-dir",
            str(bundle_dir),
            "--write-only",
        ]
    )

    assert rc == 0
    manifest = _read_manifest(bundle_dir / "manifest.json")
    assert manifest["stage"] == "evaluate"
    assert manifest["task_count"] == 9
    task = manifest["tasks"][0]
    assert task["argv"][0] == "evaluate"
    assert "--dataset-types" in task["argv"]
    assert "--shard-count" not in task["argv"]
    assert "--shard-index" not in task["argv"]
    assert "--settings" not in task["argv"]
    assert "--modes" not in task["argv"]
    assert "--augmented-dataset" in task["argv"]

    sbatch_text = next(bundle_dir.glob("*.sbatch")).read_text(encoding="utf-8")
    assert "#SBATCH --array=0-8%4" in sbatch_text
    assert "#SBATCH --gres=gpu:rtxa6000:1" in sbatch_text
    assert "logs/slurm/evaluate/cluster-eval" in sbatch_text


def test_submit_generate_cluster_rejects_hosted_models(tmp_path, capsys):
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
            "--models",
            "gpt-5.2-2025-12-11",
            "--output-dir",
            str(bundle_dir),
            "--write-only",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "only support local vllm" in captured.out


def test_submit_evaluate_cluster_can_submit_single_array_script(tmp_path, monkeypatch):
    dataset_path = tmp_path / "processed"
    bundle_dir = tmp_path / "bundle"
    _processed_dataset(dataset_path)
    calls = []

    def fake_submit(paths):
        calls.append(paths.sbatch_path)
        return CompletedProcess(args=["sbatch"], returncode=0, stdout="Submitted batch job 123\n", stderr="")

    monkeypatch.setattr(app_main, "submit_bundle", fake_submit)

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
        ]
    )

    assert rc == 0
    assert len(calls) == 1
    assert calls[0].name.startswith("final5-evaluate-cluster-eval")
