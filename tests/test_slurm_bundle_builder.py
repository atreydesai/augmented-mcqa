import json
from pathlib import Path

from datasets import Dataset, DatasetDict

from scripts import build_eval_slurm_bundle


def _build_generator_dataset(path: Path, *, arc_rows: int, mmlu_rows: int, gpqa_rows: int) -> None:
    arc = Dataset.from_list([{"id": i} for i in range(arc_rows)])
    mmlu = Dataset.from_list([{"id": i} for i in range(mmlu_rows)])
    gpqa = Dataset.from_list([{"id": i} for i in range(gpqa_rows)])
    ds = DatasetDict({
        "arc_challenge": arc,
        "mmlu_pro": mmlu,
        "gpqa": gpqa,
    })
    ds.save_to_disk(str(path))


def _write_regen_manifest(path: Path, datasets_root: Path) -> None:
    gpt_path = datasets_root / "gpt"
    opus_path = datasets_root / "opus"
    gemini_path = datasets_root / "gemini"
    _build_generator_dataset(gpt_path, arc_rows=6, mmlu_rows=6, gpqa_rows=3)
    _build_generator_dataset(opus_path, arc_rows=6, mmlu_rows=6, gpqa_rows=3)
    _build_generator_dataset(gemini_path, arc_rows=6, mmlu_rows=6, gpqa_rows=3)

    payload = {
        "manifest_version": 1,
        "generators": [
            {
                "model": "gpt-5.2-2025-12-11",
                "dataset_path": str(gpt_path),
                "returncode": 0,
            },
            {
                "model": "claude-opus-4-6",
                "dataset_path": str(opus_path),
                "returncode": 0,
            },
            {
                "model": "gemini-3.1-pro-preview",
                "dataset_path": str(gemini_path),
                "returncode": 0,
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_eval_slurm_bundle_generates_expected_shard_math(tmp_path, monkeypatch):
    regen_manifest = tmp_path / "regen_manifest.json"
    _write_regen_manifest(regen_manifest, tmp_path / "generated_ds")

    output_dir = tmp_path / "bundle"

    argv = [
        "--manifest",
        str(regen_manifest),
        "--output-dir",
        str(output_dir),
        "--target-rows-per-subsplit",
        "3",
        "--output-base",
        "results",
    ]
    monkeypatch.setattr("sys.argv", ["build_eval_slurm_bundle.py", *argv])

    rc = build_eval_slurm_bundle.main()
    assert rc == 0

    manifest_path = output_dir / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Pair groups = 3 generators * 3 eval models = 9
    assert manifest["total_pairs"] == 9
    assert manifest["total_sbatch_files"] == 9
    # Work-units per pair: 2 modes * (2+2+1) dataset parts = 10
    assert manifest["total_array_tasks"] == 90
    assert manifest["total_work_units"] == 90
    assert manifest["execution_mode"] == "single_job_per_pair_manifest"
    # Eval rows = 3 generators * 15 rows * 3 eval models * 2 modes * 5 settings
    assert manifest["expected_eval_rows"] == 1350

    sbatch_files = sorted(output_dir.glob("*.sbatch"))
    assert len(sbatch_files) == 9
    work_units_files = sorted(output_dir.glob("*.work_units.json"))
    assert len(work_units_files) == 9
    run_manifest_files = sorted(output_dir.glob("*.run_manifest.json"))
    assert len(run_manifest_files) == 9
    sample_units = json.loads(work_units_files[0].read_text(encoding="utf-8"))
    assert len(sample_units) == 10
    sample_run_manifest = json.loads(run_manifest_files[0].read_text(encoding="utf-8"))
    assert sample_run_manifest["summary"]["total"] == 50  # 10 work units * 5 settings
    assert sample_run_manifest["metadata"]["work_unit_count"] == 10

    sample = sbatch_files[0].read_text(encoding="utf-8")
    assert "RUN_MANIFEST_FILE" in sample
    assert "--manifest \"$RUN_MANIFEST_FILE\"" in sample
    assert "#SBATCH --array" not in sample

    submit_all = output_dir / "submit_all.sh"
    assert submit_all.exists()
    assert "sbatch" in submit_all.read_text(encoding="utf-8")
