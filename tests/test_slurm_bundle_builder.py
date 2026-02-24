import json
from pathlib import Path

from scripts import build_eval_slurm_bundle


def _write_regen_manifest(path: Path) -> None:
    payload = {
        "manifest_version": 1,
        "generators": [
            {
                "model": "gpt-5.2-2025-12-11",
                "dataset_path": "datasets/augmented/gpt-5.2-2025-12-11",
                "returncode": 0,
            },
            {
                "model": "claude-opus-4-6",
                "dataset_path": "datasets/augmented/claude-opus-4-6",
                "returncode": 0,
            },
            {
                "model": "gemini-3.1-pro-preview",
                "dataset_path": "datasets/augmented/gemini-3.1-pro-preview",
                "returncode": 0,
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_eval_slurm_bundle_generates_expected_shard_math(tmp_path, monkeypatch):
    regen_manifest = tmp_path / "regen_manifest.json"
    _write_regen_manifest(regen_manifest)

    output_dir = tmp_path / "bundle"

    argv = [
        "--manifest",
        str(regen_manifest),
        "--output-dir",
        str(output_dir),
        "--num-gpus",
        "8",
        "--entry-shards",
        "4",
        "--output-base",
        "results",
    ]
    monkeypatch.setattr("sys.argv", ["build_eval_slurm_bundle.py", *argv])

    rc = build_eval_slurm_bundle.main()
    assert rc == 0

    manifest_path = output_dir / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Base groups = 3 generators * 3 eval models * 2 modes = 18
    assert manifest["total_base_groups"] == 18
    # Job groups = base groups * entry_shards = 18 * 4 = 72
    assert manifest["total_job_groups"] == 72
    # Array tasks = job groups * num_gpus = 72 * 8 = 576
    assert manifest["total_array_tasks"] == 576
    # Eval rows = 3 gen * 3 eval * 2 modes * 3 datasets * 5 settings * 1000 limit
    assert manifest["expected_eval_rows"] == 270000

    sbatch_files = sorted(output_dir.glob("*.sbatch"))
    assert len(sbatch_files) == 72

    sample = sbatch_files[0].read_text(encoding="utf-8")
    assert "--entry-shards \"$ENTRY_SHARDS\"" in sample
    assert "--num-shards \"$NUM_SHARDS\"" in sample

    submit_all = output_dir / "submit_all.sh"
    assert submit_all.exists()
    assert "sbatch" in submit_all.read_text(encoding="utf-8")
