from importlib import import_module
from pathlib import Path


def test_build_eval_slurm_bundle_writes_thin_array_wrappers(tmp_path, monkeypatch):
    mod = import_module("scripts.05_build_eval_slurm_bundle")
    out_dir = tmp_path / "bundle"
    argv = [
        "build_eval_slurm_bundle.py",
        "--run-name",
        "eval-run",
        "--generator-run-name",
        "gen-run",
        "--generator-model",
        "gpt-5.2-2025-12-11",
        "--output-dir",
        str(out_dir),
        "--shard-count",
        "4",
    ]
    monkeypatch.setattr("sys.argv", argv)
    assert mod.main() == 0

    submit = out_dir / "submit_all.sh"
    sbatch_files = sorted(out_dir.glob("*.sbatch"))
    assert submit.exists()
    assert len(sbatch_files) == 3
    sample = sbatch_files[0].read_text(encoding="utf-8")
    assert "#SBATCH --array=0-3" in sample
    assert 'main.py evaluate \\' in sample
    assert '--shard-index "${SLURM_ARRAY_TASK_ID}"' in sample
