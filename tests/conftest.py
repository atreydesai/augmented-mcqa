import os
import shutil
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_ARTIFACTS_ROOT = PROJECT_ROOT / "test-artifacts" / "pytest"


def _configure_test_artifact_roots() -> None:
    shutil.rmtree(TEST_ARTIFACTS_ROOT, ignore_errors=True)

    results_dir = TEST_ARTIFACTS_ROOT / "results"
    datasets_dir = TEST_ARTIFACTS_ROOT / "datasets"
    hf_home = TEST_ARTIFACTS_ROOT / "hf_home"

    results_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)

    os.environ["RESULTS_DIR"] = str(results_dir)
    os.environ["DATASETS_DIR"] = str(datasets_dir)
    os.environ["HF_HOME"] = str(hf_home)


_configure_test_artifact_roots()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
