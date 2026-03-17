import json

from datasets import Dataset

import main as app_main
from utils.constants import AUGMENTED_STORE_MANIFEST, AUGMENTED_STORE_SCHEMA_VERSION, FINAL5_SETTINGS, SETTING_SPECS
from utils.sharding import sample_id_for_row


def _setting_row(dataset_type: str, base_row: dict[str, object], setting: str) -> dict[str, object]:
    model_count = int(SETTING_SPECS[setting]["num_model"])
    return {
        "id": base_row.get("id"),
        "question_id": base_row.get("question_id"),
        "dataset_type": dataset_type,
        "row_index": 0,
        "sample_id": sample_id_for_row(dataset_type, base_row, 0),
        "question": str(base_row["question"]),
        "num_model": model_count,
        "model_distractors": [f"{setting}_distractor_{index}" for index in range(model_count)],
    }


def test_diagnose_failures_reads_setting_scoped_store(tmp_path, capsys):
    dataset_type = "arc_challenge"
    base_row = {
        "id": "arc-1",
        "question": "ARC question 1",
    }
    root = tmp_path / "augmented"
    root.mkdir(parents=True, exist_ok=True)
    (root / AUGMENTED_STORE_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": AUGMENTED_STORE_SCHEMA_VERSION,
                "storage_kind": "setting_records",
                "dataset_types": [dataset_type],
                "settings": list(FINAL5_SETTINGS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for setting in ("model_from_scratch", "augment_human", "augment_ablation"):
        (root / dataset_type).mkdir(parents=True, exist_ok=True)
        Dataset.from_list([_setting_row(dataset_type, base_row, setting)]).save_to_disk(
            str(root / dataset_type / setting)
        )

    rc = app_main.main(["diagnose-failures", "--dataset-path", str(root)])

    output = capsys.readouterr().out
    assert rc == 1
    assert "arc_challenge: failed_rows=1" in output
    assert '"missing": ["augment_model"]' in output
