import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset

from analysis.irt import (
    benchmarker_validity_frame,
    gradient,
    make_design,
    objective,
    run_irt_analysis,
    safe_name,
    setting_difficulty_frame,
    fit_design,
    load_irt_frame,
    taker_ability_frame,
)
from cli.app import main
from utils.constants import EVALUATED_STORE_MANIFEST


def write_group(root: Path, *, generator: str, test_taker: str, rows_by_split: dict[tuple[str, str, str], list[dict[str, object]]]) -> None:
    group_root = root / safe_name("irt_run") / safe_name(generator) / safe_name(test_taker)
    group_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "evaluated_mcqa_setting_mode_records_v1",
        "storage_kind": "evaluated_setting_mode_records",
        "dataset_types": sorted({dataset for dataset, _setting, _mode in rows_by_split}),
        "settings": sorted({setting for _dataset, setting, _mode in rows_by_split}),
        "modes": sorted({mode for _dataset, _setting, mode in rows_by_split}),
        "generation_model": generator,
        "evaluation_model": test_taker,
    }
    (group_root / EVALUATED_STORE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    for (dataset, setting, mode), rows in rows_by_split.items():
        path = group_root / dataset / setting / mode
        path.parent.mkdir(parents=True, exist_ok=True)
        Dataset.from_list(rows).save_to_disk(str(path))


def row(sample_id: str, dataset: str, setting: str, mode: str, num_choices: int, correct: bool) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "question": f"Question {sample_id}",
        "dataset_type": dataset,
        "evaluation_status": "success",
        "evaluation_used_random_fallback": False,
        "evaluation_prediction": "A" if correct else "B",
        "evaluation_is_correct": correct,
        "num_choices": num_choices,
        "setting": setting,
    }


def synthetic_collected(root: Path) -> Path:
    rng = np.random.default_rng(7)
    generators = {"generator_easy": -0.2, "generator_hard": 0.5}
    test_takers = {"weak_taker": -0.6, "strong_taker": 0.7}
    settings = {"human_from_scratch": (0.0, 4), "augment_model": (1.0, 10)}
    stems = {f"item-{i}": -1.2 + 0.12 * i for i in range(24)}

    for generator, generator_difficulty in generators.items():
        for test_taker, theta in test_takers.items():
            rows_by_split = {}
            for setting, (setting_difficulty, num_choices) in settings.items():
                split_rows = []
                for sample_id, stem_difficulty in stems.items():
                    c = 1.0 / num_choices
                    b = stem_difficulty + generator_difficulty + setting_difficulty
                    p = c + (1.0 - c) / (1.0 + np.exp(-(theta - b)))
                    split_rows.append(row(sample_id, "arc_challenge", setting, "full_question", num_choices, bool(rng.random() < p)))
                rows_by_split[("arc_challenge", setting, "full_question")] = split_rows
            write_group(root, generator=generator, test_taker=test_taker, rows_by_split=rows_by_split)
    return root


def test_load_irt_frame_creates_expected_ids(tmp_path):
    frame = load_irt_frame(synthetic_collected(tmp_path / "collected"))
    assert not frame.empty
    assert set(frame["mode"]) == {"full_question"}
    assert set(frame["choice_group"]) == {"4-choice", "10-choice"}
    assert frame["stem_id"].str.contains("::").all()
    assert frame["item_id"].str.contains("::").all()


def test_load_irt_frame_excludes_unusable_evaluations_but_keeps_fallbacks(tmp_path):
    rows = [
        row("kept-correct", "arc_challenge", "human_from_scratch", "full_question", 4, True),
        row("kept-wrong", "arc_challenge", "human_from_scratch", "full_question", 4, False),
        {
            **row("missing-status", "arc_challenge", "human_from_scratch", "full_question", 4, False),
            "evaluation_status": "missing",
        },
        {
            **row("random-fallback", "arc_challenge", "human_from_scratch", "full_question", 4, False),
            "evaluation_used_random_fallback": True,
        },
        {
            **row("blank-prediction", "arc_challenge", "human_from_scratch", "full_question", 4, False),
            "evaluation_prediction": "",
        },
        {
            **row("null-correct", "arc_challenge", "human_from_scratch", "full_question", 4, False),
            "evaluation_is_correct": None,
        },
    ]
    write_group(
        tmp_path / "collected",
        generator="generator",
        test_taker="taker",
        rows_by_split={("arc_challenge", "human_from_scratch", "full_question"): rows},
    )

    frame = load_irt_frame(tmp_path / "collected")
    assert set(frame["sample_id"]) == {"kept-correct", "kept-wrong", "random-fallback"}
    assert frame.set_index("sample_id")["correct"].to_dict() == {
        "kept-correct": 1.0,
        "kept-wrong": 0.0,
        "random-fallback": 0.0,
    }


def test_gradient_matches_finite_difference(tmp_path):
    design = make_design(load_irt_frame(synthetic_collected(tmp_path / "collected")))
    beta = np.linspace(-0.05, 0.05, design.n_params)
    analytic = gradient(beta, design)
    eps = 1e-6
    for idx in (0, min(4, design.n_params - 1), design.n_params - 1):
        step = np.zeros(design.n_params)
        step[idx] = eps
        numeric = (objective(beta + step, design) - objective(beta - step, design)) / (2.0 * eps)
        assert abs(analytic[idx] - numeric) < 1e-4


def test_fit_recovers_basic_ordering(tmp_path):
    design = make_design(load_irt_frame(synthetic_collected(tmp_path / "collected")))
    fit = fit_design(design, maxiter=80)
    assert fit.success

    settings = setting_difficulty_frame(fit)
    human = float(settings.loc[settings["setting"] == "human_from_scratch", "estimate"].iloc[0])
    augment = float(settings.loc[settings["setting"] == "augment_model", "estimate"].iloc[0])
    assert augment > human

    takers = taker_ability_frame(fit)
    weak = float(takers.loc[takers["test_taker"] == "weak_taker", "estimate"].iloc[0])
    strong = float(takers.loc[takers["test_taker"] == "strong_taker", "estimate"].iloc[0])
    assert strong > weak


def test_run_irt_analysis_writes_outputs(tmp_path):
    outputs = run_irt_analysis(collected_root=synthetic_collected(tmp_path / "collected"), output_dir=tmp_path / "irt", maxiter=50)
    names = {path.name for path in outputs}
    assert "setting_difficulty.csv" in names
    assert "dataset_difficulty.csv" in names
    assert "generator_difficulty.csv" in names
    assert "test_taker_ability.csv" in names
    assert "instantiated_item_parameters.csv" in names
    assert "irt_quality_by_dataset_setting.csv" in names
    assert "benchmarker_validity_by_dataset_setting.csv" in names
    assert "question_quality_by_dataset_setting.csv" in names
    assert "fit_summary.json" in names
    assert "setting_difficulty.png" in names
    assert "question_quality_all_settings.png" in names
    assert "item_fit.png" in names


def test_benchmarker_validity_frame_uses_cached_table(tmp_path):
    table_path = tmp_path / "writing_flaw_rows.csv"
    pd.DataFrame(
        [
            {
                "dataset": "arc_challenge",
                "config": "human_from_scratch",
                "generator_model": "openai/gpt-5.2-2025-12-11",
                "question": "Question 1",
                "flaw_value": 0.75,
                "n_flaws": 3,
            }
        ]
    ).to_csv(table_path, index=False)

    frame = benchmarker_validity_frame(path=tmp_path / "missing.jsonl", table_path=table_path)
    assert list(frame["dataset"]) == ["arc_challenge"]
    assert list(frame["setting"]) == ["human_from_scratch"]
    assert float(frame.loc[0, "mean_flaws"]) == 3.0
    assert float(frame.loc[0, "mean_flaws_se"]) == 0.0


def test_cli_analyze_irt_smoke(tmp_path):
    output_dir = tmp_path / "irt-cli"
    rc = main(
        [
            "analyze-irt",
            "--collected-root",
            str(synthetic_collected(tmp_path / "collected")),
            "--output-dir",
            str(output_dir),
            "--maxiter",
            "40",
        ]
    )
    assert rc == 0
    assert (output_dir / "fit_summary.json").exists()
