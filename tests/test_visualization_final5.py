import json
from pathlib import Path

from analysis.visualize import collect_final5_results, plot_final5_pairwise


def _write_summary(path: Path, *, total: int, correct: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {"name": "cfg"},
        "summary": {
            "total": total,
            "correct": correct,
            "accuracy": correct / total,
            "attempted_entries": total,
            "successful_entries": total,
            "failed_entries": 0,
            "entry_failure_count": 0,
            "behavioral_counts": {"G": correct, "H": total - correct, "M": 0, "?": 0},
            "accuracy_by_category": {"cat": correct / total},
        },
        "timing": {"start": "2026-01-01T00:00:00Z", "end": "2026-01-01T00:00:10Z"},
        "entry_failures": [],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_collect_and_plot_final5_outputs_include_random_baseline_and_ci(tmp_path):
    root = tmp_path / "results"
    generator = "gpt-5.2-2025-12-11"
    dataset = "mmlu_pro"
    mode = "full_question"
    eval_models = ["Qwen_Qwen3-4B-Instruct-2507", "allenai_Olmo-3-7B-Instruct"]

    settings = [
        "human_from_scratch",
        "model_from_scratch",
        "augment_human",
        "augment_model",
        "augment_ablation",
    ]

    for model_idx, eval_model in enumerate(eval_models):
        for setting_idx, setting in enumerate(settings):
            total = 100
            correct = 20 + model_idx * 5 + setting_idx
            out = root / generator / eval_model / mode / dataset / setting / "summary.json"
            _write_summary(out, total=total, correct=correct)

    df = collect_final5_results(root)
    assert not df.empty
    assert {"random_baseline", "delta_over_random", "ci_low", "ci_high"}.issubset(df.columns)

    # 3H+0M and 0H+3M both have 4 answer choices -> random 0.25
    baseline_h = df[df["setting"] == "human_from_scratch"].iloc[0]["random_baseline"]
    baseline_m = df[df["setting"] == "model_from_scratch"].iloc[0]["random_baseline"]
    assert baseline_h == 0.25
    assert baseline_m == 0.25

    output_dir = tmp_path / "plots"
    outputs = plot_final5_pairwise(root, output_dir, include_tables=True)

    assert outputs
    png_outputs = [path for path in outputs if path.suffix == ".png"]
    assert len(png_outputs) == 2
    assert (output_dir / "tables" / "final5_results_summary.csv").exists()


def test_plot_final5_pairwise_groups_datasets_side_by_side_into_four_mode_plots(tmp_path):
    root = tmp_path / "results"
    generator = "gpt-5.2-2025-12-11"
    modes = ["full_question", "choices_only"]
    datasets = ["arc_challenge", "mmlu_pro", "gpqa"]
    eval_models = ["Qwen_Qwen3-4B-Instruct-2507", "allenai_Olmo-3-7B-Instruct"]
    settings = [
        "human_from_scratch",
        "model_from_scratch",
        "augment_human",
        "augment_model",
        "augment_ablation",
    ]

    for mode_idx, mode in enumerate(modes):
        for dataset_idx, dataset in enumerate(datasets):
            for model_idx, eval_model in enumerate(eval_models):
                for setting_idx, setting in enumerate(settings):
                    total = 40
                    correct = 10 + mode_idx + dataset_idx + model_idx + setting_idx
                    out = root / generator / eval_model / mode / dataset / setting / "summary.json"
                    _write_summary(out, total=total, correct=correct)

    output_dir = tmp_path / "plots"
    outputs = plot_final5_pairwise(root, output_dir, include_tables=True)

    png_outputs = [path for path in outputs if path.suffix == ".png"]
    assert len(png_outputs) == 4

    csv_outputs = [path for path in outputs if path.suffix == ".csv"]
    assert len(csv_outputs) == 5  # 4 per-plot csv + 1 full summary csv
    assert all(path.parent == output_dir / "tables" for path in csv_outputs)
