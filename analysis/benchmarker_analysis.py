"""Benchmarker writing-flaw analysis for Inspect-native Final5 runs.

Example:
  uv run python analysis/benchmarker_analysis.py \
    --writing-flaw-jsonl /path/to/writing_flaw_rows.jsonl \
    --results-root results/inspect/evaluation \
    --generator-run-name my-generation-run \
    --generator-model gpt-5.2-2025-12-11
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import zipfile
from pathlib import Path
from typing import Iterable, Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from data.final5_store import _load_dataset_dict
from utils.constants import (
    CHOICE_LABELS,
    DEFAULT_AUGMENTED_CACHE_ROOT,
    DEFAULT_EVALUATION_LOG_ROOT,
    FINAL5_SETTINGS,
)
from utils.logs import iter_eval_logs
from utils.modeling import resolve_model_name, safe_name
from utils.sharding import sample_id_for_row


CONFIG_ORDER = list(FINAL5_SETTINGS)
CONFIG_LABELS = {
    "human_from_scratch": "Human\n(4-choice)",
    "model_from_scratch": "Model\n(4-choice)",
    "augment_human": "Aug-Human\n(10-choice)",
    "augment_model": "Aug-Model\n(10-choice)",
    "augment_ablation": "Aug-Ablation\n(10-choice)",
}
CONFIG_SHORT = {
    "human_from_scratch": "Hum-4",
    "model_from_scratch": "Mdl-4",
    "augment_human": "AugH-10",
    "augment_model": "AugM-10",
    "augment_ablation": "AugA-10",
}
DATASET_ORDER = ["arc_challenge", "gpqa", "mmlu_pro"]
DATASET_LABELS = {"arc_challenge": "ARC-Challenge", "gpqa": "GPQA", "mmlu_pro": "MMLU-Pro"}
RULES_ORDER = [
    "avoid_k_type",
    "avoid_negatives",
    "avoid_repetition",
    "clear_language",
    "equal_length_options",
    "focused_stem",
    "grammatical_consistency",
    "no_absolute_terms",
    "no_all_of_the_above",
    "no_convergence_cues",
    "no_extraneous_info",
    "no_fill_in_blank",
    "no_logical_cues",
    "no_none_of_the_above",
    "no_vague_terms",
    "ordered_options",
    "plausible_distractors",
    "problem_in_stem",
    "single_best_answer",
]
PALETTE = {
    "human_from_scratch": "#2196F3",
    "model_from_scratch": "#FF9800",
    "augment_human": "#4CAF50",
    "augment_model": "#F44336",
    "augment_ablation": "#9C27B0",
}
DATASET_PALETTE = {
    "arc_challenge": "#1976D2",
    "gpqa": "#388E3C",
    "mmlu_pro": "#D32F2F",
}
K_MAP = {
    "human_from_scratch": 4,
    "model_from_scratch": 4,
    "augment_human": 10,
    "augment_model": 10,
    "augment_ablation": 10,
}


def _csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _display_model(value: str) -> str:
    raw = str(value).replace("/", "_")
    mapping = {
        "Qwen_Qwen3-4B-Instruct-2507": "Qwen3-4B",
        "allenai_Olmo-3-7B-Instruct": "Olmo3-7B",
        "meta-llama_Llama-3.1-8B-Instruct": "Llama3.1-8B",
    }
    return mapping.get(raw, raw)


def _mean_ci(series: pd.Series, z: float = 1.96) -> tuple[float, float]:
    clean = pd.Series(series).dropna().astype(float)
    if clean.empty:
        return float("nan"), float("nan")
    mean = float(clean.mean())
    if len(clean) == 1:
        return mean, 0.0
    se = float(clean.std(ddof=1) / np.sqrt(len(clean)))
    return mean, se * z


def _safe_pearsonr(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 2:
        return float("nan"), float("nan")
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return float("nan"), float("nan")
    result = stats.pearsonr(x_arr, y_arr)
    return float(result.statistic), float(result.pvalue)


def _safe_regression(x: Iterable[float], y: Iterable[float]) -> tuple[float, float] | None:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 2:
        return None
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return None
    fit = stats.linregress(x_arr, y_arr)
    return float(fit.slope), float(fit.intercept)


def _iter_jsonl(path: Path) -> Iterator[dict]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            member = next((name for name in zf.namelist() if name.endswith(".jsonl")), None)
            if member is None:
                raise FileNotFoundError(f"No .jsonl file found in {path}")
            with zf.open(member) as handle:
                for raw_line in handle:
                    line = raw_line.decode("utf-8").strip()
                    if line:
                        yield json.loads(line)
        return

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _parse_rule_answers(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def load_writing_flaw_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for obj in _iter_jsonl(path):
        writing_flaw = dict(obj.get("writing_flaw", {}) or {})
        answers = _parse_rule_answers(writing_flaw.get("answer"))
        outcomes = [(item.strip().lower() == "pass") for item in answers[: len(RULES_ORDER)]]
        rows.append(
            {
                "dataset": str(obj.get("dataset", "")),
                "config": str(obj.get("config", "")),
                "question": str(obj.get("question", "")),
                "flaw_value": float(writing_flaw.get("value", float("nan"))),
                "n_flaws": sum(not passed for passed in outcomes),
                **{
                    f"rule_{rule}": (outcomes[i] if i < len(outcomes) else np.nan)
                    for i, rule in enumerate(RULES_ORDER)
                },
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No writing flaw rows found in {path}")
    df["config"] = pd.Categorical(df["config"], categories=CONFIG_ORDER, ordered=True)
    df["has_ge2_flaws"] = df["n_flaws"] >= 2

    rule_cols = [f"rule_{rule}" for rule in RULES_ORDER]
    df_long = df.melt(
        id_vars=["dataset", "config", "question", "flaw_value"],
        value_vars=rule_cols,
        var_name="rule_col",
        value_name="passed",
    )
    df_long = df_long.dropna(subset=["passed"]).copy()
    df_long["rule"] = df_long["rule_col"].str.replace("rule_", "", regex=False)
    df_long["failed"] = ~df_long["passed"].astype(bool)
    return df, df_long


def _model_matches(filter_value: str | None, actual: str) -> bool:
    if not filter_value:
        return True
    normalized = resolve_model_name(filter_value)
    actual = str(actual)
    return actual == normalized or actual == filter_value or actual.endswith(f"/{filter_value}")


def _resolve_augmented_dataset_path(
    *,
    augmented_dataset: str | None,
    cache_root: str | Path,
    generator_model: str | None,
    generator_run_name: str | None,
) -> Path:
    if augmented_dataset:
        return Path(augmented_dataset)

    root = Path(cache_root)
    if (root / "dataset_dict.json").exists():
        return root

    if generator_model and generator_run_name:
        resolved_model = resolve_model_name(generator_model)
        candidate = root / safe_name(generator_run_name) / safe_name(resolved_model)
        if (candidate / "dataset_dict.json").exists():
            return candidate

    candidates = sorted(path.parent for path in root.glob("**/dataset_dict.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No augmented dataset cache found under {root}. "
            "Pass --augmented-dataset explicitly or provide --cache-root plus "
            "--generator-run-name and --generator-model."
        )
    if len(candidates) == 1:
        return candidates[0]

    preview = ", ".join(str(path) for path in candidates[:5])
    raise ValueError(
        f"Multiple augmented dataset caches found under {root}: {preview}. "
        "Pass --augmented-dataset explicitly or provide --generator-run-name and --generator-model."
    )


def load_evaluation_samples(
    *,
    results_root: Path,
    generator_model: str | None,
    generator_run_name: str | None,
    eval_models: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for log_path, log in iter_eval_logs(results_root, kind="evaluation"):
        log_meta = dict(getattr(log.eval, "metadata", {}) or {})
        generation_model = str(log_meta.get("generation_model", ""))
        logged_generation_run_name = str(log_meta.get("generation_run_name", ""))
        evaluation_model = str(log_meta.get("evaluation_model") or log.eval.model)
        if not _model_matches(generator_model, generation_model):
            continue
        if generator_run_name and logged_generation_run_name != str(generator_run_name):
            continue
        if eval_models and not any(_model_matches(model, evaluation_model) for model in eval_models):
            continue

        for sample in getattr(log, "samples", []):
            if not sample.scores:
                continue
            score = next(iter(sample.scores.values()))
            score_meta = dict(getattr(score, "metadata", {}) or {})
            dataset = str(score_meta.get("dataset_type", ""))
            setting = str(score_meta.get("setting") or log_meta.get("setting", ""))
            mode = str(score_meta.get("mode") or log_meta.get("mode", ""))
            if not dataset or setting not in CONFIG_ORDER:
                continue
            rows.append(
                {
                    "results_path": str(log_path),
                    "generation_model": generation_model,
                    "generation_run_name": logged_generation_run_name,
                    "eval_model": evaluation_model,
                    "dataset": dataset,
                    "setting": setting,
                    "mode": mode,
                    "sample_id": str(score_meta.get("sample_id") or sample.id),
                    "question_idx": int(score_meta.get("question_idx", -1)),
                    "category": str(score_meta.get("category", "")),
                    "prediction": str(score_meta.get("prediction", "") or "").strip().upper(),
                    "prediction_type": str(score_meta.get("prediction_type", "?")),
                    "gold_answer_letter": str(score_meta.get("gold_answer_letter", "") or "").strip().upper(),
                    "is_correct": bool(getattr(score, "value", False)),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No matching evaluation rows found under {results_root}")

    generation_models = sorted({value for value in df["generation_model"].dropna().unique() if value})
    if not generator_model and len(generation_models) > 1:
        raise ValueError(
            "Multiple generation models found in results; pass --generator-model to disambiguate: "
            + ", ".join(generation_models)
        )
    run_names = sorted({value for value in df["generation_run_name"].dropna().unique() if value})
    if not generator_run_name and len(run_names) > 1:
        raise ValueError(
            "Multiple generation run names found in results; pass --generator-run-name to disambiguate: "
            + ", ".join(run_names)
        )
    return df


def summarize_accuracy(eval_samples: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        eval_samples.groupby(
            ["generation_model", "generation_run_name", "eval_model", "mode", "dataset", "setting"],
            observed=True,
        )["is_correct"]
        .agg(total="count", correct="sum")
        .reset_index()
    )
    grouped["accuracy"] = grouped["correct"] / grouped["total"]
    grouped["stderr"] = grouped.apply(
        lambda row: math.sqrt(max(0.0, row["accuracy"] * (1.0 - row["accuracy"])) / row["total"]) if row["total"] else 0.0,
        axis=1,
    )
    grouped["random_baseline"] = grouped["setting"].map(lambda setting: 1.0 / K_MAP[str(setting)])
    grouped["delta_over_random"] = grouped["accuracy"] - grouped["random_baseline"]
    grouped["config"] = pd.Categorical(grouped["setting"], categories=CONFIG_ORDER, ordered=True)
    return grouped


def load_augmented_rows(path: Path) -> pd.DataFrame:
    dataset_dict = _load_dataset_dict(path)
    rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        if dataset not in dataset_dict:
            continue
        split = dataset_dict[dataset]
        for row_index, row in enumerate(split):
            payload = dict(row)
            sample_id = sample_id_for_row(dataset, payload, row_index)
            question = str(payload.get("question", ""))
            for setting in CONFIG_ORDER:
                options = list(payload.get(f"{setting}_options_randomized") or [])
                gold_letter = str(payload.get(f"{setting}_correct_answer_letter", "") or "")
                if not options or not gold_letter:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "config": setting,
                        "sample_id": sample_id,
                        "question_idx": row_index,
                        "question": question,
                        "gold_answer_letter": gold_letter,
                        "options_randomized": options,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No augmented rows found at {path}")
    df["config"] = pd.Categorical(df["config"], categories=CONFIG_ORDER, ordered=True)
    return df


def build_per_question_dataframe(
    *,
    eval_samples: pd.DataFrame,
    augmented_rows: pd.DataFrame,
    flaw_df: pd.DataFrame,
) -> pd.DataFrame:
    full_question = eval_samples[eval_samples["mode"] == "full_question"].copy()
    if full_question.empty:
        return pd.DataFrame()

    full_question["config"] = full_question["setting"]
    merged = full_question.merge(
        augmented_rows,
        on=["dataset", "sample_id", "config"],
        how="left",
        suffixes=("", "_augmented"),
    )
    rule_cols = [f"rule_{rule}" for rule in RULES_ORDER]
    merged = merged.merge(
        flaw_df[["dataset", "config", "question", "flaw_value", *rule_cols]],
        on=["dataset", "config", "question"],
        how="left",
    )
    return merged


def _grouped_bar_quality(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric_col: str,
    *,
    ylabel: str,
    title: str,
    ci_col: str | None = None,
) -> None:
    n_configs = len(CONFIG_ORDER)
    x = np.arange(len(DATASET_ORDER))
    width = 0.14
    offsets = np.linspace(-(n_configs - 1) / 2, (n_configs - 1) / 2, n_configs) * width

    for i, config in enumerate(CONFIG_ORDER):
        vals: list[float] = []
        errs: list[float] = []
        for dataset in DATASET_ORDER:
            row = data[(data["dataset"] == dataset) & (data["config"] == config)]
            if row.empty:
                vals.append(0.0)
                errs.append(0.0)
                continue
            vals.append(float(row[metric_col].iloc[0]))
            errs.append(float(row[ci_col].iloc[0]) if ci_col else 0.0)
        ax.bar(
            x + offsets[i],
            vals,
            width=width * 0.9,
            label=CONFIG_LABELS[config].replace("\n", " "),
            color=PALETTE[config],
            alpha=0.85,
            yerr=errs if ci_col else None,
            capsize=3,
            error_kw={"elinewidth": 1},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[dataset] for dataset in DATASET_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    path = output_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def _full_question_accuracy(acc_df: pd.DataFrame) -> pd.DataFrame:
    full_question = acc_df[acc_df["mode"] == "full_question"].copy()
    grouped = (
        full_question.groupby(["dataset", "config"], observed=True)
        .agg(
            mean_acc=("accuracy", "mean"),
            se_acc=("accuracy", lambda x: float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0),
            mean_delta=("delta_over_random", "mean"),
        )
        .reset_index()
    )
    grouped["config"] = pd.Categorical(grouped["config"], categories=CONFIG_ORDER, ordered=True)
    return grouped


def _print_quality_summary(summary_df: pd.DataFrame) -> None:
    print("\nReproduced summary:")
    print(f"{'dataset':<20} {'config':<25} {'flaw_value':>12} {'p(>=2 flaws)':>14} {'n':>6}")
    print("-" * 80)
    for _, row in summary_df.iterrows():
        print(
            f"{row['dataset']:<20} {row['config']:<25} "
            f"{row['writing_flaws_mean']:.3f} +/- {row['writing_flaws_ci']:.3f}  "
            f"{row['p_ge2_mean']*100:.2f}% +/- {row['p_ge2_ci']*100:.2f}%  "
            f"{int(row['n']):>6}"
        )


def run_analysis(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    augmented_dataset_path = _resolve_augmented_dataset_path(
        augmented_dataset=args.augmented_dataset,
        cache_root=args.cache_root,
        generator_model=args.generator_model,
        generator_run_name=args.generator_run_name,
    )

    print("Loading writing flaw data ...")
    flaw_df, flaw_long_df = load_writing_flaw_data(Path(args.writing_flaw_jsonl))
    print(f"  Loaded {len(flaw_df):,} rows.")

    print("Loading evaluation logs ...")
    eval_samples = load_evaluation_samples(
        results_root=Path(args.results_root),
        generator_model=args.generator_model,
        generator_run_name=args.generator_run_name,
        eval_models=_csv_list(args.eval_models),
    )
    print(f"  Loaded {len(eval_samples):,} evaluation samples.")

    print("Loading augmented dataset cache ...")
    print(f"  Using augmented dataset cache at {augmented_dataset_path}")
    augmented_rows = load_augmented_rows(augmented_dataset_path)
    print(f"  Loaded {len(augmented_rows):,} augmented setting rows.")

    print("\n" + "=" * 70)
    print("SECTION 1: WRITING QUALITY SUMMARY")
    print("=" * 70)
    summary_rows: list[dict[str, object]] = []
    for (dataset, config), group in flaw_df.groupby(["dataset", "config"], observed=True):
        mean_flaw, ci_flaw = _mean_ci(group["flaw_value"])
        mean_p2, ci_p2 = _mean_ci(group["has_ge2_flaws"].astype(float))
        summary_rows.append(
            {
                "dataset": dataset,
                "config": config,
                "writing_flaws_mean": mean_flaw,
                "writing_flaws_ci": ci_flaw,
                "p_ge2_mean": mean_p2,
                "p_ge2_ci": ci_p2,
                "n": len(group),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df["config"] = pd.Categorical(summary_df["config"], categories=CONFIG_ORDER, ordered=True)
    summary_df = summary_df.sort_values(["dataset", "config"]).reset_index(drop=True)
    _print_quality_summary(summary_df)

    print("\n" + "=" * 70)
    print("SECTION 2: WRITING QUALITY FIGURES")
    print("=" * 70)
    fig2a, ax2a = plt.subplots(figsize=(8, 5))
    _grouped_bar_quality(
        ax2a,
        summary_df,
        "writing_flaws_mean",
        ylabel="Mean fraction of rules passed (higher is better)",
        title="Fig 2a: Writing Quality by Config x Dataset",
        ci_col="writing_flaws_ci",
    )
    ax2a.set_ylim(0.0, 1.0)
    _save(fig2a, output_dir, "fig2a_quality_mean.png")

    fig2b, ax2b = plt.subplots(figsize=(8, 5))
    _grouped_bar_quality(
        ax2b,
        summary_df,
        "p_ge2_mean",
        ylabel="P(>=2 writing flaws) lower is better",
        title="Fig 2b: Fraction of Questions with >=2 Writing Flaws",
        ci_col="p_ge2_ci",
    )
    ax2b.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2b.set_ylim(0.0, 1.05)
    _save(fig2b, output_dir, "fig2b_p_ge2_flaws.png")

    choice_group_df = flaw_df.copy()
    choice_group_df["choice_group"] = choice_group_df["config"].map(
        lambda config: "4-choice\n(from_scratch)" if "from_scratch" in str(config) else "10-choice\n(augment_*)"
    )
    grouped_4v10 = (
        choice_group_df.groupby(["dataset", "choice_group"], observed=True)
        .agg(
            mean_flaw=("flaw_value", "mean"),
            ci_flaw=("flaw_value", lambda x: _mean_ci(pd.Series(x))[1]),
            mean_p2=("has_ge2_flaws", "mean"),
            ci_p2=("has_ge2_flaws", lambda x: _mean_ci(pd.Series(x).astype(float))[1]),
        )
        .reset_index()
    )
    fig2c, axes2c = plt.subplots(1, 2, figsize=(11, 5))
    groups = ["4-choice\n(from_scratch)", "10-choice\n(augment_*)"]
    x = np.arange(len(DATASET_ORDER))
    width = 0.3
    colors = ["#2196F3", "#F44336"]
    for ax, metric, ci_col, ylabel in [
        (axes2c[0], "mean_flaw", "ci_flaw", "Mean fraction of rules passed"),
        (axes2c[1], "mean_p2", "ci_p2", "P(>=2 writing flaws)"),
    ]:
        for offset, (group, color) in enumerate(zip(groups, colors)):
            vals: list[float] = []
            errs: list[float] = []
            for dataset in DATASET_ORDER:
                row = grouped_4v10[(grouped_4v10["dataset"] == dataset) & (grouped_4v10["choice_group"] == group)]
                vals.append(float(row[metric].iloc[0]) if not row.empty else 0.0)
                errs.append(float(row[ci_col].iloc[0]) if not row.empty else 0.0)
            ax.bar(
                x + (offset - 0.5) * width,
                vals,
                width=width * 0.9,
                label=group.replace("\n", " "),
                color=color,
                alpha=0.85,
                yerr=errs,
                capsize=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[dataset] for dataset in DATASET_ORDER])
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    axes2c[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axes2c[0].set_title("Fig 2c (left): Quality - 4-choice vs 10-choice")
    axes2c[1].set_title("Fig 2c (right): P(>=2 flaws) - 4-choice vs 10-choice")
    _save(fig2c, output_dir, "fig2c_4choice_vs_10choice.png")

    print("\n" + "=" * 70)
    print("SECTION 3: PER-RULE FAILURE RATE ANALYSIS")
    print("=" * 70)
    rule_fail = (
        flaw_long_df.groupby(["dataset", "config", "rule"], observed=True)["failed"]
        .mean()
        .reset_index()
        .rename(columns={"failed": "fail_rate"})
    )
    fig3a, axes3a = plt.subplots(1, len(DATASET_ORDER), figsize=(18, 8))
    if len(DATASET_ORDER) == 1:
        axes3a = [axes3a]
    for ax, dataset in zip(axes3a, DATASET_ORDER):
        pivot = (
            rule_fail[rule_fail["dataset"] == dataset]
            .pivot(index="rule", columns="config", values="fail_rate")
            .reindex(index=RULES_ORDER, columns=CONFIG_ORDER)
        )
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7},
            linewidths=0.4,
            cbar_kws={"shrink": 0.6},
            xticklabels=[CONFIG_SHORT[config] for config in CONFIG_ORDER],
        )
        ax.set_title(f"Fig 3a: {DATASET_LABELS[dataset]}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Rule" if dataset == DATASET_ORDER[0] else "")
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
    _save(fig3a, output_dir, "fig3a_rule_heatmap.png")

    sensitivity_rows: list[dict[str, object]] = []
    augmentation_rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        for rule in RULES_ORDER:
            sub = rule_fail[(rule_fail["dataset"] == dataset) & (rule_fail["rule"] == rule)]
            hum = float(sub[sub["config"] == "human_from_scratch"]["fail_rate"].iloc[0]) if not sub[sub["config"] == "human_from_scratch"].empty else 0.0
            mdl = float(sub[sub["config"] == "model_from_scratch"]["fail_rate"].iloc[0]) if not sub[sub["config"] == "model_from_scratch"].empty else 0.0
            aug = float(sub[sub["config"] == "augment_human"]["fail_rate"].iloc[0]) if not sub[sub["config"] == "augment_human"].empty else 0.0
            sensitivity_rows.append({"dataset": dataset, "rule": rule, "delta": mdl - hum})
            augmentation_rows.append({"dataset": dataset, "rule": rule, "delta": aug - hum})
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_avg = sensitivity_df.groupby("rule")["delta"].mean().reindex(RULES_ORDER)
    fig3b, ax3b = plt.subplots(figsize=(11, 6))
    ax3b.barh(
        RULES_ORDER,
        sensitivity_avg.values,
        color=["#F44336" if value > 0 else "#2196F3" for value in sensitivity_avg.values],
        alpha=0.85,
    )
    ax3b.axvline(0, color="black", linewidth=0.8)
    ax3b.set_xlabel("Delta fail rate (model_from_scratch - human_from_scratch)")
    ax3b.set_title("Fig 3b: Model-Sensitivity by Rule")
    ax3b.tick_params(axis="y", labelsize=8)
    _save(fig3b, output_dir, "fig3b_model_sensitivity.png")

    augmentation_df = pd.DataFrame(augmentation_rows)
    augmentation_avg = augmentation_df.groupby("rule")["delta"].mean().reindex(RULES_ORDER)
    fig3c, ax3c = plt.subplots(figsize=(11, 6))
    ax3c.barh(
        RULES_ORDER,
        augmentation_avg.values,
        color=["#F44336" if value > 0 else "#2196F3" for value in augmentation_avg.values],
        alpha=0.85,
    )
    ax3c.axvline(0, color="black", linewidth=0.8)
    ax3c.set_xlabel("Delta fail rate (augment_human - human_from_scratch)")
    ax3c.set_title("Fig 3c: Augmentation Penalty by Rule")
    ax3c.tick_params(axis="y", labelsize=8)
    _save(fig3c, output_dir, "fig3c_augmentation_penalty.png")

    print("\n" + "=" * 70)
    print("SECTION 4: ACCURACY ANALYSIS")
    print("=" * 70)
    acc_df = summarize_accuracy(eval_samples)
    full_question_acc = _full_question_accuracy(acc_df)

    fig4a, ax4a = plt.subplots(figsize=(8, 5))
    _grouped_bar_quality(
        ax4a,
        full_question_acc.rename(columns={"mean_acc": "metric"}),
        "metric",
        ylabel="Mean accuracy (full_question)",
        title="Fig 4a: Accuracy by Config x Dataset",
    )
    ax4a.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    _save(fig4a, output_dir, "fig4a_accuracy_avg.png")

    eval_models = sorted(acc_df["eval_model"].unique())
    fig4b, axes4b = plt.subplots(1, len(eval_models), figsize=(max(6, 5 * len(eval_models)), 5), sharey=True)
    if len(eval_models) == 1:
        axes4b = [axes4b]
    for ax, eval_model in zip(axes4b, eval_models):
        sub = acc_df[(acc_df["mode"] == "full_question") & (acc_df["eval_model"] == eval_model)]
        sub_grouped = (
            sub.groupby(["dataset", "config"], observed=True)
            .agg(mean_acc=("accuracy", "mean"), se_acc=("accuracy", lambda x: float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0))
            .reset_index()
        )
        n_configs = len(CONFIG_ORDER)
        x = np.arange(len(DATASET_ORDER))
        width = 0.14
        offsets = np.linspace(-(n_configs - 1) / 2, (n_configs - 1) / 2, n_configs) * width
        for i, config in enumerate(CONFIG_ORDER):
            vals: list[float] = []
            errs: list[float] = []
            for dataset in DATASET_ORDER:
                row = sub_grouped[(sub_grouped["dataset"] == dataset) & (sub_grouped["config"] == config)]
                vals.append(float(row["mean_acc"].iloc[0]) if not row.empty else 0.0)
                errs.append(float(row["se_acc"].iloc[0]) if not row.empty else 0.0)
            ax.bar(
                x + offsets[i],
                vals,
                width=width * 0.9,
                color=PALETTE[config],
                alpha=0.85,
                label=CONFIG_SHORT[config],
                yerr=errs,
                capsize=2,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[dataset] for dataset in DATASET_ORDER], fontsize=8)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_title(_display_model(eval_model), fontsize=9)
        ax.legend(fontsize=6, ncol=2)
    axes4b[0].set_ylabel("Accuracy")
    _save(fig4b, output_dir, "fig4b_accuracy_by_model.png")

    fig4c, ax4c = plt.subplots(figsize=(8, 5))
    _grouped_bar_quality(
        ax4c,
        full_question_acc.rename(columns={"mean_delta": "metric"}),
        "metric",
        ylabel="Delta over random baseline",
        title="Fig 4c: Accuracy Delta Over Random Baseline",
    )
    ax4c.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    _save(fig4c, output_dir, "fig4c_delta_over_random.png")

    print("\n" + "=" * 70)
    print("SECTION 5: VALIDITY vs ACCURACY")
    print("=" * 70)
    validity_df = (
        flaw_df.groupby(["dataset", "config"], observed=True)
        .agg(mean_validity=("flaw_value", "mean"), mean_p2=("has_ge2_flaws", "mean"))
        .reset_index()
    )
    cross = validity_df.merge(
        full_question_acc.rename(columns={"mean_acc": "mean_accuracy"})[["dataset", "config", "mean_accuracy"]],
        on=["dataset", "config"],
        how="inner",
    )
    cross["choice_group"] = cross["config"].map(
        lambda config: "4-choice" if "from_scratch" in str(config) else "10-choice"
    )
    r_val, p_val = _safe_pearsonr(cross["mean_validity"], cross["mean_accuracy"])
    r_p2, p_p2 = _safe_pearsonr(cross["mean_p2"], cross["mean_accuracy"])
    print(f"  Pearson r(validity, accuracy): r={r_val:.3f}, p={p_val:.4f}")
    print(f"  Pearson r(p>=2 flaws, accuracy): r={r_p2:.3f}, p={p_p2:.4f}")

    fig5a, ax5a = plt.subplots(figsize=(8, 6))
    markers = {"4-choice": "o", "10-choice": "s"}
    for _, row in cross.iterrows():
        ax5a.scatter(
            row["mean_validity"],
            row["mean_accuracy"],
            color=DATASET_PALETTE.get(str(row["dataset"]), "#666666"),
            marker=markers[str(row["choice_group"])],
            s=100,
            zorder=3,
        )
        ax5a.annotate(CONFIG_SHORT[str(row["config"])], (row["mean_validity"], row["mean_accuracy"]), textcoords="offset points", xytext=(5, 3), fontsize=7)
    fit = _safe_regression(cross["mean_validity"], cross["mean_accuracy"])
    if fit is not None:
        x_fit = np.linspace(float(cross["mean_validity"].min()), float(cross["mean_validity"].max()), 100)
        ax5a.plot(x_fit, fit[0] * x_fit + fit[1], "k--", linewidth=1, label=f"r={r_val:.2f}, p={p_val:.3f}")
    for dataset in DATASET_ORDER:
        ax5a.scatter([], [], color=DATASET_PALETTE[dataset], label=DATASET_LABELS[dataset], s=80)
    for choice_group, marker in markers.items():
        ax5a.scatter([], [], color="gray", marker=marker, label=choice_group, s=80)
    ax5a.legend(fontsize=8, loc="best")
    ax5a.set_xlabel("Mean writing quality (fraction of rules passed)")
    ax5a.set_ylabel("Mean accuracy (full_question)")
    ax5a.set_title("Fig 5a: Writing Quality vs Accuracy")
    ax5a.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    _save(fig5a, output_dir, "fig5a_validity_vs_accuracy.png")

    baseline_validity = cross[cross["config"] == "human_from_scratch"].set_index("dataset")
    delta_rows: list[dict[str, object]] = []
    for _, row in cross[cross["config"] != "human_from_scratch"].iterrows():
        dataset = str(row["dataset"])
        if dataset not in baseline_validity.index:
            continue
        delta_rows.append(
            {
                "dataset": dataset,
                "config": str(row["config"]),
                "label": f"{DATASET_LABELS[dataset]}\n{CONFIG_SHORT[str(row['config'])]}",
                "validity_drop": float(row["mean_validity"] - baseline_validity.loc[dataset, "mean_validity"]),
                "accuracy_drop": float(row["mean_accuracy"] - baseline_validity.loc[dataset, "mean_accuracy"]),
            }
        )
    delta_df = pd.DataFrame(delta_rows)
    fig5b, ax5b = plt.subplots(figsize=(12, 6))
    x = np.arange(len(delta_df))
    width = 0.35
    ax5b.bar(x - width / 2, delta_df["validity_drop"], width, label="Delta validity", color="#4CAF50", alpha=0.8)
    ax5b.bar(x + width / 2, delta_df["accuracy_drop"], width, label="Delta accuracy", color="#F44336", alpha=0.8)
    ax5b.axhline(0, color="black", linewidth=0.7)
    ax5b.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax5b.set_ylabel("Delta from human_from_scratch")
    ax5b.set_xticks(x)
    ax5b.set_xticklabels(delta_df["label"], fontsize=7, rotation=30, ha="right")
    ax5b.legend(fontsize=9, loc="lower left")
    ax5b.set_title("Fig 5b: Validity Drop vs Accuracy Drop")
    _save(fig5b, output_dir, "fig5b_validity_vs_accuracy_delta.png")

    fig5c, ax5c = plt.subplots(figsize=(8, 6))
    for _, row in cross.iterrows():
        ax5c.scatter(
            row["mean_p2"],
            row["mean_accuracy"],
            color=DATASET_PALETTE.get(str(row["dataset"]), "#666666"),
            marker=markers[str(row["choice_group"])],
            s=100,
            zorder=3,
        )
        ax5c.annotate(CONFIG_SHORT[str(row["config"])], (row["mean_p2"], row["mean_accuracy"]), textcoords="offset points", xytext=(5, 3), fontsize=7)
    fit = _safe_regression(cross["mean_p2"], cross["mean_accuracy"])
    if fit is not None:
        x_fit = np.linspace(float(cross["mean_p2"].min()), float(cross["mean_p2"].max()), 100)
        ax5c.plot(x_fit, fit[0] * x_fit + fit[1], "k--", linewidth=1, label=f"r={r_p2:.2f}, p={p_p2:.3f}")
    for dataset in DATASET_ORDER:
        ax5c.scatter([], [], color=DATASET_PALETTE[dataset], label=DATASET_LABELS[dataset], s=80)
    for choice_group, marker in markers.items():
        ax5c.scatter([], [], color="gray", marker=marker, label=choice_group, s=80)
    ax5c.legend(fontsize=8)
    ax5c.set_xlabel("P(>=2 writing flaws)")
    ax5c.set_ylabel("Mean accuracy (full_question)")
    ax5c.set_title("Fig 5c: P(>=2 Writing Flaws) vs Accuracy")
    ax5c.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax5c.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    _save(fig5c, output_dir, "fig5c_p2flaws_vs_accuracy.png")

    print("\n" + "=" * 70)
    print("SECTION 6: WITHIN-GROUP CORRELATION")
    print("=" * 70)
    cross_4 = cross[cross["choice_group"] == "4-choice"]
    cross_10 = cross[cross["choice_group"] == "10-choice"]
    r4, p4 = _safe_pearsonr(cross_4["mean_validity"], cross_4["mean_accuracy"])
    r10, p10 = _safe_pearsonr(cross_10["mean_validity"], cross_10["mean_accuracy"])
    print(f"  4-choice: r={r4:.3f}, p={p4:.4f}")
    print(f"  10-choice: r={r10:.3f}, p={p10:.4f}")
    fig6a, ax6a = plt.subplots(figsize=(8, 6))
    for _, row in cross.iterrows():
        ax6a.scatter(
            row["mean_validity"],
            row["mean_accuracy"],
            color=DATASET_PALETTE.get(str(row["dataset"]), "#666666"),
            marker=markers[str(row["choice_group"])],
            s=110,
            zorder=3,
        )
        ax6a.annotate(CONFIG_SHORT[str(row["config"])], (row["mean_validity"], row["mean_accuracy"]), textcoords="offset points", xytext=(5, 3), fontsize=7)
    for choice_group, subset, style, r_value, p_value in [
        ("4-choice", cross_4, "--", r4, p4),
        ("10-choice", cross_10, ":", r10, p10),
    ]:
        fit = _safe_regression(subset["mean_validity"], subset["mean_accuracy"])
        if fit is None:
            continue
        x_fit = np.linspace(float(subset["mean_validity"].min()), float(subset["mean_validity"].max()), 100)
        ax6a.plot(x_fit, fit[0] * x_fit + fit[1], style, linewidth=1.5, label=f"{choice_group}: r={r_value:.2f}, p={p_value:.3f}")
    for dataset in DATASET_ORDER:
        ax6a.scatter([], [], color=DATASET_PALETTE[dataset], label=DATASET_LABELS[dataset], s=80)
    for choice_group, marker in markers.items():
        ax6a.scatter([], [], color="gray", marker=marker, label=choice_group, s=80)
    ax6a.legend(fontsize=7, loc="best")
    ax6a.set_xlabel("Mean writing quality")
    ax6a.set_ylabel("Mean accuracy (full_question)")
    ax6a.set_title("Fig 6a: Within-Group Correlation")
    ax6a.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    _save(fig6a, output_dir, "fig6a_within_group_correlation.png")

    print("\n" + "=" * 70)
    print("SECTION 7: choices_only vs full_question")
    print("=" * 70)
    mode_acc = (
        acc_df.groupby(["dataset", "config", "mode"], observed=True)
        .agg(mean_acc=("accuracy", "mean"))
        .reset_index()
    )
    mode_pivot = mode_acc.pivot_table(index=["dataset", "config"], columns="mode", values="mean_acc").reset_index()
    if "choices_only" not in mode_pivot.columns:
        mode_pivot["choices_only"] = np.nan
    if "full_question" not in mode_pivot.columns:
        mode_pivot["full_question"] = np.nan
    mode_pivot["delta"] = mode_pivot["choices_only"] - mode_pivot["full_question"]
    mode_pivot["label"] = [
        f"{DATASET_LABELS[str(row['dataset'])]}\n{CONFIG_SHORT[str(row['config'])]}"
        for _, row in mode_pivot.iterrows()
    ]
    fig7a, ax7a = plt.subplots(figsize=(14, 5))
    x = np.arange(len(mode_pivot))
    width = 0.35
    ax7a.bar(x - width / 2, mode_pivot["full_question"], width, label="full_question", color="#1976D2", alpha=0.85)
    ax7a.bar(x + width / 2, mode_pivot["choices_only"], width, label="choices_only", color="#F57C00", alpha=0.85)
    ax7a.set_xticks(x)
    ax7a.set_xticklabels(mode_pivot["label"], fontsize=6.5, rotation=30, ha="right")
    ax7a.set_ylabel("Mean accuracy")
    ax7a.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax7a.legend(fontsize=9)
    ax7a.set_title("Fig 7a: choices_only vs full_question Accuracy")
    _save(fig7a, output_dir, "fig7a_mode_accuracy_comparison.png")

    fig7b, ax7b = plt.subplots(figsize=(14, 5))
    ax7b.bar(x, mode_pivot["delta"], color=["#4CAF50" if value > 0 else "#F44336" for value in mode_pivot["delta"]], alpha=0.85)
    ax7b.axhline(0, color="black", linewidth=0.8)
    ax7b.set_xticks(x)
    ax7b.set_xticklabels(mode_pivot["label"], fontsize=6.5, rotation=30, ha="right")
    ax7b.set_ylabel("Delta accuracy (choices_only - full_question)")
    ax7b.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax7b.set_title("Fig 7b: Mode Delta")
    _save(fig7b, output_dir, "fig7b_mode_delta.png")

    print("\n" + "=" * 70)
    print("SECTION 8: PER-RULE x ACCURACY CORRELATION")
    print("=" * 70)
    rule_acc_rows: list[dict[str, object]] = []
    for rule in RULES_ORDER:
        rule_data = rule_fail[rule_fail["rule"] == rule].merge(
            cross[["dataset", "config", "mean_accuracy"]],
            on=["dataset", "config"],
            how="inner",
        )
        r_rule, p_rule = _safe_pearsonr(rule_data["fail_rate"], rule_data["mean_accuracy"])
        rule_acc_rows.append({"rule": rule, "r": r_rule, "p": p_rule})
    rule_acc_df = pd.DataFrame(rule_acc_rows).sort_values("r").reset_index(drop=True)
    fig8a, ax8a = plt.subplots(figsize=(10, 7))
    ax8a.barh(
        rule_acc_df["rule"],
        rule_acc_df["r"],
        color=["#F44336" if value < 0 else "#2196F3" for value in rule_acc_df["r"]],
        alpha=0.85,
    )
    ax8a.axvline(0, color="black", linewidth=0.8)
    ax8a.set_xlabel("Pearson r (rule fail rate vs mean accuracy)")
    ax8a.set_title("Fig 8a: Per-Rule Correlation with Accuracy")
    ax8a.tick_params(axis="y", labelsize=8)
    _save(fig8a, output_dir, "fig8a_per_rule_accuracy_corr.png")

    print("\n" + "=" * 70)
    print("SECTION 9: PER-QUESTION WRITING FLAW x ACCURACY")
    print("=" * 70)
    per_question_df = build_per_question_dataframe(
        eval_samples=eval_samples,
        augmented_rows=augmented_rows,
        flaw_df=flaw_df,
    )
    if per_question_df.empty:
        print("  No full_question rows available; skipping per-question analysis.")
    else:
        pq_r, pq_p = _safe_pearsonr(per_question_df["flaw_value"], per_question_df["is_correct"].astype(float))
        print(f"  Per-question Pearson r(flaw_value, is_correct): r={pq_r:.4f}, p={pq_p:.4f}")
        gap_rows: list[dict[str, object]] = []
        for rule in RULES_ORDER:
            column = f"rule_{rule}"
            passed = per_question_df[per_question_df[column] == True]["is_correct"].mean()
            failed = per_question_df[per_question_df[column] == False]["is_correct"].mean()
            gap_rows.append({"rule": rule, "acc_passed": passed, "acc_failed": failed, "gap": failed - passed})
        gap_df = pd.DataFrame(gap_rows).sort_values("gap").reset_index(drop=True)
        fig9a, ax9a = plt.subplots(figsize=(10, 7))
        ax9a.barh(
            gap_df["rule"],
            gap_df["gap"],
            color=["#F44336" if value < 0 else "#2196F3" for value in gap_df["gap"]],
            alpha=0.85,
        )
        ax9a.axvline(0, color="black", linewidth=0.8)
        ax9a.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax9a.set_xlabel("Accuracy gap (rule_failed - rule_passed)")
        ax9a.set_title("Fig 9a: Per-Question Accuracy Gap by Writing Rule")
        ax9a.tick_params(axis="y", labelsize=8)
        _save(fig9a, output_dir, "fig9a_per_question_rule_accuracy_gap.png")

        print("\n" + "=" * 70)
        print("SECTION 10: NONFUNCTIONAL DISTRACTOR RATE")
        print("=" * 70)
        distractor_rows: list[dict[str, object]] = []
        grouped = per_question_df.groupby(["dataset", "config", "sample_id"], observed=True)
        for (dataset, config, sample_id), group in grouped:
            first = group.iloc[0]
            gold_letter = str(first["gold_answer_letter"])
            options = list(first["options_randomized"])
            chosen = {str(value).strip().upper() for value in group["prediction"].tolist() if str(value).strip().upper() in CHOICE_LABELS[: len(options)]}
            wrong_letters = [CHOICE_LABELS[i] for i in range(len(options)) if CHOICE_LABELS[i] != gold_letter]
            distractor_rows.append(
                {
                    "dataset": dataset,
                    "config": config,
                    "sample_id": sample_id,
                    "total_distractors": len(wrong_letters),
                    "nonfunctional_count": sum(1 for letter in wrong_letters if letter not in chosen),
                }
            )
        nf_df = (
            pd.DataFrame(distractor_rows)
            .groupby(["dataset", "config"], observed=True)
            .agg(total_distractors=("total_distractors", "sum"), nonfunctional_count=("nonfunctional_count", "sum"))
            .reset_index()
        )
        nf_df["nonfunctional_rate"] = nf_df["nonfunctional_count"] / nf_df["total_distractors"]
        nf_df["config"] = pd.Categorical(nf_df["config"], categories=CONFIG_ORDER, ordered=True)
        fig10a, ax10a = plt.subplots(figsize=(9, 5))
        n_configs = len(CONFIG_ORDER)
        x = np.arange(len(DATASET_ORDER))
        width = 0.14
        offsets = np.linspace(-(n_configs - 1) / 2, (n_configs - 1) / 2, n_configs) * width
        for i, config in enumerate(CONFIG_ORDER):
            vals: list[float] = []
            for dataset in DATASET_ORDER:
                row = nf_df[(nf_df["dataset"] == dataset) & (nf_df["config"] == config)]
                vals.append(float(row["nonfunctional_rate"].iloc[0]) if not row.empty else 0.0)
            ax10a.bar(
                x + offsets[i],
                vals,
                width * 0.9,
                label=CONFIG_LABELS[config].replace("\n", " "),
                color=PALETTE[config],
                alpha=0.85,
            )
        ax10a.set_xticks(x)
        ax10a.set_xticklabels([DATASET_LABELS[dataset] for dataset in DATASET_ORDER])
        ax10a.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax10a.set_ylabel("Nonfunctional distractor rate")
        ax10a.legend(fontsize=7, ncol=2)
        ax10a.set_title("Fig 10a: Nonfunctional Distractor Rate")
        _save(fig10a, output_dir, "fig10a_nonfunctional_distractor_rate.png")

    print("\n" + "=" * 70)
    print("SECTION 11: NORMALISED ACCURACY")
    print("=" * 70)
    fq_all = acc_df[acc_df["mode"] == "full_question"].copy()
    fq_all["k"] = fq_all["setting"].map(K_MAP)
    fq_all["norm_acc"] = (fq_all["accuracy"] - 1.0 / fq_all["k"]) / (1.0 - 1.0 / fq_all["k"])
    norm_agg = fq_all.groupby(["dataset", "config"], observed=True)[["accuracy", "norm_acc"]].mean().reset_index()
    print(f"{'Dataset':<14} {'Config':<22} {'Raw acc':>9} {'Norm acc':>10} {'k':>4}")
    print("-" * 64)
    for dataset in DATASET_ORDER:
        for config in CONFIG_ORDER:
            row = norm_agg[(norm_agg["dataset"] == dataset) & (norm_agg["config"] == config)]
            if row.empty:
                continue
            print(
                f"{DATASET_LABELS[dataset]:<14} {config:<22} "
                f"{float(row['accuracy'].iloc[0]):>8.1%}  {float(row['norm_acc'].iloc[0]):>9.1%}  {K_MAP[config]:>4}"
            )
        print()

    fig11a, axes11a = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
    x = np.arange(len(DATASET_ORDER))
    n_configs = len(CONFIG_ORDER)
    width = 0.15
    offsets = np.linspace(-(n_configs - 1) / 2, (n_configs - 1) / 2, n_configs) * width
    for ax, column, title in zip(
        axes11a,
        ["accuracy", "norm_acc"],
        ["Raw accuracy", "Normalised accuracy"],
    ):
        for i, config in enumerate(CONFIG_ORDER):
            vals: list[float] = []
            for dataset in DATASET_ORDER:
                row = norm_agg[(norm_agg["dataset"] == dataset) & (norm_agg["config"] == config)]
                vals.append(float(row[column].iloc[0]) if not row.empty else 0.0)
            ax.bar(
                x + offsets[i],
                vals,
                width * 0.9,
                label=CONFIG_LABELS[config].replace("\n", " "),
                color=PALETTE[config],
                alpha=0.85,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[dataset] for dataset in DATASET_ORDER])
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
    _save(fig11a, output_dir, "fig11a_normalised_accuracy.png")

    baseline_norm = norm_agg[norm_agg["config"] == "human_from_scratch"].set_index("dataset")
    delta_norm_rows: list[dict[str, object]] = []
    for _, row in norm_agg[norm_agg["config"] != "human_from_scratch"].iterrows():
        dataset = str(row["dataset"])
        if dataset not in baseline_norm.index:
            continue
        delta_norm_rows.append(
            {
                "dataset": dataset,
                "config": str(row["config"]),
                "label": f"{DATASET_LABELS[dataset]}\n{CONFIG_SHORT[str(row['config'])]}",
                "raw_drop": float(row["accuracy"] - baseline_norm.loc[dataset, "accuracy"]),
                "norm_drop": float(row["norm_acc"] - baseline_norm.loc[dataset, "norm_acc"]),
            }
        )
    delta_norm_df = pd.DataFrame(delta_norm_rows)
    fig11b, ax11b = plt.subplots(figsize=(13, 5))
    x = np.arange(len(delta_norm_df))
    width = 0.35
    ax11b.bar(x - width / 2, delta_norm_df["raw_drop"], width, label="Delta raw accuracy", color="#1976D2", alpha=0.85)
    ax11b.bar(x + width / 2, delta_norm_df["norm_drop"], width, label="Delta normalised accuracy", color="#FF9800", alpha=0.85)
    ax11b.axhline(0, color="black", linewidth=0.7)
    ax11b.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax11b.set_ylabel("Delta from human_from_scratch")
    ax11b.set_xticks(x)
    ax11b.set_xticklabels(delta_norm_df["label"], fontsize=7, rotation=30, ha="right")
    ax11b.legend(fontsize=9)
    ax11b.set_title("Fig 11b: Raw vs Normalised Accuracy Drop")
    _save(fig11b, output_dir, "fig11b_normalised_accuracy_delta.png")

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    if not summary_df.empty:
        best_arc = summary_df[(summary_df["dataset"] == "arc_challenge") & (summary_df["config"] == "human_from_scratch")]
        worst_arc = summary_df[(summary_df["dataset"] == "arc_challenge") & (summary_df["config"] == "augment_human")]
        if not best_arc.empty and not worst_arc.empty:
            print(
                "  ARC writing-quality drop from human_from_scratch to augment_human: "
                f"{float(best_arc['writing_flaws_mean'].iloc[0] - worst_arc['writing_flaws_mean'].iloc[0]):.3f}"
            )
    print(f"  Figures saved to: {output_dir.resolve()}")
    print(f"  Total figures: {len(list(output_dir.glob('*.png')))}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmarker writing-flaw analysis for Inspect-native Final5 runs")
    parser.add_argument("--writing-flaw-jsonl", required=True)
    parser.add_argument("--results-root", default=str(DEFAULT_EVALUATION_LOG_ROOT))
    parser.add_argument("--augmented-dataset", default=None)
    parser.add_argument("--cache-root", default=str(DEFAULT_AUGMENTED_CACHE_ROOT))
    parser.add_argument("--output-dir", default="analysis/figures/benchmarker")
    parser.add_argument("--generator-model", default=None)
    parser.add_argument("--generator-run-name", default=None)
    parser.add_argument("--eval-models", default=None, help="Comma-separated list of eval models to include")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_analysis(args)


if __name__ == "__main__":
    raise SystemExit(main())
