"""BenchMarker writing-flaw analysis for augmented MCQA runs.

Example:
  uv run python analysis/benchmarker_analysis.py \
    --writing-flaw-jsonl /path/to/writing_flaw_rows.jsonl \
    --generator-model gpt-5.2-2025-12-11 \
    --irt-table-dir results/augmented_mcqa_irt/tables
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.constants import (
    SETTING_NAMES,
    SETTING_SPECS,
)
from utils.modeling import resolve_model_name


CONFIG_ORDER = list(SETTING_NAMES)
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
CONFIG_LABEL_PREFIXES = {
    "human_from_scratch": "Human",
    "model_from_scratch": "Model",
    "augment_human": "Aug-Human",
    "augment_model": "Aug-Model",
    "augment_ablation": "Aug-Ablation",
}
CONFIG_SHORT_PREFIXES = {
    "human_from_scratch": "Hum",
    "model_from_scratch": "Mdl",
    "augment_human": "AugH",
    "augment_model": "AugM",
    "augment_ablation": "AugA",
}
K_MAP = {setting: int(SETTING_SPECS[setting]["num_choices"]) for setting in CONFIG_ORDER}
CONFIG_LABELS = {
    setting: f"{CONFIG_LABEL_PREFIXES[setting]}\n({K_MAP[setting]}-choice)" for setting in CONFIG_ORDER
}
CONFIG_SHORT = {
    setting: f"{CONFIG_SHORT_PREFIXES[setting]}-{K_MAP[setting]}" for setting in CONFIG_ORDER
}
CHOICE_GROUP_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">")


def _choice_count_label(choice_count: int) -> str:
    return f"{int(choice_count)}-choice"


def _choice_group_markers(choice_counts: Iterable[int]) -> dict[int, str]:
    unique_counts = sorted({int(count) for count in choice_counts})
    return {
        count: CHOICE_GROUP_MARKERS[index % len(CHOICE_GROUP_MARKERS)]
        for index, count in enumerate(unique_counts)
    }


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


def load_writing_flaw_data(path: Path, generator_model: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for obj in _iter_jsonl(path):
        model = str(obj.get("model", ""))
        if generator_model and model and model != "None" and not _model_matches(generator_model, model):
            continue
        writing_flaw = dict(obj.get("writing_flaw", {}) or {})
        answers = _parse_rule_answers(writing_flaw.get("answer"))
        outcomes = [(item.strip().lower() == "pass") for item in answers[: len(RULES_ORDER)]]
        rows.append(
            {
                "dataset": str(obj.get("dataset", "")),
                "config": str(obj.get("config", "")),
                "generator_model": model,
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
        id_vars=["dataset", "config", "generator_model", "question", "flaw_value"],
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
    try:
        normalized = resolve_model_name(filter_value)
    except ValueError:
        normalized = str(filter_value)
    actual = str(actual).strip()
    return actual == normalized or actual == filter_value or actual.endswith(f"/{filter_value}") or (actual in filter_value) or (actual in normalized)


def _config_fail_rate(data: pd.DataFrame, config: str) -> float:
    row = data[data["config"] == config]
    return float(row["fail_rate"].iloc[0]) if not row.empty else 0.0


def _filter_generator_rows(frame: pd.DataFrame, generator_model: str | None) -> pd.DataFrame:
    if not generator_model or "generator" not in frame.columns:
        return frame.copy()
    generator = frame["generator"].fillna("").astype(str)
    keep = generator.eq("") | generator.map(lambda actual: _model_matches(generator_model, actual))
    return frame[keep].copy()


def load_irt_quality_data(table_dir: Path, generator_model: str | None = None) -> pd.DataFrame:
    grouped_path = table_dir / "final_grouped_question_quality.csv"
    ablation_path = table_dir / "final_ablation_question_quality.csv"
    if not grouped_path.exists():
        raise FileNotFoundError(f"Missing IRT table: {grouped_path}")
    if not ablation_path.exists():
        raise FileNotFoundError(f"Missing IRT table: {ablation_path}")

    grouped = _filter_generator_rows(pd.read_csv(grouped_path), generator_model)
    ablation = _filter_generator_rows(pd.read_csv(ablation_path), generator_model)
    columns = [
        "dataset",
        "setting",
        "generator",
        "difficulty",
        "difficulty_se",
        "discrimination",
        "discrimination_se",
    ]
    quality = pd.concat([grouped[columns], ablation[columns]], ignore_index=True)
    quality = (
        quality.groupby(["dataset", "setting"], dropna=False)
        .agg(
            difficulty=("difficulty", "mean"),
            difficulty_se=("difficulty_se", "mean"),
            discrimination=("discrimination", "mean"),
            discrimination_se=("discrimination_se", "mean"),
        )
        .reset_index()
        .rename(columns={"setting": "config"})
    )
    quality["config"] = pd.Categorical(quality["config"], categories=CONFIG_ORDER, ordered=True)
    return quality.sort_values(["dataset", "config"]).reset_index(drop=True)


def _plot_metric_scatter(
    data: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_dir: Path,
    name: str,
    percent_x: bool = False,
) -> None:
    r_value, p_value = _safe_pearsonr(data[x_col], data[y_col])
    print(f"  Pearson r({x_col}, {y_col}): r={r_value:.3f}, p={p_value:.4f}")
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = _choice_group_markers(data["choice_count"])
    for _, row in data.iterrows():
        ax.scatter(
            row[x_col],
            row[y_col],
            color=DATASET_PALETTE.get(str(row["dataset"]), "#666666"),
            marker=markers[int(row["choice_count"])],
            s=100,
            zorder=3,
        )
        ax.annotate(
            CONFIG_SHORT[str(row["config"])],
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=7,
        )
    fit = _safe_regression(data[x_col], data[y_col])
    if fit is not None:
        x_fit = np.linspace(float(data[x_col].min()), float(data[x_col].max()), 100)
        ax.plot(x_fit, fit[0] * x_fit + fit[1], "k--", linewidth=1, label=f"r={r_value:.2f}, p={p_value:.3f}")
    for dataset in DATASET_ORDER:
        ax.scatter([], [], color=DATASET_PALETTE[dataset], label=DATASET_LABELS[dataset], s=80)
    for choice_count, marker in markers.items():
        ax.scatter([], [], color="gray", marker=marker, label=_choice_count_label(choice_count), s=80)
    ax.legend(fontsize=8, loc="best")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if percent_x:
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    _save(fig, output_dir, name)


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

    print("Loading writing flaw data ...")
    flaw_df, flaw_long_df = load_writing_flaw_data(Path(args.writing_flaw_jsonl), generator_model=args.generator_model)
    print(f"  Loaded {len(flaw_df):,} rows.")

    print("Loading IRT quality data ...")
    irt_df = load_irt_quality_data(Path(args.irt_table_dir), generator_model=args.generator_model)
    print(f"  Loaded {len(irt_df):,} dataset/config IRT rows.")

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
            hum = _config_fail_rate(sub, "human_from_scratch")
            mdl = _config_fail_rate(sub, "model_from_scratch")
            aug = _config_fail_rate(sub, "augment_human")
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
    print("SECTION 4: IRT QUALITY METRICS")
    print("=" * 70)
    print(
        irt_df[["dataset", "config", "difficulty", "difficulty_se", "discrimination", "discrimination_se"]]
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    fig4a, ax4a = plt.subplots(figsize=(8, 5))
    _grouped_bar_quality(
        ax4a,
        irt_df,
        "difficulty",
        ylabel="IRT difficulty (higher = harder)",
        title="Fig 4a: IRT Difficulty",
        ci_col="difficulty_se",
    )
    _save(fig4a, output_dir, "fig4a_irt_difficulty.png")

    fig4b, ax4b = plt.subplots(figsize=(8, 5))
    _grouped_bar_quality(
        ax4b,
        irt_df,
        "discrimination",
        ylabel="IRT discrimination (higher = separates better)",
        title="Fig 4b: IRT Discrimination",
        ci_col="discrimination_se",
    )
    _save(fig4b, output_dir, "fig4b_irt_discrimination.png")

    print("\n" + "=" * 70)
    print("SECTION 5: VALIDITY vs IRT")
    print("=" * 70)
    validity_df = (
        flaw_df.groupby(["dataset", "config"], observed=True)
        .agg(mean_validity=("flaw_value", "mean"), mean_p2=("has_ge2_flaws", "mean"))
        .reset_index()
    )
    cross = validity_df.merge(irt_df, on=["dataset", "config"], how="inner")
    cross["choice_count"] = cross["config"].map(lambda config: K_MAP[str(config)])
    cross["choice_group"] = cross["choice_count"].map(_choice_count_label)

    _plot_metric_scatter(
        cross,
        x_col="mean_validity",
        y_col="difficulty",
        xlabel="Mean writing quality (fraction of rules passed)",
        ylabel="IRT difficulty (higher = harder)",
        title="Fig 5a: Writing Quality vs IRT Difficulty",
        output_dir=output_dir,
        name="fig5a_validity_vs_irt_difficulty.png",
    )
    _plot_metric_scatter(
        cross,
        x_col="mean_validity",
        y_col="discrimination",
        xlabel="Mean writing quality (fraction of rules passed)",
        ylabel="IRT discrimination (higher = separates better)",
        title="Fig 5b: Writing Quality vs IRT Discrimination",
        output_dir=output_dir,
        name="fig5b_validity_vs_irt_discrimination.png",
    )
    _plot_metric_scatter(
        cross,
        x_col="mean_p2",
        y_col="difficulty",
        xlabel="P(>=2 writing flaws)",
        ylabel="IRT difficulty (higher = harder)",
        title="Fig 5c: P(>=2 Flaws) vs IRT Difficulty",
        output_dir=output_dir,
        name="fig5c_p2flaws_vs_irt_difficulty.png",
        percent_x=True,
    )
    _plot_metric_scatter(
        cross,
        x_col="mean_p2",
        y_col="discrimination",
        xlabel="P(>=2 writing flaws)",
        ylabel="IRT discrimination (higher = separates better)",
        title="Fig 5d: P(>=2 Flaws) vs IRT Discrimination",
        output_dir=output_dir,
        name="fig5d_p2flaws_vs_irt_discrimination.png",
        percent_x=True,
    )

    print("\n" + "=" * 70)
    print("SECTION 6: DELTA FROM HUMAN BASELINE")
    print("=" * 70)
    baseline = cross[cross["config"] == "human_from_scratch"].set_index("dataset")
    delta_rows: list[dict[str, object]] = []
    for _, row in cross[cross["config"] != "human_from_scratch"].iterrows():
        dataset = str(row["dataset"])
        if dataset not in baseline.index:
            continue
        delta_rows.append(
            {
                "dataset": dataset,
                "config": str(row["config"]),
                "label": f"{DATASET_LABELS[dataset]}\n{CONFIG_SHORT[str(row['config'])]}",
                "delta_validity": float(row["mean_validity"] - baseline.loc[dataset, "mean_validity"]),
                "delta_difficulty": float(row["difficulty"] - baseline.loc[dataset, "difficulty"]),
                "delta_discrimination": float(row["discrimination"] - baseline.loc[dataset, "discrimination"]),
            }
        )
    delta_df = pd.DataFrame(delta_rows)
    fig6a, axes6a = plt.subplots(1, 2, figsize=(13, 5))
    for ax, y_col, ylabel, title in [
        (axes6a[0], "delta_difficulty", "Delta IRT difficulty", "Difficulty"),
        (axes6a[1], "delta_discrimination", "Delta IRT discrimination", "Discrimination"),
    ]:
        for _, row in delta_df.iterrows():
            ax.scatter(
                row["delta_validity"],
                row[y_col],
                color=DATASET_PALETTE.get(str(row["dataset"]), "#666666"),
                s=90,
                zorder=3,
            )
            ax.annotate(str(row["config"]), (row["delta_validity"], row[y_col]), textcoords="offset points", xytext=(4, 3), fontsize=6)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Delta writing quality from human")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    for dataset in DATASET_ORDER:
        axes6a[1].scatter([], [], color=DATASET_PALETTE[dataset], label=DATASET_LABELS[dataset], s=80)
    axes6a[1].legend(fontsize=8)
    _save(fig6a, output_dir, "fig6a_validity_irt_delta.png")

    print("\n" + "=" * 70)
    print("SECTION 8: PER-RULE x IRT CORRELATION")
    print("=" * 70)
    rule_metric_rows: list[dict[str, object]] = []
    for rule in RULES_ORDER:
        rule_data = rule_fail[rule_fail["rule"] == rule].merge(
            cross[["dataset", "config", "difficulty", "discrimination"]],
            on=["dataset", "config"],
            how="inner",
        )
        r_diff, p_diff = _safe_pearsonr(rule_data["fail_rate"], rule_data["difficulty"])
        r_disc, p_disc = _safe_pearsonr(rule_data["fail_rate"], rule_data["discrimination"])
        rule_metric_rows.append({"rule": rule, "difficulty_r": r_diff, "difficulty_p": p_diff, "discrimination_r": r_disc, "discrimination_p": p_disc})
    rule_metric_df = pd.DataFrame(rule_metric_rows).sort_values("difficulty_r").reset_index(drop=True)
    fig8a, axes8a = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    for ax, column, title in [
        (axes8a[0], "difficulty_r", "Rule Fail Rate vs IRT Difficulty"),
        (axes8a[1], "discrimination_r", "Rule Fail Rate vs IRT Discrimination"),
    ]:
        ax.barh(
            rule_metric_df["rule"],
            rule_metric_df[column],
            color=["#F44336" if value > 0 else "#2196F3" for value in rule_metric_df[column]],
            alpha=0.85,
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Pearson r")
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=8)
    _save(fig8a, output_dir, "fig8a_per_rule_irt_corr.png")

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
    parser = argparse.ArgumentParser(description="BenchMarker writing-flaw analysis using IRT quality metrics")
    parser.add_argument("--writing-flaw-jsonl", required=True)
    parser.add_argument("--irt-table-dir", default="results/augmented_mcqa_irt/tables")
    parser.add_argument("--output-dir", default="results/benchmarker_analysis")
    parser.add_argument("--generator-model", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_analysis(args)


if __name__ == "__main__":
    raise SystemExit(main())
