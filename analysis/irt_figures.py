"""Figure rendering utilities for the Augmented MCQA IRT analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from analysis.irt_model import (
    DATASET_LABELS,
    DATASET_ORDER,
    MODEL_ORDER,
    SETTING_SHORT_LABELS,
)
from utils.constants import SETTING_NAMES


LOGO_DIR = Path(__file__).parent / "assets" / "logos"
FONT_DIR = Path(__file__).parent / "assets" / "fonts"
INTER_FONT = FONT_DIR / "InterVariable.ttf"
HUMAN_COLOR = "#0072B2"
MODEL_COLOR = "#D55E00"
ABLATION_COLOR = "#CC79A7"
MISSING_COLOR = "#F2F2F2"
MODEL_HATCHES = {
    "openai/gpt-5.2-2025-12-11": "",
    "google/gemini-3.1-pro-preview": "//",
    "together/Qwen/Qwen3.5-397B-A17B": "xx",
}
MODEL_LOGOS = {
    "GPT": LOGO_DIR / "chatgpt_logo.png",
    "Gemini": LOGO_DIR / "google_gemini_icon.png",
    "Qwen": LOGO_DIR / "qwen_logo.png",
    "GPT Ablation": LOGO_DIR / "chatgpt_logo.png",
    "Gemini Ablation": LOGO_DIR / "google_gemini_icon.png",
    "Qwen Ablation": LOGO_DIR / "qwen_logo.png",
}
MODEL_LOGO_ZOOM = {
    "GPT": 0.057,
    "Gemini": 0.070,
    "Qwen": 0.090,
    "GPT Ablation": 0.057,
    "Gemini Ablation": 0.070,
    "Qwen Ablation": 0.080,
}


def use_inter_font() -> None:
    if INTER_FONT.exists():
        font_manager.fontManager.addfont(str(INTER_FONT))
        family = font_manager.FontProperties(fname=str(INTER_FONT)).get_name()
        plt.rcParams["font.family"] = family
        plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
    else:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Inter", "DejaVu Sans"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

def bar_style(row: pd.Series) -> dict[str, object]:
    if str(row.get("source")) == "Human":
        return {"color": HUMAN_COLOR, "hatch": "", "edgecolor": "black"}
    if str(row.get("source")) == "Augment Human":
        return {
            "color": HUMAN_COLOR,
            "hatch": MODEL_HATCHES.get(str(row.get("generator", "")), ""),
            "edgecolor": "black",
        }
    if str(row.get("source")) == "Augment Model":
        return {
            "color": MODEL_COLOR,
            "hatch": MODEL_HATCHES.get(str(row.get("generator", "")), ""),
            "edgecolor": "black",
        }
    if str(row.get("source")) == "Ablation":
        return {
            "color": ABLATION_COLOR,
            "hatch": MODEL_HATCHES.get(str(row.get("generator", "")), ""),
            "edgecolor": "black",
        }
    return {
        "color": MODEL_COLOR,
        "hatch": MODEL_HATCHES.get(str(row.get("generator", "")), ""),
        "edgecolor": "black",
    }


def draw_final_bar(ax: plt.Axes, x: float, value: float, err: float, row: pd.Series, *, missing: bool, width: float = 0.18) -> None:
    if missing or not np.isfinite(value):
        ax.bar(
            x,
            0.0,
            width=width,
            color=MISSING_COLOR,
            edgecolor="#666666",
            linewidth=0.7,
            hatch="//",
        )
        ax.text(x, 0.5, "to be filled in", ha="center", va="center", rotation=90, fontsize=7, transform=ax.get_xaxis_transform())
        return
    style = bar_style(row)
    ax.bar(
        x,
        value,
        yerr=err if np.isfinite(err) else None,
        width=width,
        capsize=3,
        alpha=0.9,
        linewidth=0.6,
        error_kw={"elinewidth": 1.0},
        **style,
    )


def add_logo_tick(ax: plt.Axes, x: float, path: Path, *, zoom: float = 0.055, y: float = -0.08) -> None:
    if not path.exists():
        return
    image = plt.imread(path)
    artist = AnnotationBbox(
        OffsetImage(image, zoom=zoom),
        (x, y),
        xycoords=("data", "axes fraction"),
        frameon=False,
        box_alignment=(0.5, 0.5),
        pad=0.0,
        clip_on=False,
    )
    ax.add_artist(artist)


def add_human_tick(ax: plt.Axes, x: float, y: float = -0.08) -> None:
    add_logo_tick(ax, x, LOGO_DIR / "human_logo.png", zoom=0.14, y=y)


def plot_final_grouped_quality(summary: pd.DataFrame, path: Path) -> Path:
    datasets = sorted(summary["dataset"].unique(), key=dataset_sort_key)
    metrics = [
        ("difficulty", "difficulty_se", "IRT Difficulty\n(higher = harder)"),
        ("discrimination", "discrimination_se", "IRT Discrimination\n(higher = separates better)"),
        ("mean_flaws", "mean_flaws_se", "Avg. Writing Flaws\n(lower = higher quality)"),
    ]
    fig, axes = plt.subplots(len(metrics), len(datasets), figsize=(4.6 * len(datasets), 10.2), squeeze=False)

    section_center = 0.0
    offsets = np.array([-0.27, -0.09, 0.09, 0.27])
    labels = ["Human", "GPT", "Gemini", "Qwen"]
    for col, dataset in enumerate(datasets):
        dataset_df = summary[(summary["dataset"] == dataset) & (summary["section"] == "From Scratch")].copy()
        for row_idx, (metric, err_col, ylabel) in enumerate(metrics):
            ax = axes[row_idx][col]
            plotted_values = []
            plotted_bounds = []
            group = dataset_df.set_index("label")
            for offset, label in zip(offsets, labels, strict=True):
                if label not in group.index:
                    continue
                record = group.loc[label]
                x = section_center + offset
                missing = metric == "mean_flaws" and not bool(record.get("validity_available", False))
                value = float(record[metric]) if pd.notna(record[metric]) else float("nan")
                err = float(record[err_col]) if pd.notna(record[err_col]) else 0.0
                draw_final_bar(ax, x, value, err, record, missing=missing)
                if not missing and np.isfinite(value):
                    finite_err = err if np.isfinite(err) else 0.0
                    plotted_values.append(value + finite_err)
                    plotted_bounds.append((value - finite_err, value + finite_err))

            ax.set_title(DATASET_LABELS.get(str(dataset), str(dataset)) if row_idx == 0 else "")
            ax.set_xlim(-0.43, 0.43)
            ax.set_xticks([section_center])
            ax.set_xticklabels(["From Scratch"], rotation=0)
            ax.tick_params(axis="x", pad=21, bottom=False)
            add_human_tick(ax, section_center + float(offsets[0]))
            for offset, label in zip(offsets[1:], labels[1:], strict=True):
                add_logo_tick(
                    ax,
                    section_center + float(offset),
                    MODEL_LOGOS[label],
                    zoom=MODEL_LOGO_ZOOM[label],
                )
            ax.set_ylabel(ylabel if col == 0 else "", fontsize=14)
            ax.grid(axis="y", alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if metric == "discrimination":
                ymax = max(plotted_values) * 1.08 if plotted_values else 1.2
                ax.set_ylim(0.9, max(ymax, 1.05))
            elif plotted_values:
                ymin = min(0.0, min(plotted_values) * 1.05)
                ymax = max(plotted_values) * 1.1 if max(plotted_values) > 0 else 1.0
                if dataset == "mmlu_pro" and metric == "difficulty" and plotted_bounds:
                    lows, _ = zip(*plotted_bounds, strict=True)
                    ymin = min(ymin, min(lows) * 1.1)
                ax.set_ylim(ymin, ymax)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=HUMAN_COLOR, edgecolor="black", label="Human source"),
        plt.Rectangle((0, 0), 1, 1, facecolor=MODEL_COLOR, edgecolor="black", label="Model source / GPT"),
        plt.Rectangle((0, 0), 1, 1, facecolor=MODEL_COLOR, edgecolor="black", hatch="//", label="Model source / Gemini"),
        plt.Rectangle((0, 0), 1, 1, facecolor=MODEL_COLOR, edgecolor="black", hatch="xx", label="Model source / Qwen"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncols=4, frameon=False)
    fig.subplots_adjust(top=0.96, bottom=0.115, hspace=0.34, wspace=0.20)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    # TIFF export disabled; PNG figures are sufficient for this analysis bundle.
    # fig.savefig(path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_final_ablation_quality(summary: pd.DataFrame, path: Path) -> Path:
    datasets = sorted(summary["dataset"].unique(), key=dataset_sort_key)
    metrics = [
        ("difficulty", "difficulty_se", "IRT Difficulty\n(higher = harder)"),
        ("discrimination", "discrimination_se", "IRT Discrimination\n(higher = separates better)"),
        ("mean_flaws", "mean_flaws_se", "Avg. Writing Flaws\n(lower = higher quality)"),
    ]
    fig, axes = plt.subplots(len(metrics), len(datasets), figsize=(5.8 * len(datasets), 9.8), squeeze=False)
    sections = [
        ("Augment Human", "Augment Human"),
        ("Augment Model", "Augment Model"),
        ("Ablation", "Augment Ablation"),
    ]
    section_centers = np.array([-0.72, 0.0, 0.72])
    model_offsets = np.array([-0.16, 0.0, 0.16])
    model_labels = [label for _, label in MODEL_ORDER]
    for col, dataset in enumerate(datasets):
        dataset_df = summary[summary["dataset"] == dataset].set_index("label")
        for row_idx, (metric, err_col, ylabel) in enumerate(metrics):
            ax = axes[row_idx][col]
            plotted = []
            plotted_bounds = []
            for section_center, (source, setting_label) in zip(section_centers, sections, strict=True):
                for model_offset, model_label in zip(model_offsets, model_labels, strict=True):
                    label = f"{model_label} {setting_label}"
                    record = dataset_df.loc[label]
                    x = float(section_center + model_offset)
                    value = float(record[metric]) if pd.notna(record[metric]) else float("nan")
                    err = float(record[err_col]) if pd.notna(record[err_col]) else 0.0
                    missing = not np.isfinite(value) or (metric == "mean_flaws" and not bool(record.get("validity_available", False)))
                    draw_final_bar(ax, x, value, err, record, missing=missing, width=0.12)
                    if not missing:
                        finite_err = err if np.isfinite(err) else 0.0
                        plotted.append(value + finite_err)
                        plotted_bounds.append((value - finite_err, value + finite_err))
            ax.set_title(DATASET_LABELS.get(str(dataset), str(dataset)) if row_idx == 0 else "")
            ax.set_xlim(-1.08, 1.08)
            ax.set_xticks(section_centers)
            ax.set_xticklabels(["Extension\n(Human)", "Extension\n(Model)", "Extension\n(Ablation)"], rotation=0)
            ax.tick_params(axis="x", pad=21, bottom=False)
            for section_center in section_centers:
                for model_offset, model_label in zip(model_offsets, model_labels, strict=True):
                    add_logo_tick(
                        ax,
                        float(section_center + model_offset),
                        MODEL_LOGOS[model_label],
                        zoom=MODEL_LOGO_ZOOM[model_label],
                    )
            ax.set_ylabel(ylabel if col == 0 else "", fontsize=14)
            ax.grid(axis="y", alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if metric == "discrimination":
                ymax = max(plotted) * 1.08 if plotted else 1.05
                ax.set_ylim(0.9, max(ymax, 1.05))
            elif plotted:
                ymin = min(0.0, min(plotted) * 1.05)
                if dataset == "arc_challenge" and metric == "difficulty" and ymin > -1.5:
                    ymin = -1.5
                ymax = max(plotted) * 1.1 if max(plotted) > 0 else 1.0
                if dataset == "mmlu_pro" and metric == "difficulty" and plotted_bounds:
                    lows, _ = zip(*plotted_bounds, strict=True)
                    ymin = min(ymin, min(lows) * 1.1)
                ax.set_ylim(ymin, ymax)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=HUMAN_COLOR, edgecolor="black", label="Human Guided"),
        plt.Rectangle((0, 0), 1, 1, facecolor=MODEL_COLOR, edgecolor="black", label="LLM Only"),
        plt.Rectangle((0, 0), 1, 1, facecolor=ABLATION_COLOR, edgecolor="black", label="Ablation"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", label="GPT"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch="//", label="Gemini"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch="xx", label="Qwen"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncols=6, frameon=False)
    fig.subplots_adjust(top=0.96, bottom=0.13, hspace=0.34, wspace=0.22)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    # TIFF export disabled; PNG figures are sufficient for this analysis bundle.
    # fig.savefig(path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    return path


def _pool_rows(rows: pd.DataFrame, metric: str, err_col: str) -> tuple[float, float]:
    """Pool means and SEs from multiple rows using weighted combination.

    Aggregation method: inverse-variance weighting over the per-generator SEs
    stored in the summary CSV.  When SEs are zero we fall back to simple mean
    and empirical SE across the group means so the bar is still informative.

    Returns
    -------
    pooled_mean : float
    pooled_se : float
    """
    import math as _math

    vals = rows[metric].dropna().astype(float).to_numpy()
    errs = rows[err_col].fillna(0.0).astype(float).to_numpy()
    if len(vals) == 0:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), float(errs[0])
    # If all SEs are zero, fall back to simple mean + empirical SE
    if np.all(errs == 0):
        m = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / _math.sqrt(len(vals)))
        return m, se
    # Inverse-variance weighting
    vars_ = np.where(errs > 0, errs ** 2, np.nanmin(errs[errs > 0]) ** 2)
    weights = 1.0 / vars_
    pooled_mean = float(np.sum(weights * vals) / np.sum(weights))
    pooled_var = float(1.0 / np.sum(weights))
    return pooled_mean, float(_math.sqrt(pooled_var))


def build_combined_summary(
    grouped: pd.DataFrame,
    ablation: pd.DataFrame,
) -> pd.DataFrame:
    """Build the four-bar-per-dataset summary needed by plot_combined_quality_summary.

    Reads ``final_grouped_question_quality.csv`` (4-choice data) and
    ``final_ablation_question_quality.csv`` (10-choice data) and produces one
    row per (dataset, bar_label) combination.

    Bar labels
    ----------
    * ``"4-choice Human"``   – human_from_scratch rows
    * ``"4-choice Model"``   – model_from_scratch rows (GPT + Gemini + Qwen pooled)
    * ``"10-choice Human"``  – augment_human rows (GPT + Gemini + Qwen pooled)
    * ``"10-choice Model"``  – augment_model rows (GPT + Gemini + Qwen pooled)

    Aggregation
    -----------
    Model bars are formed by inverse-variance pooling over the three generators
    (GPT, Gemini, Qwen) stored in the summary CSVs.  The method is documented
    in ``_pool_rows``.  Augment-ablation rows are intentionally excluded.
    """
    metrics = [
        ("difficulty", "difficulty_se"),
        ("discrimination", "discrimination_se"),
        ("mean_flaws", "mean_flaws_se"),
    ]
    rows = []
    for dataset in sorted(grouped["dataset"].unique(), key=dataset_sort_key):
        g = grouped[grouped["dataset"] == dataset]
        a = ablation[ablation["dataset"] == dataset]

        bar_specs: list[tuple[str, str, pd.DataFrame]] = [
            ("4-choice Human", "4H", g[g["source"] == "Human"]),
            ("4-choice Model", "4M", g[g["source"] == "Model"]),
            ("10-choice Human", "10H", a[a["source"] == "Augment Human"]),
            ("10-choice Model", "10M", a[a["source"] == "Augment Model"]),
        ]
        for bar_label, bar_key, subset in bar_specs:
            row: dict[str, object] = {
                "dataset": dataset,
                "bar_label": bar_label,
                "bar_key": bar_key,
            }
            for metric, err_col in metrics:
                m, se = _pool_rows(subset, metric, err_col)
                row[metric] = m
                row[err_col] = se
            rows.append(row)
    return pd.DataFrame(rows)


def _plot_choice_quality_panel(
    summary: pd.DataFrame,
    bar_label_human: str,
    bar_label_model: str,
    legend_human: str,
    legend_model: str,
    fig_title: str,
    path: Path,
    *,
    show_x_labels: bool = True,
) -> Path:
    """Shared helper: draw a 1-row × 3-column quality figure with 2 bars per dataset group.

    Parameters
    ----------
    summary:
        Output of ``build_combined_summary``.  Only rows matching
        ``bar_label_human`` / ``bar_label_model`` are used.
    bar_label_human / bar_label_model:
        Values of the ``bar_label`` column that correspond to the human and
        model conditions for this figure (e.g. ``"4-choice Human"``).
    legend_human / legend_model:
        Text shown in the shared legend.
    fig_title:
        Figure-level suptitle (used as row header when stacking).
    path:
        Output PNG path; a matching PDF is also written.
    show_x_labels:
        If False the x-axis tick labels are hidden (useful when stacking).
    """
    import matplotlib.ticker as mticker

    SPLIT_HUMAN_COLOR = "#E8DAB2"
    SPLIT_MODEL_COLOR = "#DD6E42"

    metrics = [
        ("difficulty",      "difficulty_se",      "IRT Difficulty"),
        ("discrimination",  "discrimination_se",  "IRT Discrimination"),
        ("mean_flaws",      "mean_flaws_se",       "Average Writing Flaws"),
    ]
    n_panels   = len(metrics)
    datasets   = sorted(summary["dataset"].unique(), key=dataset_sort_key)
    n_datasets = len(datasets)

    bar_specs = [
        (bar_label_human, SPLIT_HUMAN_COLOR, legend_human),
        (bar_label_model, SPLIT_MODEL_COLOR, legend_model),
    ]
    n_bars    = len(bar_specs)
    bar_width = 0.18
    group_offsets = np.array([-(n_bars - 1) / 2 * bar_width + i * bar_width for i in range(n_bars)])
    group_gap     = n_bars * bar_width + 0.22
    group_centers = np.arange(n_datasets) * group_gap

    # Each figure is half the combined height (3.2 / 2 = 1.6 in)
    FIG_H     = 1.6
    FONT_TITLE = 7.5
    FONT_TICK  = 6.5

    fig, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels, FIG_H), squeeze=False)

    for col, (metric, err_col, panel_title) in enumerate(metrics):
        ax = axes[0][col]
        all_tops: list[float] = []
        all_bots: list[float] = []

        for g_idx, dataset in enumerate(datasets):
            ds_df = summary[summary["dataset"] == dataset].set_index("bar_label")
            for b_idx, (bar_label, color, _) in enumerate(bar_specs):
                x = group_centers[g_idx] + group_offsets[b_idx]
                if bar_label not in ds_df.index:
                    continue
                record = ds_df.loc[bar_label]
                value  = float(record[metric])   if pd.notna(record[metric])   else float("nan")
                err    = float(record[err_col]) if pd.notna(record[err_col]) else 0.0
                if not np.isfinite(value):
                    continue
                ax.bar(
                    x, value,
                    width=bar_width,
                    color=color,
                    edgecolor="none",
                    linewidth=0,
                    yerr=err if np.isfinite(err) else None,
                    capsize=2,
                    error_kw={"elinewidth": 0.7, "ecolor": "#444444", "capthick": 0.7},
                    alpha=1.0,
                )
                fe = err if np.isfinite(err) else 0.0
                all_tops.append(value + fe)
                all_bots.append(value - fe)

        ax.set_xticks(group_centers)
        if show_x_labels:
            ax.set_xticklabels(
                [DATASET_LABELS.get(str(d), str(d)) for d in datasets],
                fontsize=FONT_TICK, rotation=0,
            )
        else:
            ax.set_xticklabels([""] * n_datasets)
            ax.tick_params(axis="x", length=0)

        ax.set_title(panel_title, fontsize=FONT_TITLE, pad=3)
        ax.tick_params(axis="y", labelsize=FONT_TICK)

        if metric == "difficulty":
            ax.axhline(0.0, color="#888888", linewidth=0.5, linestyle="--", zorder=0)

        if all_tops:
            rng = max(all_tops) - min(all_bots)
            pad = rng * 0.10
            ylo = min(all_bots) - pad
            yhi = max(all_tops) + pad
            if metric == "discrimination":
                ylo = min(ylo, 0.90)
            elif metric == "difficulty":
                ylo = min(ylo, -0.1)
            ax.set_ylim(ylo, yhi)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune="both"))

        half_gw = group_gap / 2
        ax.set_xlim(group_centers[0] - half_gw, group_centers[-1] + half_gw)

        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_color("#333333")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none", linewidth=0, label=lbl)
        for _, color, lbl in bar_specs
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncols=n_bars,
        frameon=False,
        fontsize=FONT_TICK,
        handlelength=1.2,
        handleheight=0.8,
        columnspacing=1.0,
    )

    fig.suptitle(fig_title, fontsize=FONT_TITLE + 0.5, y=1.02)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.28, top=0.88, wspace=0.34)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_4choice_quality_summary(
    grouped: pd.DataFrame,
    path: Path,
) -> Path:
    """3-panel quality figure for the 4-choice (From Scratch) condition.

    Two bars per dataset group: Human (#E8DAB2) and Model (#DD6E42).
    Model bar pools GPT, Gemini, and Qwen via inverse-variance weighting.
    Figure height is 1.6 in so that stacking with ``plot_10choice_quality_summary``
    reproduces the vertical footprint of the original combined figure.
    """
    # Build a minimal summary using only the 4-choice rows
    metrics = [
        ("difficulty",     "difficulty_se"),
        ("discrimination", "discrimination_se"),
        ("mean_flaws",     "mean_flaws_se"),
    ]
    rows = []
    for dataset in sorted(grouped["dataset"].unique(), key=dataset_sort_key):
        g = grouped[grouped["dataset"] == dataset]
        for bar_label, subset in [
            ("4-choice Human", g[g["source"] == "Human"]),
            ("4-choice Model", g[g["source"] == "Model"]),
        ]:
            row: dict[str, object] = {"dataset": dataset, "bar_label": bar_label}
            for metric, err_col in metrics:
                m, se = _pool_rows(subset, metric, err_col)
                row[metric] = m
                row[err_col] = se
            rows.append(row)
    summary = pd.DataFrame(rows)

    return _plot_choice_quality_panel(
        summary,
        bar_label_human="4-choice Human",
        bar_label_model="4-choice Model",
        legend_human="Human",
        legend_model="Model",
        fig_title="4-Choice MCQ",
        path=path,
    )


def plot_10choice_quality_summary(
    ablation: pd.DataFrame,
    path: Path,
) -> Path:
    """3-panel quality figure for the 10-choice (Augmented) condition.

    Two bars per dataset group: Human (#E8DAB2) and Model (#DD6E42).
    Augment-ablation rows are excluded.  Each model bar pools GPT, Gemini, and
    Qwen via inverse-variance weighting.
    Figure height is 1.6 in so that stacking with ``plot_4choice_quality_summary``
    reproduces the vertical footprint of the original combined figure.
    """
    metrics = [
        ("difficulty",     "difficulty_se"),
        ("discrimination", "discrimination_se"),
        ("mean_flaws",     "mean_flaws_se"),
    ]
    rows = []
    for dataset in sorted(ablation["dataset"].unique(), key=dataset_sort_key):
        a = ablation[ablation["dataset"] == dataset]
        for bar_label, subset in [
            ("10-choice Human", a[a["source"] == "Augment Human"]),
            ("10-choice Model", a[a["source"] == "Augment Model"]),
        ]:
            row: dict[str, object] = {"dataset": dataset, "bar_label": bar_label}
            for metric, err_col in metrics:
                m, se = _pool_rows(subset, metric, err_col)
                row[metric] = m
                row[err_col] = se
            rows.append(row)
    summary = pd.DataFrame(rows)

    return _plot_choice_quality_panel(
        summary,
        bar_label_human="10-choice Human",
        bar_label_model="10-choice Model",
        legend_human="Human",
        legend_model="Model",
        fig_title="10-Choice MCQ (Augmented)",
        path=path,
    )


def plot_stacked_quality_summary(
    grouped: pd.DataFrame,
    ablation: pd.DataFrame,
    path: Path,
    *,
    scale: float = 1.0,
) -> Path:
    """Single figure with 4-choice (top row) and 10-choice (bottom row) stacked.

    Parameters
    ----------
    scale:
        Scales only the figure HEIGHT.
        1.0 → full golden-ratio height (10.8 × 6.68 in).
        0.5 → same width, half height (10.8 × 3.34 in) — for two-column layouts.

    Layout
    ------
    * 2 rows × 3 columns.  Width always 10.8 in; height = (10.8 / φ) * scale.
    * Row 0 (4-choice): panel titles shown; x-axis labels hidden.
    * Row 1 (10-choice): no panel titles; x-axis labels (ARC Challenge, MMLU, GPQA) shown.
    * Left y-axis carries row labels: "4-Choice" / "10-Choice".
    * Exactly 5 y-axis tick marks per column, shared across both rows:
        - IRT Difficulty:        −2.5, −1.5, −0.5, 0.5, 1.0  (axis from −2.75 → ~1.25)
        - IRT Discrimination:   1.0, 1.1, 1.2, 1.3, 1.4       (axis from 1.0 → ~1.55)
        - Average Writing Flaws: 0, 1, 2, 3, 4                  (axis from 0 → ~4.3)
    * One shared legend (Human / Model) with light-grey box, placed just below subplots.
    * Colors: Human = #E8DAB2, Model = #DD6E42.
    * Saved as both PNG (200 dpi) and PDF.
    """
    STACKED_HUMAN_COLOR = "#E8DAB2"
    STACKED_MODEL_COLOR = "#DD6E42"
    PHI = (1 + 5 ** 0.5) / 2   # golden ratio ≈ 1.618

    # Explicit 5-tick specs per metric: (ticks, ymin, ymax)
    #   - difficulty:     min −2.75 visible (axis bottom); top tick 1.0 with headroom → axis to 1.25
    #   - discrimination: min 1.0 visible; top tick 1.4 (near 1.5) with headroom → axis to 1.55
    #   - mean_flaws:     0–4 (5 even ticks); axis to 4.3 so 4 is near but not at very top
    YTICK_SPECS: dict[str, tuple[list[float], float, float]] = {
        "difficulty":     ([-2.5, -1.5, -0.5,  0.5       ], -2.75, 0.75),
        "discrimination": ([ 1.0,  1.1,  1.2,  1.3       ],  1.00, 1.35),
        "mean_flaws":     ([ 0.0,  1.0,  2.0,  3.0       ],  0.00, 3.60),
    }

    metrics = [
        ("difficulty",     "difficulty_se",     "IRT Difficulty"),
        ("discrimination", "discrimination_se", "IRT Discrimination"),
        ("mean_flaws",     "mean_flaws_se",     "Average Writing Flaws"),
    ]
    metric_pairs = [(m, e) for m, e, _ in metrics]

    def _build(source_df: pd.DataFrame, human_source: str, model_source: str,
               human_label: str, model_label: str) -> pd.DataFrame:
        rows = []
        for dataset in sorted(source_df["dataset"].unique(), key=dataset_sort_key):
            g = source_df[source_df["dataset"] == dataset]
            for bar_label, subset in [
                (human_label, g[g["source"] == human_source]),
                (model_label, g[g["source"] == model_source]),
            ]:
                row: dict[str, object] = {"dataset": dataset, "bar_label": bar_label}
                for metric, err_col in metric_pairs:
                    m, se = _pool_rows(subset, metric, err_col)
                    row[metric] = m
                    row[err_col] = se
                rows.append(row)
        return pd.DataFrame(rows)

    summary_4  = _build(grouped,  "Human",         "Model",         "4H", "4M")
    summary_10 = _build(ablation, "Augment Human",  "Augment Model", "10H", "10M")

    datasets   = sorted(summary_4["dataset"].unique(), key=dataset_sort_key)
    n_datasets = len(datasets)
    n_panels   = len(metrics)

    bar_specs = [
        ("H", STACKED_HUMAN_COLOR, "Human"),
        ("M", STACKED_MODEL_COLOR, "Model"),
    ]
    n_bars    = 2
    bar_width = 0.18
    group_offsets = np.array([-(n_bars - 1) / 2 * bar_width + i * bar_width for i in range(n_bars)])
    inter_gap     = 0.11
    group_gap     = n_bars * bar_width + inter_gap
    group_centers = np.arange(n_datasets) * group_gap

    # Width fixed at 10.8 in; only height scales
    FIG_W = 10.8
    FIG_H = (FIG_W / PHI) * scale   # 6.68 at scale=1, 3.34 at scale=0.5

    FONT_TITLE = 8.5
    FONT_TICK  = 7.5
    FONT_ROW   = 8.0

    fig, axes = plt.subplots(2, n_panels, figsize=(FIG_W, FIG_H), squeeze=False)

    row_configs = [
        (summary_4,  "4H",  "4M",  True,  False, "4-Choice"),
        (summary_10, "10H", "10M", False,  True,  "10-Choice"),
    ]

    for row_idx, (summary, hkey, mkey, show_titles, show_xlabels, row_label) in enumerate(row_configs):
        for col, (metric, err_col, panel_title) in enumerate(metrics):
            ax = axes[row_idx][col]

            for g_idx, dataset in enumerate(datasets):
                ds_df = summary[summary["dataset"] == dataset].set_index("bar_label")
                for b_idx, (_, color, _lbl) in enumerate(bar_specs):
                    bar_key = hkey if b_idx == 0 else mkey
                    x = group_centers[g_idx] + group_offsets[b_idx]
                    if bar_key not in ds_df.index:
                        continue
                    record = ds_df.loc[bar_key]
                    value  = float(record[metric])   if pd.notna(record[metric])   else float("nan")
                    err    = float(record[err_col]) if pd.notna(record[err_col]) else 0.0
                    if not np.isfinite(value):
                        continue
                    ax.bar(
                        x, value,
                        width=bar_width,
                        color=color,
                        edgecolor="none",
                        linewidth=0,
                        yerr=err if np.isfinite(err) else None,
                        capsize=2,
                        error_kw={"elinewidth": 0.7, "ecolor": "#444444", "capthick": 0.7},
                        alpha=1.0,
                    )

            # X-axis
            ax.set_xticks(group_centers)
            if show_xlabels:
                ax.set_xticklabels(
                    [DATASET_LABELS.get(str(d), str(d)) for d in datasets],
                    fontsize=FONT_TICK, rotation=0,
                )
            else:
                ax.set_xticklabels([""] * n_datasets)
                ax.tick_params(axis="x", length=0)

            # Panel title (top row only)
            if show_titles:
                ax.set_title(panel_title, fontsize=FONT_TITLE, pad=4)

            ax.tick_params(axis="y", labelsize=FONT_TICK)

            # Row label on leftmost column
            if col == 0:
                ax.set_ylabel(row_label, fontsize=FONT_ROW, labelpad=5)

            # Zero baseline for difficulty
            if metric == "difficulty":
                ax.axhline(0.0, color="#888888", linewidth=0.6, linestyle="--", zorder=0)

            # X limits
            half_gw = group_gap / 2
            ax.set_xlim(group_centers[0] - half_gw, group_centers[-1] + half_gw)

            # Style: no grid, full rectangular border
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.6)
                spine.set_color("#333333")

    # Apply exactly 5 explicit y-ticks, shared across both rows per column
    for col, (metric, _, _) in enumerate(metrics):
        ticks, ymin, ymax = YTICK_SPECS[metric]
        for row_idx in range(2):
            ax = axes[row_idx][col]
            ax.set_ylim(ymin, ymax)
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(
                lambda v, _: f"{v:.1f}" if v != int(v) else f"{int(v)}"
            ))

    # Shared legend: placed just below the subplot area without overlapping x-axis.
    # At half height the same figure-fraction sits proportionally closer to the axis,
    # so push it further down at smaller scales.
    legend_y = -0.02 - 0.06 * (1.0 - scale)   # −0.02 at scale=1, −0.05 at scale=0.5
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none", linewidth=0, label=lbl)
        for _, color, lbl in bar_specs
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncols=n_bars,
        frameon=True,
        facecolor="#f0f0f0",
        edgecolor="#cccccc",
        framealpha=0.9,
        fontsize=FONT_TICK,
        handlelength=1.2,
        handleheight=0.9,
        columnspacing=1.0,
    )

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.10, top=0.94, hspace=0.12, wspace=0.18)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path



def plot_stacked_extended_summary(
    grouped: pd.DataFrame,
    ablation: pd.DataFrame,
    path: Path,
) -> Path:
    """2-row × 3-column per-model (un-averaged) stacked bar figure.

    Top row (4-choice):   Human | GPT | Gemini | Qwen  (4 bars / group).
    Bottom row (10-choice): AugH-GPT | AugH-Gemini | AugH-Qwen |
                             AugM-GPT | AugM-Gemini | AugM-Qwen  (6 bars / group).
    Icons are rendered inside each bar.  Two centered one-line legend boxes.
    """
    from io import BytesIO

    try:
        import cairosvg
    except ModuleNotFoundError:
        cairosvg = None
    from matplotlib.image import BboxImage
    from matplotlib.offsetbox import DrawingArea
    from matplotlib.patches import Circle, Ellipse, Rectangle
    from matplotlib.transforms import Bbox, TransformedBbox

    use_inter_font()

    PHI = (1 + 5 ** 0.5) / 2
    EXT_HUMAN_COLOR = "#E8DAB2"
    EXT_MODEL_COLOR = "#DD6E42"

    GEN_KEYS = [
        "openai/gpt-5.2-2025-12-11",
        "google/gemini-3.1-pro-preview",
        "together/Qwen/Qwen3.5-397B-A17B",
    ]
    HUMAN_MARKER = "__human_bust__"
    GEN_LOGOS = {
        "openai/gpt-5.2-2025-12-11":      LOGO_DIR / "chatgpt_logo.svg",
        "google/gemini-3.1-pro-preview":   LOGO_DIR / "google_gemini_icon.svg",
        "together/Qwen/Qwen3.5-397B-A17B": LOGO_DIR / "qwen_logo.svg",
    }
    HUMAN_SIZE = 12.0
    GEN_SIZES = {
        "openai/gpt-5.2-2025-12-11": 9.0,
        "google/gemini-3.1-pro-preview": 9.0,
        "together/Qwen/Qwen3.5-397B-A17B": 12.0,
    }

    GEN_DISPLAY = {
        "openai/gpt-5.2-2025-12-11":      "GPT 5.2",
        "google/gemini-3.1-pro-preview":   "Gemini 3.1 Pro",
        "together/Qwen/Qwen3.5-397B-A17B": "Qwen 3.5 389B A17B",
    }

    YTICK_SPECS = {
        "difficulty":     ([0.5, 1.5, 2.5, 3.5], 0.00, 4.4),
        "discrimination": ([ 1.0,  1.1,  1.2,  1.3,  1.4],  1.00, 1.425),
        "mean_flaws":     ([ 0.0,  1.0,  2.0,  3.0, 4.0],  0.00, 4.50),
    }
    metrics = [
        ("difficulty",     "difficulty_se",     "IRT Difficulty"),
        ("discrimination", "discrimination_se", "IRT Discriminability"),
        ("mean_flaws",     "mean_flaws_se",     "Average Writing Flaws"),
    ]

    datasets  = sorted(grouped["dataset"].unique(), key=dataset_sort_key)
    n_datasets = len(datasets)
    n_panels   = len(metrics)
    difficulty_values = pd.concat(
        [
            grouped[grouped["source"].isin(["Human", "Model"])]["difficulty"],
            ablation[ablation["source"].isin(["Augment Human", "Augment Model"])]["difficulty"],
        ],
        ignore_index=True,
    ).dropna()
    difficulty_offset = 0.5 - float(difficulty_values.min())

    # Bar specs: (bar_key, generator_key_or_None, color, logo_path, zoom)
    # aug_human bars: show the GENERATOR icon (not human)
    top_specs = [
        ("human", None,        EXT_HUMAN_COLOR, HUMAN_MARKER,        HUMAN_SIZE),
    ] + [
        (gen,     gen,         EXT_MODEL_COLOR, GEN_LOGOS[gen],      GEN_SIZES[gen])
        for gen in GEN_KEYS
    ]
    bot_specs = (
        [("aug_h_" + gen, gen, EXT_HUMAN_COLOR, GEN_LOGOS[gen], GEN_SIZES[gen]) for gen in GEN_KEYS] +
        [("aug_m_" + gen, gen, EXT_MODEL_COLOR, GEN_LOGOS[gen], GEN_SIZES[gen]) for gen in GEN_KEYS]
    )

    bar_width = 0.123
    intra_gap = 0.012
    inter_gap = 0.142

    def _offsets(n):
        total = n * bar_width + (n - 1) * intra_gap
        return np.arange(n) * (bar_width + intra_gap) - total / 2 + bar_width / 2

    top_off = _offsets(len(top_specs))
    bot_off = _offsets(len(bot_specs))
    top_gap = len(top_specs) * (bar_width + intra_gap) + inter_gap
    bot_gap = len(bot_specs) * (bar_width + intra_gap) + inter_gap
    top_ctr = np.arange(n_datasets) * top_gap
    bot_ctr = np.arange(n_datasets) * bot_gap

    FIG_W = 13.0
    FIG_H = (FIG_W / PHI) / 2
    FONT_TITLE = 9.5
    FONT_TICK  = 8.5
    FONT_ROW   = 9.5

    fig, axes = plt.subplots(2, n_panels, figsize=(FIG_W, FIG_H), squeeze=False)

    # Storage for icon placement (draw after bars)
    icon_jobs: list[tuple] = []   # (ax, x, anchor_y, sign, marker_or_logo, size)

    def _bar(ax, x, value, err, color):
        if not np.isfinite(value):
            return
        ax.bar(x, value, width=bar_width, color=color, edgecolor="none", linewidth=0,
               yerr=err if np.isfinite(err) else None, capsize=1.5,
               error_kw={"elinewidth": 0.6, "ecolor": "#333333", "capthick": 0.6})

    def _get(df, metric, err_col):
        if df.empty:
            return float("nan"), 0.0
        r = df.iloc[0]
        return (float(r[metric]) if pd.notna(r[metric]) else float("nan"),
                float(r[err_col]) if pd.notna(r[err_col]) else 0.0)

    def _icon_anchor(value, err):
        sign = 1.0 if value >= 0 else -1.0
        ferr = err if np.isfinite(err) else 0.0
        return value + sign * ferr, sign

    for col, (metric, err_col, title) in enumerate(metrics):
        ax_top = axes[0][col]
        ax_bot = axes[1][col]
        abl = ablation[ablation["source"] != "Ablation"]

        for g_idx, dataset in enumerate(datasets):
            g = grouped[grouped["dataset"] == dataset]
            a = abl[abl["dataset"] == dataset]

            # Top row
            for b_idx, (bk, gen, color, logo, zoom) in enumerate(top_specs):
                x = top_ctr[g_idx] + top_off[b_idx]
                if gen is None:
                    v, e = _get(g[g["source"] == "Human"], metric, err_col)
                else:
                    v, e = _get(g[(g["source"] == "Model") & (g["generator"] == gen)], metric, err_col)
                if metric == "difficulty" and np.isfinite(v):
                    v += difficulty_offset
                _bar(ax_top, x, v, e, color)
                if np.isfinite(v):
                    anchor_y, sign = _icon_anchor(v, e)
                    icon_jobs.append((ax_top, x, anchor_y, sign, logo, zoom))

            # Bottom row
            for b_idx, (bk, gen, color, logo, zoom) in enumerate(bot_specs):
                x = bot_ctr[g_idx] + bot_off[b_idx]
                src = "Augment Human" if "aug_h_" in bk else "Augment Model"
                v, e = _get(a[(a["source"] == src) & (a["generator"] == gen)], metric, err_col)
                if metric == "difficulty" and np.isfinite(v):
                    v += difficulty_offset
                _bar(ax_bot, x, v, e, color)
                if np.isfinite(v):
                    anchor_y, sign = _icon_anchor(v, e)
                    icon_jobs.append((ax_bot, x, anchor_y, sign, logo, zoom))

    # Apply y-ticks / limits / styling
    for col, (metric, err_col, title) in enumerate(metrics):
        ticks, ymin, ymax = YTICK_SPECS[metric]
        for row_idx, (ax, centers, show_x, show_title, row_lbl) in enumerate([
            (axes[0][col], top_ctr, False, True,  "MCQ Generation"   if col == 0 else None),
            (axes[1][col], bot_ctr, True,  False, "MCQ Extension" if col == 0 else None),
        ]):
            ax.set_ylim(ymin, ymax)
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(
                lambda v, _: f"{v:.1f}" if v != int(v) else f"{int(v)}"
            ))
            ax.set_xticks(centers)
            if show_x:
                ax.set_xticklabels([DATASET_LABELS.get(str(d), str(d)) for d in datasets],
                                   fontsize=FONT_TICK)
            else:
                ax.set_xticklabels([""] * n_datasets)
                ax.tick_params(axis="x", length=0)
            if show_title:
                ax.set_title(title, fontsize=FONT_TITLE, pad=4)
            ax.tick_params(axis="y", labelsize=FONT_TICK)
            if row_lbl:
                ax.set_ylabel(row_lbl, fontsize=FONT_ROW, labelpad=5)
            half = (centers[1] - centers[0]) / 2 if len(centers) > 1 else 0.5
            ax.set_xlim(centers[0] - half, centers[-1] + half)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_linewidth(0.45); spine.set_color("#666666")

    def _load_svg_logo(svg_path: Path, output_px: int = 1024):
        if cairosvg is None:
            return plt.imread(svg_path.with_suffix(".png"))
        png = cairosvg.svg2png(url=str(svg_path), output_width=output_px, output_height=output_px)
        return plt.imread(BytesIO(png), format="png")

    svg_images = {path: _load_svg_logo(path) for path in set(GEN_LOGOS.values())}

    def _logo_box(svg_path, size, *, legend=False):
        y_shift = -0.10 * size if legend else 0.0
        da = DrawingArea(size, size, 0, 0)
        bbox = Bbox.from_bounds(0, y_shift, size, size)
        image = BboxImage(TransformedBbox(bbox, da.get_transform()), data=svg_images[svg_path], interpolation="bilinear")
        da.add_artist(image)
        return da

    def _data_y_from_pixel_offset(ax, y, sign, px=8):
        _, y_disp = ax.transData.transform((0, y))
        return ax.transData.inverted().transform((0, y_disp + sign * px))[1]

    # Place every icon 8 px from the corresponding error-bar end.
    def _add_bust(ax, x, y, size):
        da = DrawingArea(size, size, 0, 0)
        da.add_artist(Circle((size * 0.5, size * 0.68), radius=size * 0.20, facecolor="black", edgecolor="black", linewidth=0))
        da.add_artist(Ellipse((size * 0.5, size * 0.30), width=size * 0.72, height=size * 0.42, facecolor="black", edgecolor="black", linewidth=0))
        ab = AnnotationBbox(da, (x, y), xycoords="data", frameon=False, box_alignment=(0.5, 0.5), pad=0.0, clip_on=True)
        ax.add_artist(ab)

    for ax, x, anchor_y, sign, marker, size in icon_jobs:
        y_mid = _data_y_from_pixel_offset(ax, anchor_y, sign, px=8)
        if marker == HUMAN_MARKER:
            _add_bust(ax, x, y_mid, size)
            continue
        ab = AnnotationBbox(_logo_box(marker, size), (x, y_mid),
                            xycoords="data", frameon=False,
                            box_alignment=(0.5, 0.5), pad=0.0, clip_on=True)
        ax.add_artist(ab)

    def _fig_square_size(points: float) -> tuple[float, float]:
        inches = points / 72.0
        return inches / FIG_W, inches / FIG_H

    def _legend_frame(x0: float, y0: float, width: float, height: float) -> None:
        fig.add_artist(
            Rectangle(
                (x0, y0),
                width,
                height,
                transform=fig.transFigure,
                facecolor="none",
                edgecolor="#cccccc",
                linewidth=0.7,
                clip_on=False,
            )
        )

    LEGEND_FONT = 9.5

    def _legend_text(
        x: float,
        y: float,
        text: str,
        *,
        weight: str = "normal",
        ha: str = "center",
    ) -> None:
        fig.text(
            x,
            y,
            text,
            ha=ha,
            va="center",
            fontsize=LEGEND_FONT,
            fontweight=weight,
            transform=fig.transFigure,
        )

    def _legend_swatch(x: float, y: float, color: str) -> None:
        sw_w, sw_h = 0.018, 0.020
        fig.add_artist(
            Rectangle(
                (x - sw_w / 2, y - sw_h / 2),
                sw_w,
                sw_h,
                transform=fig.transFigure,
                facecolor=color,
                edgecolor="none",
                clip_on=False,
            )
        )

    def _legend_logo(svg_path: Path, x: float, y: float) -> None:
        point_sizes = {
            "chatgpt_logo.svg": 10.8,
            "google_gemini_icon.svg": 9.8,
            "qwen_logo.svg": 12.0,
        }
        w, h = _fig_square_size(point_sizes.get(svg_path.name, 7.2))
        bbox = Bbox.from_bounds(x - w / 2, y - h / 2, w, h)
        fig.add_artist(
            BboxImage(
                TransformedBbox(bbox, fig.transFigure),
                data=svg_images[svg_path],
                interpolation="bilinear",
                clip_on=False,
            )
        )

    def _legend_bust(x: float, y: float) -> None:
        size = 13.0
        da = DrawingArea(size, size, 0, 0)
        da.add_artist(Circle((size * 0.5, size * 0.68), radius=size * 0.20, facecolor="black", edgecolor="black", linewidth=0))
        da.add_artist(Ellipse((size * 0.5, size * 0.30), width=size * 0.72, height=size * 0.42, facecolor="black", edgecolor="black", linewidth=0))
        fig.add_artist(
            AnnotationBbox(
                da,
                (x, y + 0.002),
                xycoords=fig.transFigure,
                frameon=False,
                box_alignment=(0.5, 0.5),
                pad=0.0,
                clip_on=False,
            )
        )

    legend_y = -0.006
    legend_h = 0.056
    left_x, left_w = 0.045, 0.320
    right_x, right_w = 0.380, 0.595
    legend_cy = legend_y + legend_h / 2
    _legend_frame(left_x, legend_y, left_w, legend_h)
    _legend_frame(right_x, legend_y, right_w, legend_h)

    _legend_text(left_x + left_w * 0.13, legend_cy, "Type")
    _legend_swatch(left_x + left_w * 0.31, legend_cy, EXT_HUMAN_COLOR)
    _legend_text(left_x + left_w * 0.35, legend_cy, "Human Guided", ha="left")
    _legend_swatch(left_x + left_w * 0.68, legend_cy, EXT_MODEL_COLOR)
    _legend_text(left_x + left_w * 0.72, legend_cy, "LLM Only", ha="left")

    right_slots = [
        (right_x + right_w * 0.04, right_x + right_w * 0.04, "Generator", None),
        (right_x + right_w * 0.17, right_x + right_w * 0.205, "Human", HUMAN_MARKER),
        (right_x + right_w * 0.32, right_x + right_w * 0.345, GEN_DISPLAY[GEN_KEYS[0]], GEN_LOGOS[GEN_KEYS[0]]),
        (right_x + right_w * 0.50, right_x + right_w * 0.530, GEN_DISPLAY[GEN_KEYS[1]], GEN_LOGOS[GEN_KEYS[1]]),
        (right_x + right_w * 0.75, right_x + right_w * 0.780, GEN_DISPLAY[GEN_KEYS[2]], GEN_LOGOS[GEN_KEYS[2]]),
    ]
    for marker_x, label_x, label, marker in right_slots:
        if marker == HUMAN_MARKER:
            _legend_bust(marker_x, legend_cy)
        elif isinstance(marker, Path):
            _legend_logo(marker, marker_x, legend_cy + 0.001)
        _legend_text(label_x, legend_cy, label, ha="left")

    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.12, top=0.94, hspace=0.12, wspace=0.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=400, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    return path



    """2-row × 3-column figure with per-model (un-averaged) bars.

    Layout
    ------
    Top row (4-choice):   Human | GPT | Gemini | Qwen  — 4 bars per dataset group.
    Bottom row (10-choice): Aug-Human-GPT | Aug-Human-Gemini | Aug-Human-Qwen |
                             Aug-Model-GPT | Aug-Model-Gemini | Aug-Model-Qwen — 6 bars.

    Figure width ≈ 12.5 in; height = (width / φ) / 2 (same vertical scale as the
    half-size stacked figure but proportionally wider).

    Icons (model logos) are placed above each bar group on a per-bar basis.
    Two legend boxes:
      - Left:  ■ Human color  ■ Model color
      - Right: [icon] Human · [icon] GPT 5.2 · [icon] Gemini 3.1 Pro · [icon] Qwen 3.5 389B A17B

    Colors: Human = #E8DAB2, Model = #DD6E42 (same as quality_stacked_half).
    Saved as both PNG (200 dpi) and PDF.
    """
    PHI = (1 + 5 ** 0.5) / 2

    EXT_HUMAN_COLOR = "#E8DAB2"
    EXT_MODEL_COLOR = "#DD6E42"

    # Generator order and display names
    GEN_ORDER = [
        ("openai/gpt-5.2-2025-12-11",       "GPT"),
        ("google/gemini-3.1-pro-preview",    "Gemini"),
        ("together/Qwen/Qwen3.5-397B-A17B",  "Qwen"),
    ]
    GEN_KEYS   = [g for g, _ in GEN_ORDER]
    GEN_LABELS = {g: lbl for g, lbl in GEN_ORDER}

    # Logo paths
    HUMAN_LOGO  = LOGO_DIR / "human_logo.png"
    GEN_LOGOS   = {
        "openai/gpt-5.2-2025-12-11":      LOGO_DIR / "chatgpt_logo.png",
        "google/gemini-3.1-pro-preview":   LOGO_DIR / "google_gemini_icon.png",
        "together/Qwen/Qwen3.5-397B-A17B": LOGO_DIR / "qwen_logo.png",
    }
    LOGO_ZOOM_MAP = {
        "openai/gpt-5.2-2025-12-11":      0.048,
        "google/gemini-3.1-pro-preview":   0.055,
        "together/Qwen/Qwen3.5-397B-A17B": 0.072,
    }

    # Display names for the icon legend
    ICON_LEGEND_LABELS = {
        "human":                            "Human",
        "openai/gpt-5.2-2025-12-11":       "GPT 5.2",
        "google/gemini-3.1-pro-preview":    "Gemini 3.1 Pro",
        "together/Qwen/Qwen3.5-397B-A17B":  "Qwen 3.5 389B A17B",
    }

    YTICK_SPECS: dict[str, tuple[list[float], float, float]] = {
        "difficulty":     ([-2.5, -1.5, -0.5,  0.5       ], -2.75, 0.75),
        "discrimination": ([ 1.0,  1.1,  1.2,  1.3       ],  1.00, 1.35),
        "mean_flaws":     ([ 0.0,  1.0,  2.0,  3.0       ],  0.00, 3.60),
    }

    metrics = [
        ("difficulty",     "difficulty_se",     "IRT Difficulty"),
        ("discrimination", "discrimination_se", "IRT Discrimination"),
        ("mean_flaws",     "mean_flaws_se",     "Average Writing Flaws"),
    ]
    metric_pairs = [(m, e) for m, e, _ in metrics]

    datasets = sorted(grouped["dataset"].unique(), key=dataset_sort_key)
    n_datasets = len(datasets)
    n_panels   = len(metrics)

    # ── Bar specs per row ─────────────────────────────────────────────────────
    # Top (4-choice): Human, then one bar per generator
    top_specs = [("human", None, EXT_HUMAN_COLOR)] + [
        (gen, gen, EXT_MODEL_COLOR) for gen in GEN_KEYS
    ]
    # Bottom (10-choice): Augment Human × 3, then Augment Model × 3
    bot_specs = (
        [("aug_human_" + gen, gen, EXT_HUMAN_COLOR) for gen in GEN_KEYS]
        + [("aug_model_" + gen, gen, EXT_MODEL_COLOR) for gen in GEN_KEYS]
    )

    n_top = len(top_specs)   # 4
    n_bot = len(bot_specs)   # 6

    bar_width = 0.13
    # Within-group gap (between bars of different types in the same group)
    intra_gap = 0.04
    # Gap between groups (ARC → MMLU etc.)
    inter_gap = 0.22

    def group_offsets(n_bars: int) -> np.ndarray:
        total_w = n_bars * bar_width + (n_bars - 1) * intra_gap
        starts  = np.arange(n_bars) * (bar_width + intra_gap)
        return starts - total_w / 2 + bar_width / 2

    top_offsets   = group_offsets(n_top)
    bot_offsets   = group_offsets(n_bot)

    group_gap_top = n_top * (bar_width + intra_gap) + inter_gap
    group_gap_bot = n_bot * (bar_width + intra_gap) + inter_gap
    top_centers   = np.arange(n_datasets) * group_gap_top
    bot_centers   = np.arange(n_datasets) * group_gap_bot

    # Figure sizing: 12.5 in wide, golden-ratio half height
    FIG_W = 12.5
    FIG_H = (FIG_W / PHI) / 2   # ≈ 3.86 in

    FONT_TITLE = 8.5
    FONT_TICK  = 7.0
    FONT_ROW   = 8.0

    fig, axes = plt.subplots(2, n_panels, figsize=(FIG_W, FIG_H), squeeze=False)

    # ── Helper: draw one bar ──────────────────────────────────────────────────
    def draw_bar(ax, x, metric, err_col, df_row, color):
        value = float(df_row[metric])   if pd.notna(df_row[metric])   else float("nan")
        err   = float(df_row[err_col]) if pd.notna(df_row[err_col]) else 0.0
        if not np.isfinite(value):
            return
        ax.bar(
            x, value,
            width=bar_width,
            color=color,
            edgecolor="none",
            linewidth=0,
            yerr=err if np.isfinite(err) else None,
            capsize=1.5,
            error_kw={"elinewidth": 0.6, "ecolor": "#444444", "capthick": 0.6},
            alpha=1.0,
        )

    # ── Draw bars ─────────────────────────────────────────────────────────────
    for col, (metric, err_col, panel_title) in enumerate(metrics):
        # TOP ROW — 4-choice
        ax_top = axes[0][col]
        for g_idx, dataset in enumerate(datasets):
            g = grouped[grouped["dataset"] == dataset]
            for b_idx, (bar_key, gen, color) in enumerate(top_specs):
                x = top_centers[g_idx] + top_offsets[b_idx]
                if gen is None:
                    subset = g[g["source"] == "Human"]
                else:
                    subset = g[(g["source"] == "Model") & (g["generator"] == gen)]
                if subset.empty:
                    continue
                row = subset.iloc[0]
                draw_bar(ax_top, x, metric, err_col, row, color)

        # BOTTOM ROW — 10-choice
        ax_bot = axes[1][col]
        abl = ablation[ablation["source"] != "Ablation"]
        for g_idx, dataset in enumerate(datasets):
            a = abl[abl["dataset"] == dataset]
            for b_idx, (bar_key, gen, color) in enumerate(bot_specs):
                x = bot_centers[g_idx] + bot_offsets[b_idx]
                src = "Augment Human" if "aug_human" in bar_key else "Augment Model"
                subset = a[(a["source"] == src) & (a["generator"] == gen)]
                if subset.empty:
                    continue
                row = subset.iloc[0]
                draw_bar(ax_bot, x, metric, err_col, row, color)

    # ── Axis styling ──────────────────────────────────────────────────────────
    def _style_ax(ax, centers, datasets, show_xticks, show_title, title, row_label, metric):
        ax.set_xticks(centers)
        if show_xticks:
            ax.set_xticklabels(
                [DATASET_LABELS.get(str(d), str(d)) for d in datasets],
                fontsize=FONT_TICK, rotation=0,
            )
        else:
            ax.set_xticklabels([""] * len(datasets))
            ax.tick_params(axis="x", length=0)
        if show_title:
            ax.set_title(title, fontsize=FONT_TITLE, pad=4)
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        if row_label:
            ax.set_ylabel(row_label, fontsize=FONT_ROW, labelpad=5)
        if metric == "difficulty":
            ax.axhline(0.0, color="#888888", linewidth=0.6, linestyle="--", zorder=0)
        n = len(centers)
        half_gw = (centers[1] - centers[0]) / 2 if n > 1 else 0.5
        ax.set_xlim(centers[0] - half_gw, centers[-1] + half_gw)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_color("#333333")

    for col, (metric, err_col, panel_title) in enumerate(metrics):
        _style_ax(axes[0][col], top_centers, datasets, False,  True,  panel_title,
                  "4-Choice" if col == 0 else None, metric)
        _style_ax(axes[1][col], bot_centers, datasets, True,   False, None,
                  "10-Choice" if col == 0 else None, metric)

    # Shared y-axis ranges
    for col, (metric, _, _) in enumerate(metrics):
        ticks, ymin, ymax = YTICK_SPECS[metric]
        for row_idx in range(2):
            ax = axes[row_idx][col]
            ax.set_ylim(ymin, ymax)
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(
                lambda v, _: f"{v:.1f}" if v != int(v) else f"{int(v)}"
            ))

    # ── Icons above each bar: placed per-bar in the bottom x-axis area ────────
    ICON_Y = -0.11   # axes-fraction below x=0

    def place_icon(ax, x, logo_path, zoom):
        if not logo_path.exists():
            return
        img = plt.imread(logo_path)
        ab = AnnotationBbox(
            OffsetImage(img, zoom=zoom),
            (x, ICON_Y),
            xycoords=("data", "axes fraction"),
            frameon=False,
            box_alignment=(0.5, 0.5),
            pad=0.0,
            clip_on=False,
        )
        ax.add_artist(ab)

    # Top row icons (only on leftmost panel to keep it clean, or all — do all)
    for col in range(n_panels):
        ax_top = axes[0][col]
        ax_bot = axes[1][col]
        for g_idx, dataset in enumerate(datasets):
            for b_idx, (bar_key, gen, _) in enumerate(top_specs):
                x = top_centers[g_idx] + top_offsets[b_idx]
                if gen is None:
                    place_icon(ax_top, x, HUMAN_LOGO, zoom=0.11)
                else:
                    place_icon(ax_top, x, GEN_LOGOS[gen], zoom=LOGO_ZOOM_MAP[gen])
            for b_idx, (bar_key, gen, _) in enumerate(bot_specs):
                x = bot_centers[g_idx] + bot_offsets[b_idx]
                if "aug_human" in bar_key:
                    place_icon(ax_bot, x, HUMAN_LOGO, zoom=0.11)
                else:
                    place_icon(ax_bot, x, GEN_LOGOS[gen], zoom=LOGO_ZOOM_MAP[gen])

    # ── Two legend boxes ──────────────────────────────────────────────────────
    legend_y = -0.08   # figure fraction; tight bbox captures it

    # Color legend (left)
    color_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=EXT_HUMAN_COLOR, edgecolor="none", linewidth=0, label="Human"),
        plt.Rectangle((0, 0), 1, 1, facecolor=EXT_MODEL_COLOR,  edgecolor="none", linewidth=0, label="Model"),
    ]
    leg1 = fig.legend(
        handles=color_handles,
        loc="lower center",
        bbox_to_anchor=(0.30, legend_y),
        ncols=2,
        frameon=True,
        facecolor="#f0f0f0",
        edgecolor="#cccccc",
        framealpha=0.9,
        fontsize=FONT_TICK,
        handlelength=1.2,
        handleheight=0.9,
        title="Condition",
        title_fontsize=FONT_TICK,
    )
    fig.add_artist(leg1)

    # Icon legend (right) — text-only with icons rendered via AnnotationBbox
    # Build as a regular legend with blank patches, then annotate icons separately.
    icon_legend_items = [
        ("human",                           HUMAN_LOGO,              0.11),
        ("openai/gpt-5.2-2025-12-11",       GEN_LOGOS["openai/gpt-5.2-2025-12-11"],      0.048),
        ("google/gemini-3.1-pro-preview",    GEN_LOGOS["google/gemini-3.1-pro-preview"],   0.055),
        ("together/Qwen/Qwen3.5-397B-A17B",  GEN_LOGOS["together/Qwen/Qwen3.5-397B-A17B"], 0.072),
    ]
    # Use invisible rectangle patches so the legend spacing is preserved
    icon_handles = [
        plt.Rectangle((0, 0), 0, 0, facecolor="none", edgecolor="none", linewidth=0,
                       label=ICON_LEGEND_LABELS[key])
        for key, _, _ in icon_legend_items
    ]
    leg2 = fig.legend(
        handles=icon_handles,
        loc="lower center",
        bbox_to_anchor=(0.70, legend_y),
        ncols=4,
        frameon=True,
        facecolor="#f0f0f0",
        edgecolor="#cccccc",
        framealpha=0.9,
        fontsize=FONT_TICK,
        handlelength=0,
        handleheight=0,
        handletextpad=0.4,
        title="Generator",
        title_fontsize=FONT_TICK,
        columnspacing=1.5,
    )
    fig.add_artist(leg2)

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.12, top=0.94, hspace=0.12, wspace=0.18)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


# Keep the original 4-bar combined figure intact for backward compatibility.

def plot_combined_quality_summary(
    grouped: pd.DataFrame,
    ablation: pd.DataFrame,
    path: Path,
) -> Path:
    """Original 4-bar combined figure (4-choice + 10-choice in one figure).

    Retained for backward compatibility.  New code should call
    ``plot_4choice_quality_summary`` and ``plot_10choice_quality_summary``
    instead.
    """
    import matplotlib.ticker as mticker

    COMBINED_HUMAN_COLOR = "#E8DAB2"
    COMBINED_MODEL_COLOR = "#DD6E42"

    summary = build_combined_summary(grouped, ablation)
    datasets = sorted(summary["dataset"].unique(), key=dataset_sort_key)

    metrics = [
        ("difficulty", "difficulty_se", "IRT Difficulty"),
        ("discrimination", "discrimination_se", "IRT Discrimination"),
        ("mean_flaws", "mean_flaws_se", "Avg. Writing Flaws"),
    ]

    n_panels = len(metrics)
    n_datasets = len(datasets)

    bar_specs = [
        ("4-choice Human",  COMBINED_HUMAN_COLOR,  "",    "4-choice Human"),
        ("4-choice Model",  COMBINED_MODEL_COLOR,  "",    "4-choice Model"),
        ("10-choice Human", COMBINED_HUMAN_COLOR,  "//",  "10-choice Human"),
        ("10-choice Model", COMBINED_MODEL_COLOR,  "//",  "10-choice Model"),
    ]
    n_bars = len(bar_specs)
    bar_width = 0.16
    group_offsets = np.linspace(
        -(n_bars - 1) / 2 * bar_width,
        (n_bars - 1) / 2 * bar_width,
        n_bars,
    )
    group_gap = n_bars * bar_width + 0.18
    group_centers = np.arange(n_datasets) * group_gap

    fig, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels, 3.2), squeeze=False)

    FONT_TITLE = 9
    FONT_LABEL = 8
    FONT_TICK  = 7.5

    for col, (metric, err_col, panel_title) in enumerate(metrics):
        ax = axes[0][col]
        all_bar_tops: list[float] = []
        all_bar_bots: list[float] = []

        for g_idx, dataset in enumerate(datasets):
            ds_df = summary[summary["dataset"] == dataset].set_index("bar_label")
            for b_idx, (bar_label, color, hatch, _) in enumerate(bar_specs):
                x = group_centers[g_idx] + group_offsets[b_idx]
                if bar_label not in ds_df.index:
                    continue
                record = ds_df.loc[bar_label]
                value = float(record[metric]) if pd.notna(record[metric]) else float("nan")
                err   = float(record[err_col]) if pd.notna(record[err_col]) else 0.0
                if not np.isfinite(value):
                    continue
                ax.bar(
                    x, value,
                    width=bar_width, color=color, hatch=hatch,
                    edgecolor="none", linewidth=0,
                    yerr=err if np.isfinite(err) else None,
                    capsize=2,
                    error_kw={"elinewidth": 0.8, "ecolor": "#444444", "capthick": 0.8},
                    alpha=1.0,
                )
                finite_err = err if np.isfinite(err) else 0.0
                all_bar_tops.append(value + finite_err)
                all_bar_bots.append(value - finite_err)

        ax.set_xticks(group_centers)
        ax.set_xticklabels(
            [DATASET_LABELS.get(str(d), str(d)) for d in datasets],
            fontsize=FONT_TICK, rotation=0,
        )
        ax.set_title(panel_title, fontsize=FONT_TITLE, pad=4)
        ax.set_ylabel("", fontsize=FONT_LABEL)
        ax.tick_params(axis="y", labelsize=FONT_TICK)

        if metric == "difficulty":
            ax.axhline(0.0, color="#888888", linewidth=0.6, linestyle="--", zorder=0)

        if all_bar_tops:
            pad = (max(all_bar_tops) - min(all_bar_bots)) * 0.08
            ylo = min(all_bar_bots) - pad
            yhi = max(all_bar_tops) + pad
            if metric == "discrimination":
                ylo = min(ylo, 0.90)
            elif metric == "difficulty":
                ylo = min(ylo, -0.1)
            ax.set_ylim(ylo, yhi)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="both"))

        half_gw = group_gap / 2
        ax.set_xlim(group_centers[0] - half_gw, group_centers[-1] + half_gw)

        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
            spine.set_color("#333333")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none", linewidth=0, hatch=hatch, label=label)
        for _, color, hatch, label in bar_specs
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.07),
        ncols=n_bars, frameon=False, fontsize=FONT_TICK,
        handlelength=1.4, handleheight=0.9, columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.92, wspace=0.32)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def dataset_sort_key(dataset: str) -> tuple[int, str]:

    value = str(dataset)
    return (DATASET_ORDER.index(value), value) if value in DATASET_ORDER else (len(DATASET_ORDER), value)


def setting_sort_key(setting: str) -> tuple[int, str]:
    value = str(setting)
    return (SETTING_NAMES.index(value), value) if value in SETTING_NAMES else (len(SETTING_NAMES), value)


def plot_irt_quality(summary: pd.DataFrame, path: Path) -> Path:
    settings = list(SETTING_NAMES)
    plot = summary[summary["setting"].isin(settings)].copy()
    datasets = sorted(plot["dataset"].unique(), key=dataset_sort_key)
    active_settings = [setting for setting in settings if setting in set(plot["setting"])]
    colors = [plt.get_cmap("tab10")(idx % 10) for idx in range(len(active_settings))]

    fig, axes = plt.subplots(3, len(datasets), figsize=(5.3 * len(datasets), 10.0), squeeze=False)
    for col, dataset in enumerate(datasets):
        dataset_df = plot[plot["dataset"] == dataset].set_index("setting")
        x = np.arange(len(active_settings))
        labels = [SETTING_SHORT_LABELS.get(setting, setting) for setting in active_settings]

        for row, (metric, err_col, ylabel) in enumerate(
            [
                ("difficulty", "difficulty_se", "IRT Difficulty\n(higher = harder)"),
                ("discrimination", "discrimination_se", "IRT Discrimination\n(higher = separates better)"),
                ("mean_flaws", "mean_flaws_se", "Avg. Writing Flaws\n(lower = higher quality)"),
            ]
        ):
            ax = axes[row][col]
            values = np.array([float(dataset_df.loc[setting, metric]) if setting in dataset_df.index else np.nan for setting in active_settings])
            errors = np.array([float(dataset_df.loc[setting, err_col]) if setting in dataset_df.index else 0.0 for setting in active_settings])
            bars = ax.bar(
                x,
                np.nan_to_num(values, nan=0.0),
                yerr=np.nan_to_num(errors, nan=0.0),
                capsize=3,
                color=colors,
                alpha=0.88,
                edgecolor="black",
                linewidth=0.5,
            )
            for idx, value in enumerate(values):
                if np.isnan(value):
                    bars[idx].set_color("#d9d9d9")
                    bars[idx].set_hatch("//")
                    ax.text(x[idx], 0.02, "missing", ha="center", va="bottom", rotation=90, fontsize=7)

            ax.set_title(str(dataset) if row == 0 else "")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=28, ha="right")
            ax.set_ylabel(ylabel if col == 0 else "")
            ax.grid(axis="y", alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if metric == "discrimination":
                ymax = max(float(np.nanmax(values + errors)) * 1.08, 0.6)
                ax.set_ylim(0.9, ymax)

    fig.suptitle("Question Quality by Generation Setting")
    fig.subplots_adjust(top=0.92, bottom=0.1, hspace=0.42, wspace=0.22)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ranked_bars(frame: pd.DataFrame, label_col: str, estimate_col: str, title: str, xlabel: str, path: Path) -> Path:
    plot = frame.sort_values(estimate_col, ascending=True).reset_index(drop=True)
    y = np.arange(len(plot))
    values = plot[estimate_col].to_numpy(dtype=float)
    colors = [plt.get_cmap("tab10")(idx % 10) for idx in range(len(plot))]

    fig, ax = plt.subplots(figsize=(8.2, max(3.4, 0.5 * len(plot))))
    ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.5, alpha=0.88)
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(plot[label_col].astype(str))
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    span = max(float(values.max() - values.min()), 1.0)
    pad = 0.03 * span
    rank_order = plot[estimate_col].rank(method="first", ascending=False).astype(int)
    for idx, (value, rank) in enumerate(zip(values, rank_order, strict=True)):
        ha = "left" if value >= 0 else "right"
        x = value + pad if value >= 0 else value - pad
        ax.text(x, idx, f"#{rank}", ha=ha, va="center", fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_item_fit(items: pd.DataFrame, fit: pd.DataFrame, path: Path) -> Path:
    merged = items.merge(fit, on=["item_id", "dataset", "sample_id", "generator", "setting", "question"])
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    ax.scatter(merged["difficulty"], merged["outfit"], s=11, alpha=0.34, color="#1f77b4", edgecolors="none")
    ax.axhline(1.5, color="#e45756", linestyle="--", linewidth=1)
    ax.axhline(0.7, color="#72b7b2", linestyle="--", linewidth=1)
    ax.set_xlabel("relative instantiated item difficulty")
    ax.set_ylabel("outfit")
    ax.set_title("Item fit vs relative difficulty")
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
