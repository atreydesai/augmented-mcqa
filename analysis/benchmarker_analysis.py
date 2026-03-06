"""
BenchMarker Writing Flaw + Accuracy Analysis
Run: uv run python analysis/benchmarker_analysis.py
Figures saved to: analysis/figures/
"""

# =============================================================================
# SECTION 0: Setup & Data Loading
# =============================================================================
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from datasets import load_from_disk

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
FLAW_JSONL = ROOT / "datasets/benchmarker_results/atrey_writing_flaw_rows.jsonl"
RESULTS_TXT = ROOT / "datasets/benchmarker_results/results.txt"
ACCURACY_CSV = (
    ROOT
    / "results/final5_full_gpt-5.2-2025-12-11_20260226_001349"
    / "final5_plots/tables/final5_results_summary.csv"
)
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Arrow results constants ────────────────────────────────────────────────────
RESULTS_DIR = ROOT / "results/final5_full_gpt-5.2-2025-12-11_20260226_001349"
GENERATOR = "gpt-5.2-2025-12-11"

# ── Constants ──────────────────────────────────────────────────────────────────
CONFIG_ORDER = [
    "human_from_scratch",
    "model_from_scratch",
    "augment_human",
    "augment_model",
    "augment_ablation",
]
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
    "avoid_k_type", "avoid_negatives", "avoid_repetition", "clear_language",
    "equal_length_options", "focused_stem", "grammatical_consistency",
    "no_absolute_terms", "no_all_of_the_above", "no_convergence_cues",
    "no_extraneous_info", "no_fill_in_blank", "no_logical_cues",
    "no_none_of_the_above", "no_vague_terms", "ordered_options",
    "plausible_distractors", "problem_in_stem", "single_best_answer",
]

PALETTE = {
    "human_from_scratch": "#2196F3",   # blue
    "model_from_scratch": "#FF9800",   # orange
    "augment_human": "#4CAF50",        # green
    "augment_model": "#F44336",        # red
    "augment_ablation": "#9C27B0",     # purple
}
DATASET_PALETTE = {
    "arc_challenge": "#1976D2",
    "gpqa": "#388E3C",
    "mmlu_pro": "#D32F2F",
}

# ── Load Writing Flaw Data ─────────────────────────────────────────────────────
print("Loading writing flaw data ...")
rows = []
with open(FLAW_JSONL) as f:
    for line in f:
        obj = json.loads(line)
        wf = obj["writing_flaw"]
        answer_arr = ast.literal_eval(wf["answer"])  # list of 19 'pass'/'fail'
        rule_bools = [v == "pass" for v in answer_arr]  # True = pass = no flaw
        rows.append(
            {
                "dataset": obj["dataset"],
                "config": obj["config"],
                "flaw_value": wf["value"],                   # fraction of rules PASSED
                "n_flaws": sum(v == "fail" for v in answer_arr),
                **{f"rule_{r}": b for r, b in zip(RULES_ORDER, rule_bools)},
            }
        )

df = pd.DataFrame(rows)
df["config"] = pd.Categorical(df["config"], categories=CONFIG_ORDER, ordered=True)
df["has_ge2_flaws"] = df["n_flaws"] >= 2

# ── Build long-format rule DataFrame ──────────────────────────────────────────
rule_cols = [f"rule_{r}" for r in RULES_ORDER]
df_long = df.melt(
    id_vars=["dataset", "config", "flaw_value"],
    value_vars=rule_cols,
    var_name="rule_col",
    value_name="passed",
)
df_long["rule"] = df_long["rule_col"].str.replace("rule_", "", regex=False)
df_long["failed"] = ~df_long["passed"]  # True = failed this rule

print(f"  Loaded {len(df):,} rows × {len(rule_cols)} rules.")
print(f"  Datasets: {df['dataset'].unique().tolist()}")
print(f"  Configs: {df['config'].cat.categories.tolist()}\n")

# ── Load Accuracy CSV ──────────────────────────────────────────────────────────
print("Loading accuracy data ...")
acc_df = pd.read_csv(ACCURACY_CSV)
acc_df.rename(columns={"setting": "config"}, inplace=True)
acc_df["config"] = pd.Categorical(acc_df["config"], categories=CONFIG_ORDER, ordered=True)
print(f"  Loaded {len(acc_df):,} rows.\n")


# =============================================================================
# SECTION 1: Validate & Reproduce Summary
# =============================================================================
print("=" * 70)
print("SECTION 1: VALIDATE AGAINST results.txt")
print("=" * 70)

# Compute mean flaw_value and p(≥2 flaws) per (dataset, config) with 95% CI
def mean_ci(series, z=1.96):
    n = len(series)
    m = series.mean()
    se = series.std(ddof=1) / np.sqrt(n)
    return m, se * z

summary_rows = []
for (ds, cfg), grp in df.groupby(["dataset", "config"], observed=True):
    mean_flaw, ci_flaw = mean_ci(grp["flaw_value"])
    mean_p2, ci_p2 = mean_ci(grp["has_ge2_flaws"].astype(float))
    summary_rows.append(
        {
            "dataset": ds,
            "config": cfg,
            "writing_flaws_mean": mean_flaw,
            "writing_flaws_ci": ci_flaw,
            "p_ge2_mean": mean_p2,
            "p_ge2_ci": ci_p2,
            "n": len(grp),
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_df["config"] = pd.Categorical(summary_df["config"], categories=CONFIG_ORDER, ordered=True)
summary_df.sort_values(["dataset", "config"], inplace=True)

print("\nReproduced summary (compare to results.txt):")
print(f"{'dataset':<20} {'config':<25} {'flaw_value':>12} {'p(≥2 flaws)':>14} {'n':>6}")
print("-" * 80)
for _, r in summary_df.iterrows():
    print(
        f"{r['dataset']:<20} {r['config']:<25} "
        f"{r['writing_flaws_mean']:.3f} ± {r['writing_flaws_ci']:.3f}  "
        f"{r['p_ge2_mean']*100:.2f}% ± {r['p_ge2_ci']*100:.2f}%  "
        f"{r['n']:>6}"
    )

# =============================================================================
# SECTION 2: Writing Quality — Aggregate Comparison
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: WRITING QUALITY — AGGREGATE COMPARISON")
print("=" * 70)


def _grouped_bar_quality(ax, data, metric_col, ylabel, title, ci_col=None):
    """Helper: grouped bar chart per config, faceted values in caller."""
    n_configs = len(CONFIG_ORDER)
    x = np.arange(len(DATASET_ORDER))
    width = 0.14
    offsets = np.linspace(-(n_configs - 1) / 2, (n_configs - 1) / 2, n_configs) * width

    for i, cfg in enumerate(CONFIG_ORDER):
        vals, errs = [], []
        for ds in DATASET_ORDER:
            row = data[(data["dataset"] == ds) & (data["config"] == cfg)]
            if len(row) == 0:
                vals.append(0); errs.append(0)
            else:
                vals.append(float(row[metric_col].iloc[0]))
                errs.append(float(row[ci_col].iloc[0]) if ci_col else 0)
        bars = ax.bar(
            x + offsets[i], vals, width=width * 0.9,
            label=CONFIG_LABELS[cfg].replace("\n", " "),
            color=PALETTE[cfg], alpha=0.85,
            yerr=errs if ci_col else None,
            capsize=3, error_kw={"elinewidth": 1},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))


# ── Fig 2a: mean flaw_value (fraction passed) ─────────────────────────────────
fig2a, ax2a = plt.subplots(figsize=(8, 5))
_grouped_bar_quality(
    ax2a, summary_df, "writing_flaws_mean", "Mean fraction of rules passed (↑ better)",
    "Fig 2a: Writing Quality by Config × Dataset (mean flaw_value)",
    ci_col="writing_flaws_ci",
)
ax2a.set_ylim(0.7, 1.0)
plt.tight_layout()
fig2a.savefig(FIGURES_DIR / "fig2a_quality_mean.png", dpi=150)
plt.close(fig2a)
print("  Saved fig2a_quality_mean.png")

# ── Fig 2b: p(≥2 flaws) ───────────────────────────────────────────────────────
fig2b, ax2b = plt.subplots(figsize=(8, 5))
_grouped_bar_quality(
    ax2b, summary_df, "p_ge2_mean", "P(≥2 writing flaws) ↓ better",
    "Fig 2b: Fraction of Questions with ≥2 Writing Flaws",
    ci_col="p_ge2_ci",
)
ax2b.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax2b.set_ylim(0, 1.05)
plt.tight_layout()
fig2b.savefig(FIGURES_DIR / "fig2b_p_ge2_flaws.png", dpi=150)
plt.close(fig2b)
print("  Saved fig2b_p_ge2_flaws.png")

# ── Fig 2c: 4-choice vs 10-choice ─────────────────────────────────────────────
df["choice_group"] = df["config"].map(
    lambda c: "4-choice\n(from_scratch)" if "from_scratch" in str(c) else "10-choice\n(augment_*)"
)
grp_4vs10 = (
    df.groupby(["dataset", "choice_group"])
    .agg(
        mean_flaw=("flaw_value", "mean"),
        ci_flaw=("flaw_value", lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))),
        mean_p2=("has_ge2_flaws", "mean"),
        ci_p2=("has_ge2_flaws", lambda x: 1.96 * x.astype(float).std(ddof=1) / np.sqrt(len(x))),
    )
    .reset_index()
)

fig2c, axes2c = plt.subplots(1, 2, figsize=(11, 5))
groups = ["4-choice\n(from_scratch)", "10-choice\n(augment_*)"]
x = np.arange(len(DATASET_ORDER))
width = 0.3
colors = ["#2196F3", "#F44336"]

for ax, metric, ci_col, ylabel, fmt in [
    (axes2c[0], "mean_flaw", "ci_flaw", "Mean fraction of rules passed", "{:.2f}"),
    (axes2c[1], "mean_p2", "ci_p2", "P(≥2 writing flaws)", "{:.1%}"),
]:
    for j, (grp, col) in enumerate(zip(groups, colors)):
        vals, errs = [], []
        for ds in DATASET_ORDER:
            sub = grp_4vs10[(grp_4vs10["dataset"] == ds) & (grp_4vs10["choice_group"] == grp)]
            vals.append(float(sub[metric].iloc[0]) if len(sub) else 0)
            errs.append(float(sub[ci_col].iloc[0]) if len(sub) else 0)
        ax.bar(
            x + (j - 0.5) * width, vals, width=width * 0.9,
            label=grp.replace("\n", " "), color=col, alpha=0.85,
            yerr=errs, capsize=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)

axes2c[0].set_title("Fig 2c (left): Quality — 4-choice vs 10-choice")
axes2c[1].set_title("Fig 2c (right): P(≥2 flaws) — 4-choice vs 10-choice")
axes2c[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
plt.tight_layout()
fig2c.savefig(FIGURES_DIR / "fig2c_4choice_vs_10choice.png", dpi=150)
plt.close(fig2c)
print("  Saved fig2c_4choice_vs_10choice.png")

# Print key finding
arc_hum = float(summary_df[(summary_df.dataset=="arc_challenge") & (summary_df.config=="human_from_scratch")]["writing_flaws_mean"].iloc[0])
arc_mdl = float(summary_df[(summary_df.dataset=="arc_challenge") & (summary_df.config=="model_from_scratch")]["writing_flaws_mean"].iloc[0])
arc_aug = float(summary_df[(summary_df.dataset=="arc_challenge") & (summary_df.config=="augment_human")]["writing_flaws_mean"].iloc[0])
print(f"\n  Key Finding (ARC): human_from_scratch={arc_hum:.3f}, model_from_scratch={arc_mdl:.3f}, augment_human={arc_aug:.3f}")
print(f"  → 4-choice human vs model gap: {arc_hum - arc_mdl:.3f}")
print(f"  → 4→10 choice expansion cost: {arc_hum - arc_aug:.3f}")

# =============================================================================
# SECTION 3: Per-Rule Failure Rate Analysis
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: PER-RULE FAILURE RATE ANALYSIS")
print("=" * 70)

# Build fail_rate table: (dataset, config, rule) → fail_rate
rule_fail = (
    df_long.groupby(["dataset", "config", "rule"], observed=True)["failed"]
    .mean()
    .reset_index()
    .rename(columns={"failed": "fail_rate"})
)
rule_fail["config"] = pd.Categorical(rule_fail["config"], categories=CONFIG_ORDER, ordered=True)

# ── Fig 3a: Heatmap per dataset ────────────────────────────────────────────────
fig3a, axes3a = plt.subplots(1, 3, figsize=(18, 8))
for ax, ds in zip(axes3a, DATASET_ORDER):
    pivot = (
        rule_fail[rule_fail["dataset"] == ds]
        .pivot(index="rule", columns="config", values="fail_rate")
        .reindex(index=RULES_ORDER, columns=CONFIG_ORDER)
    )
    sns.heatmap(
        pivot, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.4, cbar_kws={"shrink": 0.6},
        xticklabels=[CONFIG_SHORT[c] for c in CONFIG_ORDER],
    )
    ax.set_title(f"Fig 3a: {DATASET_LABELS[ds]}\nFail Rate per Rule × Config", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("Rule" if ax == axes3a[0] else "")
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)

plt.suptitle("Per-Rule Failure Rates (red = high fail rate)", fontsize=12, y=1.01)
plt.tight_layout()
fig3a.savefig(FIGURES_DIR / "fig3a_rule_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig3a)
print("  Saved fig3a_rule_heatmap.png")

# ── Fig 3b: Model-sensitivity (model_from_scratch - human_from_scratch) ────────
sensitivity = []
for ds in DATASET_ORDER:
    for rule in RULES_ORDER:
        sub = rule_fail[(rule_fail["dataset"] == ds) & (rule_fail["rule"] == rule)]
        hum = float(sub[sub["config"] == "human_from_scratch"]["fail_rate"].iloc[0]) if len(sub[sub["config"] == "human_from_scratch"]) else 0
        mdl = float(sub[sub["config"] == "model_from_scratch"]["fail_rate"].iloc[0]) if len(sub[sub["config"] == "model_from_scratch"]) else 0
        sensitivity.append({"dataset": ds, "rule": rule, "delta": mdl - hum})

sens_df = pd.DataFrame(sensitivity)
# Average delta across datasets
sens_avg = sens_df.groupby("rule")["delta"].mean().reindex(RULES_ORDER)

fig3b, ax3b = plt.subplots(figsize=(11, 6))
colors_bar = ["#F44336" if v > 0 else "#2196F3" for v in sens_avg.values]
ax3b.barh(RULES_ORDER, sens_avg.values, color=colors_bar, alpha=0.85)
ax3b.axvline(0, color="black", linewidth=0.8)
ax3b.set_xlabel("Δ fail rate (model_from_scratch − human_from_scratch, avg. across datasets)")
ax3b.set_title("Fig 3b: Model-Sensitivity — Which rules fail more with model distractors?")
ax3b.tick_params(axis="y", labelsize=8)
plt.tight_layout()
fig3b.savefig(FIGURES_DIR / "fig3b_model_sensitivity.png", dpi=150)
plt.close(fig3b)
print("  Saved fig3b_model_sensitivity.png")

# Print top rule violations by model distractors
print("\n  Top rules impacted by switching to model distractors (avg Δ fail rate):")
top5_model = sens_avg.sort_values(ascending=False).head(5)
for rule, delta in top5_model.items():
    print(f"    {rule}: +{delta:.3f}")

# ── Fig 3c: Augmentation penalty (augment_human - human_from_scratch) ─────────
aug_penalty = []
for ds in DATASET_ORDER:
    for rule in RULES_ORDER:
        sub = rule_fail[(rule_fail["dataset"] == ds) & (rule_fail["rule"] == rule)]
        hum = float(sub[sub["config"] == "human_from_scratch"]["fail_rate"].iloc[0]) if len(sub[sub["config"] == "human_from_scratch"]) else 0
        aug = float(sub[sub["config"] == "augment_human"]["fail_rate"].iloc[0]) if len(sub[sub["config"] == "augment_human"]) else 0
        aug_penalty.append({"dataset": ds, "rule": rule, "delta": aug - hum})

aug_df = pd.DataFrame(aug_penalty)
aug_avg = aug_df.groupby("rule")["delta"].mean().reindex(RULES_ORDER)

fig3c, ax3c = plt.subplots(figsize=(11, 6))
colors_aug = ["#F44336" if v > 0 else "#2196F3" for v in aug_avg.values]
ax3c.barh(RULES_ORDER, aug_avg.values, color=colors_aug, alpha=0.85)
ax3c.axvline(0, color="black", linewidth=0.8)
ax3c.set_xlabel("Δ fail rate (augment_human − human_from_scratch, avg. across datasets)")
ax3c.set_title("Fig 3c: Augmentation Penalty — Cost of expanding to 10 choices (human distractors)")
ax3c.tick_params(axis="y", labelsize=8)
plt.tight_layout()
fig3c.savefig(FIGURES_DIR / "fig3c_augmentation_penalty.png", dpi=150)
plt.close(fig3c)
print("  Saved fig3c_augmentation_penalty.png")

print("\n  Top rules impacted by 4→10 choice expansion (avg Δ fail rate):")
top5_aug = aug_avg.sort_values(ascending=False).head(5)
for rule, delta in top5_aug.items():
    print(f"    {rule}: +{delta:.3f}")

# Near-zero rules (non-discriminating)
near_zero = sens_avg[sens_avg.abs() < 0.02].index.tolist()
print(f"\n  Near-zero model-sensitivity rules (non-discriminating): {near_zero}")

# =============================================================================
# SECTION 4: Accuracy Analysis
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: ACCURACY ANALYSIS")
print("=" * 70)

# Mean accuracy ± stderr across eval_models, full_question mode
fq_acc = acc_df[acc_df["mode"] == "full_question"].copy()
acc_by_config = (
    fq_acc.groupby(["dataset", "config"], observed=True)
    .agg(
        mean_acc=("accuracy", "mean"),
        se_acc=("stderr", lambda x: np.sqrt((x**2).mean())),  # pooled stderr
        mean_delta=("delta_over_random", "mean"),
    )
    .reset_index()
)
acc_by_config["config"] = pd.Categorical(acc_by_config["config"], categories=CONFIG_ORDER, ordered=True)

# ── Fig 4a: Accuracy by config × dataset (averaged across eval models) ─────────
fig4a, ax4a = plt.subplots(figsize=(8, 5))
_grouped_bar_quality(
    ax4a, acc_by_config, "mean_acc",
    "Mean accuracy (full_question mode, avg. 3 models)",
    "Fig 4a: Accuracy by Config × Dataset",
)
ax4a.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
plt.tight_layout()
fig4a.savefig(FIGURES_DIR / "fig4a_accuracy_avg.png", dpi=150)
plt.close(fig4a)
print("  Saved fig4a_accuracy_avg.png")

# ── Fig 4b: Split by eval model ────────────────────────────────────────────────
eval_models = sorted(fq_acc["eval_model"].unique())
eval_short = {m: m.split("_")[1][:12] if "_" in m else m[:12] for m in eval_models}

fig4b, axes4b = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, em in zip(axes4b, eval_models):
    sub = fq_acc[fq_acc["eval_model"] == em]
    acc_sub = (
        sub.groupby(["dataset", "config"], observed=True)
        .agg(mean_acc=("accuracy", "mean"), se_acc=("stderr", "mean"))
        .reset_index()
    )
    acc_sub["config"] = pd.Categorical(acc_sub["config"], categories=CONFIG_ORDER, ordered=True)

    n_configs = len(CONFIG_ORDER)
    x = np.arange(len(DATASET_ORDER))
    width = 0.14
    offsets = np.linspace(-(n_configs-1)/2, (n_configs-1)/2, n_configs) * width

    for i, cfg in enumerate(CONFIG_ORDER):
        vals, errs = [], []
        for ds in DATASET_ORDER:
            row = acc_sub[(acc_sub["dataset"] == ds) & (acc_sub["config"] == cfg)]
            vals.append(float(row["mean_acc"].iloc[0]) if len(row) else 0)
            errs.append(float(row["se_acc"].iloc[0]) if len(row) else 0)
        ax.bar(
            x + offsets[i], vals, width=width * 0.9,
            label=CONFIG_SHORT[cfg], color=PALETTE[cfg], alpha=0.85,
            yerr=errs, capsize=2, error_kw={"elinewidth": 0.8},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=8)
    ax.set_title(f"{eval_short[em]}", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=6, ncol=2)

axes4b[0].set_ylabel("Accuracy")
plt.suptitle("Fig 4b: Accuracy Split by Eval Model (full_question mode)", fontsize=11)
plt.tight_layout()
fig4b.savefig(FIGURES_DIR / "fig4b_accuracy_by_model.png", dpi=150)
plt.close(fig4b)
print("  Saved fig4b_accuracy_by_model.png")

# ── Fig 4c: Delta over random baseline ────────────────────────────────────────
fig4c, ax4c = plt.subplots(figsize=(8, 5))
_grouped_bar_quality(
    ax4c, acc_by_config, "mean_delta",
    "Δ accuracy over random baseline (full_question)",
    "Fig 4c: Accuracy Delta Over Random Baseline",
)
ax4c.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
plt.tight_layout()
fig4c.savefig(FIGURES_DIR / "fig4c_delta_over_random.png", dpi=150)
plt.close(fig4c)
print("  Saved fig4c_delta_over_random.png")

# =============================================================================
# SECTION 5: Cross-Reference — Validity vs. Accuracy
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: CROSS-REFERENCE — VALIDITY vs. ACCURACY")
print("=" * 70)

# Aggregate to 15 (dataset × config) cells
validity_15 = (
    df.groupby(["dataset", "config"], observed=True)
    .agg(
        mean_validity=("flaw_value", "mean"),
        mean_p2=("has_ge2_flaws", "mean"),
    )
    .reset_index()
)

# Accuracy: full_question mode, mean across eval_models
acc_15 = (
    fq_acc.groupby(["dataset", "config"], observed=True)
    .agg(mean_accuracy=("accuracy", "mean"))
    .reset_index()
)

cross = pd.merge(validity_15, acc_15, on=["dataset", "config"])
cross["choice_group"] = cross["config"].map(
    lambda c: "4-choice" if "from_scratch" in str(c) else "10-choice"
)
assert len(cross) == 15, f"Expected 15 rows, got {len(cross)}"
print(f"  Cross-reference DataFrame: {len(cross)} rows (3 datasets × 5 configs) ✓")

# Statistical correlation
r_val, p_val = stats.pearsonr(cross["mean_validity"], cross["mean_accuracy"])
r_p2, p_p2 = stats.pearsonr(cross["mean_p2"], cross["mean_accuracy"])
print(f"\n  Pearson r (validity vs. accuracy): r={r_val:.3f}, p={p_val:.4f}")
print(f"  Pearson r (p(≥2 flaws) vs. accuracy): r={r_p2:.3f}, p={p_p2:.4f}")
corr_desc = "positively correlated" if r_val > 0 else "negatively correlated"
sig_desc = "statistically significant" if p_val < 0.05 else "not statistically significant"
print(f"  → Writing quality and accuracy are {corr_desc} ({sig_desc}, p={p_val:.4f})")

# ── Fig 5a: Scatter validity vs accuracy ──────────────────────────────────────
fig5a, ax5a = plt.subplots(figsize=(8, 6))
markers = {"4-choice": "o", "10-choice": "s"}
for _, row in cross.iterrows():
    ax5a.scatter(
        row["mean_validity"], row["mean_accuracy"],
        color=DATASET_PALETTE[row["dataset"]],
        marker=markers[row["choice_group"]],
        s=100, zorder=3,
        label=f"{DATASET_LABELS[row['dataset']]} ({row['choice_group']})",
    )
    ax5a.annotate(
        CONFIG_SHORT[row["config"]],
        (row["mean_validity"], row["mean_accuracy"]),
        textcoords="offset points", xytext=(5, 3), fontsize=7,
    )

# Regression line
x_fit = np.linspace(cross["mean_validity"].min(), cross["mean_validity"].max(), 100)
slope, intercept, *_ = stats.linregress(cross["mean_validity"], cross["mean_accuracy"])
ax5a.plot(x_fit, slope * x_fit + intercept, "k--", linewidth=1, label=f"r={r_val:.2f}, p={p_val:.3f}")

# Deduplicate legend
handles, labels = ax5a.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax5a.legend(by_label.values(), by_label.keys(), fontsize=8, loc="best")
ax5a.set_xlabel("Mean writing quality (fraction of rules passed, ↑ better)")
ax5a.set_ylabel("Mean accuracy (full_question, 3 models)")
ax5a.set_title("Fig 5a: Writing Quality vs. Accuracy\n(one point per dataset × config)")
ax5a.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
plt.tight_layout()
fig5a.savefig(FIGURES_DIR / "fig5a_validity_vs_accuracy.png", dpi=150)
plt.close(fig5a)
print("  Saved fig5a_validity_vs_accuracy.png")

# ── Fig 5b: Delta comparison — validity drop vs accuracy drop ─────────────────
baseline_config = "human_from_scratch"
baselines_valid = cross[cross["config"] == baseline_config].set_index("dataset")["mean_validity"]
baselines_acc = cross[cross["config"] == baseline_config].set_index("dataset")["mean_accuracy"]

delta_rows = []
for _, row in cross[cross["config"] != baseline_config].iterrows():
    ds = row["dataset"]
    delta_rows.append(
        {
            "dataset": ds,
            "config": row["config"],
            "label": f"{DATASET_LABELS[ds]}\n{CONFIG_SHORT[row['config']]}",
            "validity_drop": row["mean_validity"] - baselines_valid[ds],
            "accuracy_drop": row["mean_accuracy"] - baselines_acc[ds],
        }
    )
delta_df = pd.DataFrame(delta_rows)
delta_df["config"] = pd.Categorical(delta_df["config"], categories=CONFIG_ORDER, ordered=True)
delta_df.sort_values(["dataset", "config"], inplace=True)

fig5b, ax5b = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(delta_df))
width = 0.35

ax5b.bar(x_pos - width/2, delta_df["validity_drop"], width, label="Δ validity", color="#4CAF50", alpha=0.8)
ax5b.bar(x_pos + width/2, delta_df["accuracy_drop"], width, label="Δ accuracy", color="#F44336", alpha=0.8)

ax5b.axhline(0, color="black", linewidth=0.7)
ax5b.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax5b.set_ylabel("Δ from human_from_scratch baseline")
ax5b.set_xticks(x_pos)
ax5b.set_xticklabels(delta_df["label"], fontsize=7, rotation=30, ha="right")
ax5b.legend(fontsize=9, loc="lower left")
ax5b.set_title("Fig 5b: Validity Drop vs. Accuracy Drop from human_from_scratch baseline")
plt.tight_layout()
fig5b.savefig(FIGURES_DIR / "fig5b_validity_vs_accuracy_delta.png", dpi=150)
plt.close(fig5b)
print("  Saved fig5b_validity_vs_accuracy_delta.png")

# ── Fig 5c: p(≥2 flaws) vs accuracy ───────────────────────────────────────────
fig5c, ax5c = plt.subplots(figsize=(8, 6))
for _, row in cross.iterrows():
    ax5c.scatter(
        row["mean_p2"], row["mean_accuracy"],
        color=DATASET_PALETTE[row["dataset"]],
        marker=markers[row["choice_group"]],
        s=100, zorder=3,
    )
    ax5c.annotate(
        CONFIG_SHORT[row["config"]],
        (row["mean_p2"], row["mean_accuracy"]),
        textcoords="offset points", xytext=(5, 3), fontsize=7,
    )

x_fit2 = np.linspace(cross["mean_p2"].min(), cross["mean_p2"].max(), 100)
slope2, intercept2, *_ = stats.linregress(cross["mean_p2"], cross["mean_accuracy"])
ax5c.plot(x_fit2, slope2 * x_fit2 + intercept2, "k--", linewidth=1, label=f"r={r_p2:.2f}, p={p_p2:.3f}")

# Legend: dataset/choice group markers
for ds in DATASET_ORDER:
    ax5c.scatter([], [], color=DATASET_PALETTE[ds], label=DATASET_LABELS[ds], s=80)
for cg, mk in markers.items():
    ax5c.scatter([], [], color="gray", marker=mk, label=cg, s=80)
ax5c.legend(fontsize=8)
ax5c.set_xlabel("P(≥2 writing flaws) ↑ = more flawed")
ax5c.set_ylabel("Mean accuracy (full_question, 3 models)")
ax5c.set_title("Fig 5c: P(≥2 Writing Flaws) vs. Accuracy")
ax5c.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax5c.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
plt.tight_layout()
fig5c.savefig(FIGURES_DIR / "fig5c_p2flaws_vs_accuracy.png", dpi=150)
plt.close(fig5c)
print("  Saved fig5c_p2flaws_vs_accuracy.png")

# =============================================================================
# SECTION 6: Within-Group Correlation
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: WITHIN-GROUP CORRELATION (4-choice vs 10-choice)")
print("=" * 70)

cross_4 = cross[cross.choice_group == "4-choice"]   # n=6
cross_10 = cross[cross.choice_group == "10-choice"]  # n=9

r4, p4 = stats.pearsonr(cross_4.mean_validity, cross_4.mean_accuracy)
r10, p10 = stats.pearsonr(cross_10.mean_validity, cross_10.mean_accuracy)

# Partial correlation controlling for choice_group binary covariate
cg_binary = (cross["choice_group"] == "4-choice").astype(float)
r_partial_xy, _ = stats.pearsonr(cross.mean_validity, cross.mean_accuracy)
r_partial_xz, _ = stats.pearsonr(cross.mean_validity, cg_binary)
r_partial_yz, _ = stats.pearsonr(cross.mean_accuracy, cg_binary)
denom = np.sqrt((1 - r_partial_xz**2) * (1 - r_partial_yz**2))
r_partial = (r_partial_xy - r_partial_xz * r_partial_yz) / denom if denom > 0 else float("nan")

print(f"\n  4-choice (n={len(cross_4)}):  r={r4:.3f},  p={p4:.4f}")
print(f"  10-choice (n={len(cross_10)}): r={r10:.3f}, p={p10:.4f}")
print(f"  Pooled (n=15):      r={r_val:.3f}, p={p_val:.4f}")
print(f"  Partial r (controlling for choice_group): {r_partial:.3f}")

# ── Fig 6a: Scatter with per-group regression lines ────────────────────────────
fig6a, ax6a = plt.subplots(figsize=(8, 6))
markers = {"4-choice": "o", "10-choice": "s"}
for _, row in cross.iterrows():
    ax6a.scatter(
        row["mean_validity"], row["mean_accuracy"],
        color=DATASET_PALETTE[row["dataset"]],
        marker=markers[row["choice_group"]],
        s=110, zorder=3,
    )
    ax6a.annotate(
        CONFIG_SHORT[row["config"]],
        (row["mean_validity"], row["mean_accuracy"]),
        textcoords="offset points", xytext=(5, 3), fontsize=7,
    )

# Per-group regression lines
for (grp, r_grp, p_grp, sub, ls) in [
    ("4-choice", r4, p4, cross_4, "--"),
    ("10-choice", r10, p10, cross_10, ":"),
]:
    x_g = np.linspace(sub.mean_validity.min(), sub.mean_validity.max(), 100)
    sl, ic, *_ = stats.linregress(sub.mean_validity, sub.mean_accuracy)
    ax6a.plot(x_g, sl * x_g + ic, ls, linewidth=1.5,
              label=f"{grp}: r={r_grp:.2f}, p={p_grp:.3f}")

# Proxy handles for dataset colours and shapes
for ds in DATASET_ORDER:
    ax6a.scatter([], [], color=DATASET_PALETTE[ds], label=DATASET_LABELS[ds], s=80)
for cg, mk in markers.items():
    ax6a.scatter([], [], color="gray", marker=mk, label=cg, s=80)

ax6a.legend(fontsize=7, loc="best")
ax6a.set_xlabel("Mean writing quality (fraction of rules passed, ↑ better)")
ax6a.set_ylabel("Mean accuracy (full_question, 3 models)")
ax6a.set_title(
    "Fig 6a: Within-Group Correlation\n(circle=4-choice, square=10-choice; separate regression lines)"
)
ax6a.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
plt.tight_layout()
fig6a.savefig(FIGURES_DIR / "fig6a_within_group_correlation.png", dpi=150)
plt.close(fig6a)
print("  Saved fig6a_within_group_correlation.png")

# =============================================================================
# SECTION 7: choices_only vs full_question Mode Comparison
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: choices_only vs full_question MODE COMPARISON")
print("=" * 70)

mode_acc = (
    acc_df.groupby(["dataset", "config", "mode"], observed=True)
    .agg(mean_acc=("accuracy", "mean"))
    .reset_index()
)

mode_pivot = mode_acc.pivot_table(
    index=["dataset", "config"], columns="mode", values="mean_acc"
).reset_index()
mode_pivot.columns.name = None

# Handle cases where a mode column might be missing
if "choices_only" not in mode_pivot.columns:
    mode_pivot["choices_only"] = float("nan")
if "full_question" not in mode_pivot.columns:
    mode_pivot["full_question"] = float("nan")

mode_pivot["delta"] = mode_pivot["choices_only"] - mode_pivot["full_question"]
mode_pivot["config"] = pd.Categorical(mode_pivot["config"], categories=CONFIG_ORDER, ordered=True)
mode_pivot.sort_values(["dataset", "config"], inplace=True)

print("\n  choices_only vs full_question accuracy (Δ = choices_only − full_question):")
print(f"  {'dataset':<15} {'config':<22} {'full_q':>8} {'co':>8} {'Δ':>8}")
print("  " + "-" * 65)
for _, row in mode_pivot.iterrows():
    print(
        f"  {row['dataset']:<15} {row['config']:<22} "
        f"{row['full_question']:.3f}   {row['choices_only']:.3f}   "
        f"{row['delta']:+.3f}"
    )

top_delta = mode_pivot.reindex(mode_pivot["delta"].abs().sort_values(ascending=False).index).head(3)
print("\n  Configs with largest stem effect (|Δ|):")
for _, row in top_delta.iterrows():
    direction = "choices_only better" if row["delta"] > 0 else "full_question better"
    print(f"    {row['dataset']} / {row['config']}: Δ={row['delta']:+.3f} ({direction})")

# Build label column for x-axis
mode_pivot["label"] = [
    f"{DATASET_LABELS[r['dataset']]}\n{CONFIG_SHORT[r['config']]}"
    for _, r in mode_pivot.iterrows()
]

# ── Fig 7a: Grouped bar choices_only vs full_question ─────────────────────────
fig7a, ax7a = plt.subplots(figsize=(14, 5))
x7 = np.arange(len(mode_pivot))
w = 0.35
ax7a.bar(x7 - w / 2, mode_pivot["full_question"], w, label="full_question",
         color="#1976D2", alpha=0.85)
ax7a.bar(x7 + w / 2, mode_pivot["choices_only"], w, label="choices_only",
         color="#F57C00", alpha=0.85)
ax7a.set_xticks(x7)
ax7a.set_xticklabels(mode_pivot["label"], fontsize=6.5, rotation=30, ha="right")
ax7a.set_ylabel("Mean accuracy (avg. 3 eval models)")
ax7a.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax7a.legend(fontsize=9)
ax7a.set_title("Fig 7a: choices_only vs full_question Accuracy per Dataset × Config")
plt.tight_layout()
fig7a.savefig(FIGURES_DIR / "fig7a_mode_accuracy_comparison.png", dpi=150)
plt.close(fig7a)
print("  Saved fig7a_mode_accuracy_comparison.png")

# ── Fig 7b: Delta bar ─────────────────────────────────────────────────────────
fig7b, ax7b = plt.subplots(figsize=(14, 5))
bar_colors = ["#4CAF50" if d > 0 else "#F44336" for d in mode_pivot["delta"]]
ax7b.bar(x7, mode_pivot["delta"], color=bar_colors, alpha=0.85)
ax7b.axhline(0, color="black", linewidth=0.8)
ax7b.set_xticks(x7)
ax7b.set_xticklabels(mode_pivot["label"], fontsize=6.5, rotation=30, ha="right")
ax7b.set_ylabel("Δ accuracy (choices_only − full_question)")
ax7b.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax7b.set_title(
    "Fig 7b: Δ Accuracy (choices_only − full_question)\n"
    "Green = stem adds difficulty; Red = stem helps"
)
plt.tight_layout()
fig7b.savefig(FIGURES_DIR / "fig7b_mode_delta.png", dpi=150)
plt.close(fig7b)
print("  Saved fig7b_mode_delta.png")

# =============================================================================
# SECTION 8: Per-Rule × Accuracy Correlation
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 8: PER-RULE × ACCURACY CORRELATION")
print("=" * 70)

rule_acc_rows = []
for rule in RULES_ORDER:
    rule_data = rule_fail[rule_fail.rule == rule].merge(
        cross[["dataset", "config", "mean_accuracy"]], on=["dataset", "config"]
    )
    if len(rule_data) >= 3:
        r_r, p_r = stats.pearsonr(rule_data.fail_rate, rule_data.mean_accuracy)
    else:
        r_r, p_r = float("nan"), float("nan")
    rule_acc_rows.append({"rule": rule, "r": r_r, "p": p_r})

rule_acc_df = pd.DataFrame(rule_acc_rows).sort_values("r").reset_index(drop=True)

print("\n  Top 5 rules (most negative r — higher fail rate → lower accuracy):")
for _, row in rule_acc_df.head(5).iterrows():
    print(f"    {row['rule']:<30} r={row['r']:+.3f}, p={row['p']:.4f}")

print("\n  Top 5 rules (most positive r — higher fail rate → higher accuracy):")
for _, row in rule_acc_df.tail(5).iloc[::-1].iterrows():
    print(f"    {row['rule']:<30} r={row['r']:+.3f}, p={row['p']:.4f}")

# ── Fig 8a: Horizontal bar chart ──────────────────────────────────────────────
fig8a, ax8a = plt.subplots(figsize=(10, 7))
bar_colors8 = ["#F44336" if v < 0 else "#2196F3" for v in rule_acc_df["r"]]
ax8a.barh(rule_acc_df["rule"], rule_acc_df["r"], color=bar_colors8, alpha=0.85)
ax8a.axvline(0, color="black", linewidth=0.8)
ax8a.set_xlabel("Pearson r (rule fail rate vs. mean accuracy across 15 cells)")
ax8a.set_title(
    "Fig 8a: Per-Rule Correlation with Accuracy\n"
    "Red = higher fail rate → lower accuracy; Blue = positive association"
)
ax8a.tick_params(axis="y", labelsize=8)
plt.tight_layout()
fig8a.savefig(FIGURES_DIR / "fig8a_per_rule_accuracy_corr.png", dpi=150)
plt.close(fig8a)
print("  Saved fig8a_per_rule_accuracy_corr.png")

# =============================================================================
# SECTION 9: Per-Question Writing Flaw × Accuracy
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 9: PER-QUESTION WRITING FLAW × ACCURACY")
print("=" * 70)

# 1. Build JSONL lookup: (dataset, config, question) → {flaw_value, rule_* booleans}
import ast as _ast
print("  Building JSONL lookup ...")
jl_lookup = {}
with open(FLAW_JSONL) as _f:
    for _line in _f:
        _obj = json.loads(_line)
        _wf = _obj["writing_flaw"]
        _answer_arr = _ast.literal_eval(_wf["answer"])
        _rule_bools = {f"rule_{r}": (v == "pass") for r, v in zip(RULES_ORDER, _answer_arr)}
        _key = (_obj["dataset"], _obj["config"], _obj["question"])
        jl_lookup[_key] = {"flaw_value": _wf["value"], **_rule_bools}

print(f"  JSONL lookup: {len(jl_lookup):,} entries")

# 2. Load Arrow files for full_question mode, all eval_models
print("  Loading Arrow files (full_question mode) ...")
pq_rows = []
_missing_paths = 0
for _em in eval_models:
    for _ds in DATASET_ORDER:
        for _cfg in CONFIG_ORDER:
            _path = (
                RESULTS_DIR / GENERATOR / _em / "full_question" / _ds / _cfg / "rows"
            )
            if not _path.exists():
                _missing_paths += 1
                continue
            _arrow_ds = load_from_disk(str(_path))
            for _row in _arrow_ds:
                _key = (_ds, _cfg, _row["question"])
                _jl = jl_lookup.get(_key)
                if _jl is None:
                    continue
                pq_rows.append(
                    {
                        "dataset": _ds,
                        "config": _cfg,
                        "eval_model": _em,
                        "is_correct": int(_row["is_correct"]),
                        "flaw_value": _jl["flaw_value"],
                        **{r: _jl[r] for r in [f"rule_{x}" for x in RULES_ORDER]},
                    }
                )

print(f"  Missing paths: {_missing_paths}")
pq_df = pd.DataFrame(pq_rows)
print(f"  Per-question DataFrame: {len(pq_df):,} rows")

# Overall per-question r(flaw_value, is_correct)
pq_r, pq_p = stats.pearsonr(pq_df["flaw_value"], pq_df["is_correct"])
print(f"\n  Per-question Pearson r(flaw_value, is_correct): r={pq_r:.4f}, p={pq_p:.4f}")
print(f"  (Aggregate r across 15 cells = {r_val:.3f}; per-question r should be weaker)")

# 3. Per-rule accuracy gap: mean(is_correct | rule_failed) − mean(is_correct | rule_passed)
gap_rows = []
for _rule in RULES_ORDER:
    _col = f"rule_{_rule}"
    _passed = pq_df[pq_df[_col] == True]["is_correct"].mean()
    _failed = pq_df[pq_df[_col] == False]["is_correct"].mean()
    _gap = _failed - _passed  # negative = failing rule hurts
    gap_rows.append({"rule": _rule, "acc_passed": _passed, "acc_failed": _failed, "gap": _gap})

gap_df = pd.DataFrame(gap_rows).sort_values("gap").reset_index(drop=True)

print("\n  Top 5 rules by magnitude of accuracy gap (|failed_acc − passed_acc|):")
top5_gap = gap_df.reindex(gap_df["gap"].abs().sort_values(ascending=False).index).head(5)
for _, row in top5_gap.iterrows():
    print(
        f"    {row['rule']:<30} gap={row['gap']:+.4f} "
        f"(passed={row['acc_passed']:.3f}, failed={row['acc_failed']:.3f})"
    )

# ── Fig 9a: Horizontal bar — accuracy gap per rule ────────────────────────────
fig9a, ax9a = plt.subplots(figsize=(10, 7))
bar_colors9 = ["#F44336" if v < 0 else "#2196F3" for v in gap_df["gap"]]
ax9a.barh(gap_df["rule"], gap_df["gap"], color=bar_colors9, alpha=0.85)
ax9a.axvline(0, color="black", linewidth=0.8)
ax9a.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax9a.set_xlabel("Accuracy gap (rule_failed − rule_passed)")
ax9a.set_title(
    "Fig 9a: Per-Question Accuracy Gap by Writing Rule\n"
    "Red = failing rule → lower accuracy; Blue = failing rule → higher accuracy"
)
ax9a.tick_params(axis="y", labelsize=8)
plt.tight_layout()
fig9a.savefig(FIGURES_DIR / "fig9a_per_question_rule_accuracy_gap.png", dpi=150)
plt.close(fig9a)
print("  Saved fig9a_per_question_rule_accuracy_gap.png")

# =============================================================================
# SECTION 10: Nonfunctional Distractor Rate
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 10: NONFUNCTIONAL DISTRACTOR RATE")
print("=" * 70)

# Reuse pq_df's Arrow data but need model_answer and options per question.
# Load Arrow again collecting per-question answer sets across eval_models.
print("  Loading Arrow files for distractor analysis ...")
# Build: (dataset, config, question_idx) → {gold_letter, options, model_answers: set}
_distractor_map = {}  # key → {"gold": str, "options": list, "model_answers": set}
_d_missing = 0
for _em in eval_models:
    for _ds in DATASET_ORDER:
        for _cfg in CONFIG_ORDER:
            _path = (
                RESULTS_DIR / GENERATOR / _em / "full_question" / _ds / _cfg / "rows"
            )
            if not _path.exists():
                _d_missing += 1
                continue
            _arrow_ds = load_from_disk(str(_path))
            for _row in _arrow_ds:
                _k = (_ds, _cfg, _row["question_idx"])
                if _k not in _distractor_map:
                    _distractor_map[_k] = {
                        "gold": _row["gold_answer_letter"],
                        "options": _row["options_randomized"],
                        "model_answers": set(),
                    }
                if _row["model_answer"]:
                    _distractor_map[_k]["model_answers"].add(_row["model_answer"])

print(f"  Built distractor map: {len(_distractor_map):,} unique (dataset, config, question_idx) cells")

# Compute nonfunctional rate per (dataset, config)
_nf_rows = []
for (_ds, _cfg), _group in pd.DataFrame(
    [
        {
            "dataset": k[0], "config": k[1],
            "gold": v["gold"], "options": v["options"],
            "model_answers": v["model_answers"],
        }
        for k, v in _distractor_map.items()
    ]
).groupby(["dataset", "config"]):
    _total_distractors = 0
    _nonfunctional = 0
    for _, _qrow in _group.iterrows():
        _gold = _qrow["gold"]
        _opts = _qrow["options"]
        _chosen = _qrow["model_answers"]
        # All option letters: A, B, C, ... up to len(opts)
        _letters = [chr(ord("A") + i) for i in range(len(_opts))]
        _wrong = [l for l in _letters if l != _gold]
        _total_distractors += len(_wrong)
        _nonfunctional += sum(1 for l in _wrong if l not in _chosen)
    _rate = _nonfunctional / _total_distractors if _total_distractors > 0 else float("nan")
    _nf_rows.append({
        "dataset": _ds, "config": _cfg,
        "total_distractors": _total_distractors,
        "nonfunctional_count": _nonfunctional,
        "nonfunctional_rate": _rate,
    })

nf_df = pd.DataFrame(_nf_rows)
nf_df["config"] = pd.Categorical(nf_df["config"], categories=CONFIG_ORDER, ordered=True)
nf_df.sort_values(["dataset", "config"], inplace=True)

print("\n  Nonfunctional distractor rates:")
print(f"  {'dataset':<15} {'config':<22} {'total_distractors':>18} {'nonfunctional':>14} {'rate':>7}")
print("  " + "-" * 78)
for _, row in nf_df.iterrows():
    print(
        f"  {row['dataset']:<15} {row['config']:<22} "
        f"{row['total_distractors']:>18,} {row['nonfunctional_count']:>14,} "
        f"{row['nonfunctional_rate']:>6.1%}"
    )

# Highlight human_from_scratch vs model_from_scratch gap
print("\n  human_from_scratch vs model_from_scratch nonfunctional rate gap:")
for _ds in DATASET_ORDER:
    _hum = nf_df[(nf_df.dataset == _ds) & (nf_df.config == "human_from_scratch")]["nonfunctional_rate"]
    _mdl = nf_df[(nf_df.dataset == _ds) & (nf_df.config == "model_from_scratch")]["nonfunctional_rate"]
    if len(_hum) and len(_mdl):
        _h, _m = float(_hum.iloc[0]), float(_mdl.iloc[0])
        print(f"    {_ds}: human={_h:.1%}, model={_m:.1%}, gap={_m - _h:+.1%}")

# ── Fig 10a: Grouped bar nonfunctional rate ────────────────────────────────────
nf_df["label"] = nf_df["dataset"].map(lambda d: DATASET_LABELS[d])
fig10a, ax10a = plt.subplots(figsize=(9, 5))
_n_configs = len(CONFIG_ORDER)
_x10 = np.arange(len(DATASET_ORDER))
_w10 = 0.14
_offsets10 = np.linspace(-(_n_configs - 1) / 2, (_n_configs - 1) / 2, _n_configs) * _w10

for i, _cfg in enumerate(CONFIG_ORDER):
    _vals = []
    for _ds in DATASET_ORDER:
        _sub = nf_df[(nf_df.dataset == _ds) & (nf_df.config == _cfg)]
        _vals.append(float(_sub["nonfunctional_rate"].iloc[0]) if len(_sub) else 0)
    ax10a.bar(
        _x10 + _offsets10[i], _vals, _w10 * 0.9,
        label=CONFIG_LABELS[_cfg].replace("\n", " "),
        color=PALETTE[_cfg], alpha=0.85,
    )

ax10a.set_xticks(_x10)
ax10a.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
ax10a.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax10a.set_ylabel("Nonfunctional distractor rate")
ax10a.legend(fontsize=7, ncol=2)
ax10a.set_title(
    "Fig 10a: Nonfunctional Distractor Rate per Config × Dataset\n"
    "(Wrong options never selected by any of 3 eval models)"
)
plt.tight_layout()
fig10a.savefig(FIGURES_DIR / "fig10a_nonfunctional_distractor_rate.png", dpi=150)
plt.close(fig10a)
print("  Saved fig10a_nonfunctional_distractor_rate.png")

# =============================================================================
# SECTION 11: Normalised Accuracy — (acc − 1/k) / (1 − 1/k)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 11: Normalised Accuracy (above-chance fraction retained)")
print("=" * 70)

K_MAP = {
    "human_from_scratch": 4, "model_from_scratch": 4,
    "augment_human": 10, "augment_model": 10, "augment_ablation": 10,
}

fq_all = acc_df[acc_df["mode"] == "full_question"].copy()
fq_all["k"] = fq_all["config"].map(K_MAP)
fq_all["norm_acc"] = (fq_all["accuracy"] - 1 / fq_all["k"]) / (1 - 1 / fq_all["k"])

norm_agg = (
    fq_all.groupby(["dataset", "config"])[["accuracy", "norm_acc"]]
    .mean()
    .reset_index()
)

print(f"\n{'Dataset':<14} {'Config':<22} {'Raw acc':>9} {'Norm acc':>10} {'k':>4}")
print("-" * 64)
for ds in DATASET_ORDER:
    for cfg in CONFIG_ORDER:
        sub = norm_agg[(norm_agg["dataset"] == ds) & (norm_agg["config"] == cfg)]
        if sub.empty:
            continue
        k = K_MAP[cfg]
        print(
            f"{DATASET_LABELS[ds]:<14} {cfg:<22} "
            f"{sub['accuracy'].values[0]:>8.1%}  {sub['norm_acc'].values[0]:>9.1%}  {k:>4}"
        )
    print()

# ── Fig 11a: Side-by-side raw vs normalised accuracy ─────────────────────────
fig11a, axes11 = plt.subplots(1, 2, figsize=(15, 5), sharey=False)

x11 = np.arange(len(DATASET_ORDER))
n_cfgs = len(CONFIG_ORDER)
w11 = 0.15
offsets11 = np.linspace(-(n_cfgs - 1) / 2, (n_cfgs - 1) / 2, n_cfgs) * w11

for ax, col, title in zip(
    axes11,
    ["accuracy", "norm_acc"],
    ["Raw accuracy", "Normalised accuracy  (acc − 1/k) / (1 − 1/k)"],
):
    for i, cfg in enumerate(CONFIG_ORDER):
        vals = []
        for ds in DATASET_ORDER:
            sub = norm_agg[(norm_agg["dataset"] == ds) & (norm_agg["config"] == cfg)]
            vals.append(float(sub[col].values[0]) if len(sub) else 0)
        ax.bar(
            x11 + offsets11[i], vals, w11 * 0.9,
            label=CONFIG_LABELS[cfg].replace("\n", " "),
            color=PALETTE[cfg], alpha=0.85,
        )
    ax.set_xticks(x11)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)

plt.suptitle(
    "Fig 11a: Raw vs Normalised Accuracy by Config × Dataset\n"
    "Normalised = fraction of above-chance performance retained",
    fontsize=11,
)
plt.tight_layout()
fig11a.savefig(FIGURES_DIR / "fig11a_normalised_accuracy.png", dpi=150)
plt.close(fig11a)
print("  Saved fig11a_normalised_accuracy.png")

# ── Fig 11b: Delta normalised accuracy from human_from_scratch baseline ───────
norm_baseline = norm_agg[norm_agg["config"] == "human_from_scratch"].set_index("dataset")

delta11_rows = []
for _, row in norm_agg[norm_agg["config"] != "human_from_scratch"].iterrows():
    ds = row["dataset"]
    delta11_rows.append({
        "dataset": ds,
        "config": row["config"],
        "label": f"{DATASET_LABELS[ds]}\n{CONFIG_SHORT[row['config']]}",
        "raw_drop":  row["accuracy"]  - norm_baseline.loc[ds, "accuracy"],
        "norm_drop": row["norm_acc"]  - norm_baseline.loc[ds, "norm_acc"],
    })
delta11_df = pd.DataFrame(delta11_rows)
delta11_df["config"] = pd.Categorical(delta11_df["config"], categories=CONFIG_ORDER, ordered=True)
delta11_df.sort_values(["dataset", "config"], inplace=True)

fig11b, ax11b = plt.subplots(figsize=(13, 5))
x11b = np.arange(len(delta11_df))
w11b = 0.35
ax11b.bar(x11b - w11b / 2, delta11_df["raw_drop"],  w11b, label="Δ raw accuracy",        color="#1976D2", alpha=0.85)
ax11b.bar(x11b + w11b / 2, delta11_df["norm_drop"], w11b, label="Δ normalised accuracy", color="#FF9800", alpha=0.85)
ax11b.axhline(0, color="black", linewidth=0.7)
ax11b.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax11b.set_ylabel("Δ from human_from_scratch baseline")
ax11b.set_xticks(x11b)
ax11b.set_xticklabels(delta11_df["label"], fontsize=7, rotation=30, ha="right")
ax11b.legend(fontsize=9)
ax11b.set_title(
    "Fig 11b: Raw vs Normalised Accuracy Drop from human_from_scratch\n"
    "Gap between bars = portion of raw drop explained by floor shift (more choices → lower random baseline)"
)
plt.tight_layout()
fig11b.savefig(FIGURES_DIR / "fig11b_normalised_accuracy_delta.png", dpi=150)
plt.close(fig11b)
print("  Saved fig11b_normalised_accuracy_delta.png")

# =============================================================================
# SECTION 11: Key Findings Summary
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 11: KEY FINDINGS SUMMARY")
print("=" * 70)

# Quality hierarchy
arc_hum_p2 = float(summary_df[(summary_df.dataset=="arc_challenge") & (summary_df.config=="human_from_scratch")]["p_ge2_mean"].iloc[0])
arc_mdl_p2 = float(summary_df[(summary_df.dataset=="arc_challenge") & (summary_df.config=="model_from_scratch")]["p_ge2_mean"].iloc[0])
mmlu_hum_p2 = float(summary_df[(summary_df.dataset=="mmlu_pro") & (summary_df.config=="human_from_scratch")]["p_ge2_mean"].iloc[0])
gpqa_hum_p2 = float(summary_df[(summary_df.dataset=="gpqa") & (summary_df.config=="human_from_scratch")]["p_ge2_mean"].iloc[0])

print("""
FINDING 1 — QUALITY HIERARCHY
  ARC-Challenge shows the clearest human > model quality gap in 4-choice setting.
  MMLU-Pro and GPQA are uniformly lower quality even for human_from_scratch,
  suggesting the original benchmark questions themselves carry more inherent flaws.""")
print(f"  ARC human_from_scratch P(≥2 flaws): {arc_hum_p2:.1%}")
print(f"  ARC model_from_scratch P(≥2 flaws): {arc_mdl_p2:.1%}")
print(f"  MMLU-Pro human_from_scratch P(≥2 flaws): {mmlu_hum_p2:.1%}")
print(f"  GPQA human_from_scratch P(≥2 flaws): {gpqa_hum_p2:.1%}")

print(f"""
FINDING 2 — 4-CHOICE VS 10-CHOICE IS THE DOMINANT VALIDITY DRIVER
  ARC quality drop (4→10 choices, human): {arc_hum - arc_aug:.3f} fraction of rules
  vs.
  ARC human vs model within 4-choice: {arc_hum - arc_mdl:.3f} fraction of rules
  The augmentation to 10 choices costs more in writing quality than switching
  from human to model distractors within the same choice count.""")

print(f"""
FINDING 3 — TOP RULE VIOLATIONS BY MODEL DISTRACTORS
  Rules most impacted by switching human→model distractors:""")
for rule, delta in sens_avg.sort_values(ascending=False).head(5).items():
    print(f"    {rule}: +{delta:.3f}")

print(f"""
FINDING 4 — VALIDITY vs. ACCURACY CORRELATION
  Pearson r = {r_val:.3f} (p = {p_val:.4f}) across 15 (dataset × config) cells.
  Writing quality and accuracy are {corr_desc} ({sig_desc}).
  This {'suggests' if p_val < 0.05 else 'does not provide strong evidence'} that
  writing quality is a meaningful predictor of model performance.""")

print("""
FINDING 5 — RESEARCH IMPLICATIONS
  The pipeline generates more questions (10 choices vs 4), but this comes at a
  validity cost. The key tension: harder questions (lower accuracy) vs. less valid
  questions (more writing flaws). The correlation analysis above indicates whether
  accuracy drops are explained by writing quality degradation or genuine difficulty.
  If r < 0 and significant: models struggle because questions are poorly written.
  If r > 0 or non-significant: accuracy drops likely reflect genuine difficulty,
  not writing flaws — supporting the pipeline's value as a harder benchmark.
""")

print("=" * 70)
print(f"All figures saved to: {FIGURES_DIR.resolve()}")
print(f"Total figures: {len(list(FIGURES_DIR.glob('*.png')))}")
print("Done.")
