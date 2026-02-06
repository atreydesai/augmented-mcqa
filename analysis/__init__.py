"""Analysis module for Augmented MCQA."""

from .analyzer import (
    load_results,
    compute_behavioral_signature,
    compute_gold_rate,
    compute_hierarchical_signature,
    analyze_experiment,
    compare_experiments,
    format_signature_table,
)

from .visualize import (
    load_summary_file,
    load_3H_plus_M_results,
    load_human_only_results,
    load_model_only_results,
    plot_rq1_combined,
    plot_rq2_human_distractors,
    plot_rq3_model_distractors,
    plot_all_rq,
)

from .difficulty_visualize import (
    load_difficulty_results,
    plot_arc_comparison,
    plot_supergpqa_by_difficulty,
    plot_difficulty_combined,
    plot_distractor_effect_scaling,
    plot_all_difficulty,
    DIFFICULTY_COLORS,
    DATASET_COLORS,
)

from .branching_analysis import (
    load_branching_results,
    plot_human_distractor_branching,
    plot_human_benefit_comparison,
    HUMAN_COLORS,
)

from .category_analysis import (
    compute_accuracy_by_category,
    plot_category_breakdown,
    plot_category_heatmap,
    plot_supergpqa_by_discipline,
    generate_category_report,
    MMLU_PRO_CATEGORY_GROUPS,
)


__all__ = [
    # Analyzer
    "load_results",
    "compute_behavioral_signature",
    "compute_gold_rate",
    "compute_hierarchical_signature",
    "analyze_experiment",
    "compare_experiments",
    "format_signature_table",
    # Visualize - Data loaders
    "load_summary_file",
    "load_3H_plus_M_results",
    "load_human_only_results",
    "load_model_only_results",
    # Visualize - RQ plots
    "plot_rq1_combined",
    "plot_rq2_human_distractors",
    "plot_rq3_model_distractors",
    "plot_all_rq",
    # Difficulty visualization
    "load_difficulty_results",
    "plot_arc_comparison",
    "plot_supergpqa_by_difficulty",
    "plot_difficulty_combined",
    "plot_distractor_effect_scaling",
    "plot_all_difficulty",
    "DIFFICULTY_COLORS",
    "DATASET_COLORS",
    # Branching analysis
    "load_branching_results",
    "plot_human_distractor_branching",
    "plot_human_benefit_comparison",
    "HUMAN_COLORS",
    # Category analysis
    "compute_accuracy_by_category",
    "plot_category_breakdown",
    "plot_category_heatmap",
    "plot_supergpqa_by_discipline",
    "generate_category_report",
    "MMLU_PRO_CATEGORY_GROUPS",
]

