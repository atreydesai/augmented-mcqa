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
    load_results_file,
    load_summary_file,
    load_3H_plus_M_results,
    load_human_only_results,
    load_model_only_results,
    plot_rq1_combined,
    plot_rq2_human_distractors,
    plot_rq3_model_distractors,
    plot_branching_comparison,
    plot_difficulty_comparison,
    plot_all_rq,
    DATASET_TYPE_STYLES,
    DISTRACTOR_SOURCE_LABELS,
)

from .category_analysis import (
    compute_accuracy_by_category,
    compute_accuracy_by_dataset_type,
    plot_category_breakdown,
    plot_category_heatmap,
    plot_dataset_type_breakdown,
    plot_gpqa_by_discipline,
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
    "load_results_file",
    "load_summary_file",
    "load_3H_plus_M_results",
    "load_human_only_results",
    "load_model_only_results",
    # Visualize - RQ plots
    "plot_rq1_combined",
    "plot_rq2_human_distractors",
    "plot_rq3_model_distractors",
    "plot_branching_comparison",
    "plot_difficulty_comparison",
    "plot_all_rq",
    "DATASET_TYPE_STYLES",
    "DISTRACTOR_SOURCE_LABELS",
    # Category analysis
    "compute_accuracy_by_category",
    "compute_accuracy_by_dataset_type",
    "plot_category_breakdown",
    "plot_category_heatmap",
    "plot_dataset_type_breakdown",
    "plot_gpqa_by_discipline",
    "generate_category_report",
    "MMLU_PRO_CATEGORY_GROUPS",
]
