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
]
