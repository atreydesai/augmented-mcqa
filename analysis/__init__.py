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
    create_behavioral_bar_chart,
    create_accuracy_comparison,
    create_category_heatmap,
    plot_results_summary,
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
    # Visualize
    "create_behavioral_bar_chart",
    "create_accuracy_comparison",
    "create_category_heatmap",
    "plot_results_summary",
]
