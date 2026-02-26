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
    SETTING_RANDOM_BASELINES,
    collect_final5_results,
    write_final5_summary_table,
    plot_final5_pairwise,
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
    # Final5 plotting
    "SETTING_RANDOM_BASELINES",
    "collect_final5_results",
    "write_final5_summary_table",
    "plot_final5_pairwise",
]
