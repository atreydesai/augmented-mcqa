import sys
from pathlib import Path
sys.path.append("/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa")
import pandas as pd
from analysis.irt import (
    final_ablation_quality_frame,
    final_grouped_quality_frame,
    plot_final_ablation_quality,
    plot_final_grouped_quality,
    plot_irt_quality,
    write_csv,
)

tables = Path("/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/results/augmented_mcqa_irt/tables")
figures = Path("/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/results/augmented_mcqa_irt/figures/final_figures")
figures.mkdir(parents=True, exist_ok=True)
figures_base = Path("/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/results/augmented_mcqa_irt/figures")

combined_quality = pd.read_csv(tables / "question_quality_by_dataset_setting.csv")
items = pd.read_csv(tables / "instantiated_item_parameters.csv")
validity = pd.read_csv(tables / "benchmarker_validity_by_dataset_setting.csv")
final_quality = final_grouped_quality_frame(items, validity)
ablation_quality = final_ablation_quality_frame(items, validity)
write_csv(final_quality, tables / "final_grouped_question_quality.csv")
write_csv(ablation_quality, tables / "final_ablation_question_quality.csv")

# Only plot the final ones
plot_final_grouped_quality(final_quality, figures / "question_quality_grouped.png")
plot_final_ablation_quality(ablation_quality, figures / "ablation_quality.png")
plot_irt_quality(combined_quality, figures_base / "question_quality_all_settings.png")
print("Done")
