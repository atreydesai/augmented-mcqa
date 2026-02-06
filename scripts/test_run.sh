#!/bin/bash
# test_run.sh - End-to-end test of the full pipeline
# Tests 2 generator models on all modes with a small sample

set -e

cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"

# Settings
LIMIT=5  # Small sample for testing
DATASETS_DIR="datasets"
RESULTS_DIR="results/test_run"

# Two test generators
GENERATORS=("gpt-4.1" "claude-sonnet-4-5")

# All generation modes
MODES=("from_scratch" "conditioned_human" "conditioned_synthetic")

echo ""
echo "=================================================="
echo "TEST RUN: Full Pipeline Test"
echo "=================================================="
echo "Generators: ${GENERATORS[*]}"
echo "Modes: ${MODES[*]}"
echo "Limit: $LIMIT entries per test"
echo ""

# Step 1: Verify datasets exist
echo "Step 1: Checking processed datasets..."
for ds in mmlu_pro_sorted arc_processed supergpqa_processed; do
    if [ ! -d "$DATASETS_DIR/$ds" ]; then
        echo "ERROR: $DATASETS_DIR/$ds not found. Run: python scripts/process_all.py"
        exit 1
    fi
done
echo "✅ All processed datasets found"

# Step 2: Test generation with each model and mode
echo ""
echo "Step 2: Testing distractor generation..."

mkdir -p "$DATASETS_DIR/test_augmented"

for GEN in "${GENERATORS[@]}"; do
    GEN_SHORT="${GEN//\//_}"  # Replace / with _ for filenames
    
    for MODE in "${MODES[@]}"; do
        OUTPUT="$DATASETS_DIR/test_augmented/mmlu_pro_${GEN_SHORT}_${MODE}.json"
        
        echo "  Generating: $GEN ($MODE) -> $OUTPUT"
        
        # Check if MMLU-Pro sorted has the right format (need JSON file)
        INPUT="$DATASETS_DIR/mmlu_pro_sorted/test.json"
        if [ ! -f "$INPUT" ]; then
            # Try to export from HF dataset
            python -c "
from datasets import load_from_disk
import json
ds = load_from_disk('$DATASETS_DIR/mmlu_pro_sorted')
with open('$INPUT', 'w') as f:
    json.dump(list(ds['test']), f)
print('Exported test split to JSON')
"
        fi
        
        python scripts/generate_distractors.py \
            --input "$INPUT" \
            --output "$OUTPUT" \
            --model "$GEN" \
            --mode "$MODE" \
            --limit "$LIMIT" \
            --dry-run  # Remove this for actual generation
        
        echo "    ✅ Dry run complete"
    done
done

# Step 3: Test filtering (create subsets)
echo ""
echo "Step 3: Testing subset creation..."

# For actual run, use the generated files
# For dry run, just verify the filter module works
python -c "
from data.filter import create_standard_subsets
print('Filter module loaded successfully')
print('Available configs: 3H0M, 0H3M, 3H3M, 1H0M, 2H0M, etc.')
"
echo "✅ Filter module ready"

# Step 4: Test experiment runner
echo ""
echo "Step 4: Testing experiment runner..."
python -c "
from experiments.runner import ExperimentRunner
from experiments.config import ExperimentConfig
print('Experiment modules loaded successfully')
"
echo "✅ Experiment modules ready"

# Step 5: Test analysis modules
echo ""
echo "Step 5: Testing analysis modules..."
python -c "
from analysis import (
    compute_behavioral_signature,
    plot_all_rq,
    plot_human_distractor_branching,
    plot_category_breakdown
)
print('Analysis modules loaded successfully')
"
echo "✅ Analysis modules ready"

echo ""
echo "=================================================="
echo "TEST RUN COMPLETE"
echo "=================================================="
echo ""
echo "To run actual generation (remove --dry-run):"
echo "  python scripts/generate_distractors.py \\"
echo "      --input datasets/mmlu_pro_sorted/test.json \\"
echo "      --output datasets/augmented/mmlu_pro_gpt4_scratch.json \\"
echo "      --model gpt-4.1 \\"
echo "      --mode from_scratch \\"
echo "      --limit 100"
echo ""
echo "To run experiments:"
echo "  python scripts/run_experiment.py \\"
echo "      --dataset datasets/filtered/gpt4/3H3M.json \\"
echo "      --model gpt-4.1 \\"
echo "      --output-dir results/gpt4_gpt4_3H3M"
