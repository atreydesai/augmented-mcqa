#!/bin/bash

# Experiment Orchestration for Augmented MCQA
# This script runs all experiments for RQ1, RQ2, and RQ3.

# Configuration
MODELS=("gpt-4o")
DATASETS=("mmlu_pro")
LIMIT=5
MAX_PARALLEL=8
PYTHON_EXEC="python"
SCRIPT_PATH="scripts/run_experiment.py"
DATASET_BASE_DIR="/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/datasets/processed/unified_processed"

mkdir -p logs/experiments

# Function to run a single experiment
run_exp() {
    local name=$1
    local dataset=$2
    local model=$3
    local nh=$4
    local nm=$5
    local mtype=$6
    local choices_only=$7
    
    local cmd_args="--name $name --dataset $DATASET_BASE_DIR/$dataset --model $model --num-human $nh --num-model $nm --model-type $mtype --limit $LIMIT"
    if [ "$choices_only" = "true" ]; then
        cmd_args="$cmd_args --choices-only"
    fi
    
    echo "Running: $name ($nh H, $nm M, $mtype, choices_only=$choices_only)"
    $PYTHON_EXEC $SCRIPT_PATH $cmd_args > "logs/experiments/${name}.log" 2>&1 &
    
    # Wait if too many jobs
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 2
    done
}

# =============================================================================
# RQ1 & RQ3: Distractor Sources & Scaling
# =============================================================================

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        # 1. Human Only (3H)
        run_exp "RQ_human_only_${dataset}_${model}" "$dataset" "$model" 3 0 "cond_human_q_a" "false"
        
        # 2. Existing Synthetic (MMLU-Pro Only - 6M)
        if [ "$dataset" = "mmlu_pro" ]; then
            run_exp "RQ_existing_synthetic_${dataset}_${model}" "$dataset" "$model" 0 6 "cond_model_q_a" "false"
        fi
        
        # 3. New Synthetic (From Scratch - 6M)
        run_exp "RQ_scratch_synthetic_${dataset}_${model}" "$dataset" "$model" 0 6 "cond_model_q_a_scratch" "false"
        
        # 4. New Synthetic (Conditioned on Human - 3H+6M)
        run_exp "RQ_dhuman_synthetic_${dataset}_${model}" "$dataset" "$model" 3 6 "cond_model_q_a_dhuman" "false"
        
        # 5. New Synthetic (Conditioned on Model - 3H+6M)
        run_exp "RQ_dmodel_synthetic_${dataset}_${model}" "$dataset" "$model" 3 6 "cond_model_q_a_dmodel" "false"
        
        # Choices Only versions for key configs
        run_exp "RQ_human_only_choices_${dataset}_${model}" "$dataset" "$model" 3 0 "cond_human_q_a" "true"
        run_exp "RQ_dhuman_synthetic_choices_${dataset}_${model}" "$dataset" "$model" 3 6 "cond_model_q_a_dhuman" "true"
    done
done

# =============================================================================
# RQ2: Human Distractor Benefit (1, 2, 3 H)
# =============================================================================

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for nh in 1 2; do
            run_exp "RQ2_human_benefit_${nh}H_${dataset}_${model}" "$dataset" "$model" $nh 0 "cond_human_q_a" "false"
            run_exp "RQ2_human_benefit_${nh}H_3M_${dataset}_${model}" "$dataset" "$model" $nh 3 "cond_model_q_a_dhuman" "false"
        done
        # 3H is already covered in RQ1 above
    done
done

wait
echo "All experiments completed."
