#!/bin/bash
#SBATCH --job-name=aug_gemini
#SBATCH --output=logs/augmentation/slurm_%j.out
#SBATCH --error=logs/augmentation/slurm_%j.err
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=medium
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

conda deactivate 2>/dev/null
cd /fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa
source .venv/bin/activate
mkdir -p logs/augmentation

python scripts/generate_distractors.py \
  --input datasets/processed/unified_processed \
  --model gemini-3-flash-preview \
  --limit 750 \
  --save-interval 50 \
  --output datasets/finished_sets/gemini-3-flash \
  --parallel
