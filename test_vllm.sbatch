#!/bin/bash
#SBATCH --job-name=test_vllm
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --time=00:10:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --output=test_vllm_%j.out
#SBATCH --error=test_vllm_%j.err

export MODEL_CACHE_DIR=/fs/clip-scratch/adesai10/hub
export HF_HOME=/fs/clip-scratch/adesai10/hub
./.venv/bin/vllm serve cyankiwi/Qwen3-30B-A3B-Thinking-2507-AWQ-8bit --host 0.0.0.0 --api-key inspectai --enforce-eager --max-model-len 32768 --port 30912
