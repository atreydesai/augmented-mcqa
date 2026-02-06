# Augmented MCQA

A clean, modular repository for Multiple Choice Question Answering research with synthetic distractor generation and behavioral signature analysis.

## Quick Start

```bash
# Setup environment
conda activate qgqa
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Run a proof-of-concept experiment (25 examples)
python scripts/run_experiment.py \
    --model gpt-4.1-2025-04-14 \
    --num_human 3 \
    --num_model 3 \
    --limit 25
```

## Project Structure

```
augmented-mcqa/
├── config/          # Central configuration and settings
├── data/            # Dataset downloading, processing, and adapters
├── models/          # Model API clients (OpenAI, Anthropic, Gemini, local)
├── experiments/     # Experiment configuration and runners
├── evaluation/      # Evaluation logic, answer extraction, probabilities
├── analysis/        # Behavioral signatures, visualization
├── scripts/         # CLI entry points
├── datasets/        # Downloaded/generated datasets
└── results/         # Experiment outputs
```

## Distractor Naming Convention

| Dataset | Column Name |
|---------|-------------|
| Human distractors (from MMLU) | `cond_human_q_a` |
| Model distractors (conditioned on q+a) | `cond_model_q_a` |
| Model distractors (conditioned on q+a+human) | `cond_model_q_a_dhuman` |
| Model distractors (conditioned on q+a+model) | `cond_model_q_a_dmodel` |

## Research Questions

- **RQ1**: Accuracy on human vs 2 types of model-generated distractors
- **RQ2**: Propagation of human distractor signal
- **RQ3**: Original MMLU-Pro vs Recreated MMLU-Pro

## License

Private research repository.
