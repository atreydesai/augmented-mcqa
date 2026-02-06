# Augmented MCQA

Research framework for studying model-generated distractors in multiple choice question answering.

## Setup

```bash
conda activate qgqa
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys
```

---

## Step 1: Download Datasets

Download all required datasets from HuggingFace:

```bash
# Download all datasets (run once)
python -c "
from data.downloader import download_all_datasets
download_all_datasets()
"
```

This downloads:
- **MMLU** → `datasets/mmlu/`
- **MMLU-Pro** → `datasets/mmlu_pro/`
- **ARC** (Easy + Challenge) → `datasets/arc/`
- **SuperGPQA** → `datasets/supergpqa/`

---

## Step 2: Process MMLU-Pro

Sort MMLU-Pro distractors into human vs synthetic:

```bash
python -c "
from data.sorter import process_mmlu_pro
process_mmlu_pro()
"
```

Output: `datasets/mmlu_pro_sorted/` with columns:
- `cond_human_q_a` - Original MMLU distractors
- `cond_model_q_a` - MMLU-Pro synthetic distractors

---

## Step 3: Generate Synthetic Distractors

Generate new distractors using different models. **Each run uses a unique output file** to allow concurrent execution.

```bash
# List available models
python scripts/generate_distractors.py --list-models

# Generate with different models (can run concurrently)
python scripts/generate_distractors.py \
    --input datasets/mmlu_pro_sorted/test.json \
    --output datasets/augmented/mmlu_pro_gpt4.json \
    --model gpt-4.1 \
    --mode from_scratch

python scripts/generate_distractors.py \
    --input datasets/mmlu_pro_sorted/test.json \
    --output datasets/augmented/mmlu_pro_claude.json \
    --model claude-sonnet-4-5 \
    --mode from_scratch

python scripts/generate_distractors.py \
    --input datasets/mmlu_pro_sorted/test.json \
    --output datasets/augmented/mmlu_pro_gemini.json \
    --model gemini-3-flash-preview \
    --mode from_scratch
```

### Generation Modes

| Mode | Output Column | Description |
|------|---------------|-------------|
| `from_scratch` | `cond_model_q_a` | Generate from Q+A only |
| `conditioned_human` | `cond_model_q_a_dhuman` | Conditioned on human distractors |
| `conditioned_synthetic` | `cond_model_q_a_dmodel` | Conditioned on model distractors |

---

## Step 4: Create Filtered Subsets

Create evaluation subsets with specific distractor configurations:

```bash
python -c "
from data.filter import create_standard_subsets

# Creates subsets like 3H0M (3 human, 0 model), 0H3M, 3H3M
create_standard_subsets(
    input_path='datasets/augmented/mmlu_pro_gpt4.json',
    output_dir='datasets/filtered/gpt4/'
)
"
```

---

## Step 5: Run Experiments

Run evaluation experiments. **Use unique output directories** for concurrent runs.

```bash
# Single experiment
python scripts/run_experiment.py \
    --dataset datasets/filtered/gpt4/3H3M.json \
    --model gpt-4.1 \
    --output-dir results/gpt4_eval_3H3M \
    --limit 100

# Batch experiments with different models (concurrent-safe)
for MODEL in gpt-4.1 claude-sonnet-4-5 gemini-3-flash-preview; do
    for CONFIG in 3H0M 0H3M 3H3M; do
        python scripts/run_experiment.py \
            --dataset datasets/filtered/gpt4/$CONFIG.json \
            --model $MODEL \
            --output-dir results/${MODEL}_${CONFIG} &
    done
done
wait
```

### Config via JSON

```bash
python scripts/run_experiment.py --config experiments/configs/rq1.json
```

---

## Step 6: Analyze Results

### Compute Behavioral Signatures

```bash
python -c "
from analysis import analyze_experiment

results = analyze_experiment('results/gpt4_eval_3H3M/')
print(f'Accuracy: {results[\"accuracy\"]:.2%}')
print(f'Gold rate: {results[\"gold_rate\"]:.2%}')
"
```

### Category/Topic Breakdown

```bash
python -c "
from analysis import plot_category_breakdown, generate_category_report

plot_category_breakdown(
    'results/gpt4_eval_3H3M/results.json',
    output_path='results/plots/category_gpt4_3H3M.png'
)

generate_category_report(
    'results/gpt4_eval_3H3M/results.json',
    output_dir='results/reports/'
)
"
```

---

## Step 7: Generate Visualizations

### RQ Plots

```bash
python -c "
from analysis import plot_all_rq

plot_all_rq(
    base_dir='results/',
    output_dir='results/plots/'
)
"
```

### Branching Analysis (1H vs 2H vs 3H)

```bash
python -c "
from analysis import plot_human_distractor_branching, plot_human_benefit_comparison

plot_human_distractor_branching(
    base_dir='results/',
    output_dir='results/plots/'
)

plot_human_benefit_comparison(
    base_dir='results/',
    output_dir='results/plots/'
)
"
```

### Difficulty Scaling

```bash
python -c "
from analysis import plot_all_difficulty

plot_all_difficulty(
    results_dir='results/difficulty_scaling/',
    output_dir='results/plots/difficulty/'
)
"
```

---

## Running Concurrently (Naming Conventions)

To run jobs in parallel without conflicts, use this naming pattern:

```
results/{evaluator_model}_{generator_model}_{dataset}_{config}/
```

**Examples:**
```
results/gpt4_gpt4_mmlupro_3H3M/       # GPT-4 evaluating GPT-4 generated
results/claude_gemini_mmlupro_3H0M/   # Claude evaluating Gemini generated
results/gpt4_human_arc_easy/          # GPT-4 on ARC with human distractors
```

**Concurrent batch script:**
```bash
#!/bin/bash
# run_all.sh

EVALUATORS=("gpt-4.1" "claude-sonnet-4-5")
GENERATORS=("gpt4" "claude" "gemini")
CONFIGS=("3H0M" "0H3M" "3H3M")

for EVAL in "${EVALUATORS[@]}"; do
    for GEN in "${GENERATORS[@]}"; do
        for CFG in "${CONFIGS[@]}"; do
            OUTPUT="results/${EVAL//\//_}_${GEN}_mmlupro_${CFG}"
            python scripts/run_experiment.py \
                --dataset "datasets/filtered/${GEN}/${CFG}.json" \
                --model "$EVAL" \
                --output-dir "$OUTPUT" &
        done
    done
done
wait
echo "All experiments complete"
```

---

## Project Structure

```
augmented-mcqa/
├── config/          # Configuration and settings
├── data/            # Dataset downloading, processing, adapters
│   ├── downloader.py      # Download from HuggingFace
│   ├── sorter.py          # Sort MMLU-Pro distractors
│   ├── augmentor.py       # Generate synthetic distractors
│   ├── filter.py          # Create filtered subsets
│   └── adapter.py         # Unified data access
├── models/          # Model clients (single source of truth)
│   ├── openai_client.py
│   ├── anthropic_client.py
│   ├── gemini_client.py
│   ├── deepseek_client.py
│   └── local_client.py    # vLLM for local models
├── experiments/     # Experiment config and runners
├── evaluation/      # Answer extraction, scoring
├── analysis/        # Analysis and visualization
│   ├── analyzer.py           # Behavioral signatures
│   ├── visualize.py          # RQ plots
│   ├── branching_analysis.py # 1H/2H/3H comparison
│   └── category_analysis.py  # Topic breakdown
├── scripts/         # CLI entry points
│   ├── run_experiment.py
│   └── generate_distractors.py
├── datasets/        # Downloaded/generated data
└── results/         # Experiment outputs
```

---

## Distractor Naming Convention

| Type | Column Name | Source |
|------|-------------|--------|
| Human distractors | `cond_human_q_a` | Original MMLU |
| Model distractors | `cond_model_q_a` | Generated from Q+A |
| Conditioned on human | `cond_model_q_a_dhuman` | Generated given human distractors |
| Conditioned on model | `cond_model_q_a_dmodel` | Generated given model distractors |

---

## Available Models

```bash
python scripts/generate_distractors.py --list-models
```

| Provider | Models |
|----------|--------|
| OpenAI | `gpt-4.1`, `gpt-5-mini`, `gpt-5.2` |
| Anthropic | `claude-opus-4-6`, `claude-sonnet-4-5`, `claude-haiku-4-5` |
| Google | `gemini-3-pro-preview`, `gemini-3-flash-preview`, `gemini-2.5-flash-lite` |
| DeepSeek | `deepseek-chat`, `deepseek-reasoner` |
| Local | `qwen3-8b` (vLLM) |

---

## Research Questions

- **RQ1**: Accuracy comparison across human vs model distractors
- **RQ2**: Effect of adding human distractors (1H vs 2H vs 3H)
- **RQ3**: Original MMLU-Pro vs recreated MMLU-Pro-Aug

---

## License

Private research repository.
