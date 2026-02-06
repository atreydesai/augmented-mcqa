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

## Quick Test

Run the full pipeline test (dry-run, no API calls):

```bash
bash scripts/test_run.sh
```

---

## Step 1: Download Datasets

```bash
python -c "from data.downloader import download_all_datasets; download_all_datasets()"
```

This downloads MMLU, MMLU-Pro, ARC (Easy/Challenge), and SuperGPQA.

---

## Step 2: Process All Datasets

```bash
python scripts/process_all.py
```

Or process individually:
```bash
python scripts/process_all.py --dataset mmlu_pro
python scripts/process_all.py --dataset arc
python scripts/process_all.py --dataset supergpqa
```

**Output:**
- `datasets/mmlu_pro_sorted/` - Human vs synthetic distractors separated
- `datasets/arc_processed/` - ARC-Easy and ARC-Challenge in unified format
- `datasets/supergpqa_processed/` - SuperGPQA filtered to 10-option questions

---

## Step 3: Generate Synthetic Distractors

```bash
# List available models
python scripts/generate_distractors.py --list-models

# Generate with different models (concurrent-safe naming)
python scripts/generate_distractors.py \
    --input datasets/mmlu_pro_sorted/test.json \
    --output datasets/augmented/mmlupro_gpt4_scratch.json \
    --model gpt-4.1 \
    --mode from_scratch

python scripts/generate_distractors.py \
    --input datasets/mmlu_pro_sorted/test.json \
    --output datasets/augmented/mmlupro_claude_scratch.json \
    --model claude-sonnet-4-5 \
    --mode from_scratch

python scripts/generate_distractors.py \
    --input datasets/mmlu_pro_sorted/test.json \
    --output datasets/augmented/mmlupro_gpt4_condhuman.json \
    --model gpt-4.1 \
    --mode conditioned_human
```

### Generation Modes

| Mode | Output Column | Description |
|------|---------------|-------------|
| `from_scratch` | `cond_model_q_a` | Generate from Q+A only |
| `conditioned_human` | `cond_model_q_a_dhuman` | Conditioned on human distractors |
| `conditioned_synthetic` | `cond_model_q_a_dmodel` | Conditioned on model distractors |

### Run Concurrent Generations

```bash
# All 3 modes × 2 models = 6 concurrent jobs
for MODEL in gpt-4.1 claude-sonnet-4-5; do
    SHORT="${MODEL//\//_}"
    for MODE in from_scratch conditioned_human conditioned_synthetic; do
        python scripts/generate_distractors.py \
            --input datasets/mmlu_pro_sorted/test.json \
            --output "datasets/augmented/mmlupro_${SHORT}_${MODE}.json" \
            --model "$MODEL" \
            --mode "$MODE" &
    done
done
wait
```

---

## Step 4: Create Filtered Subsets

Create evaluation subsets with specific distractor configurations:

```bash
python -c "
from data.filter import create_standard_subsets
create_standard_subsets(
    input_path='datasets/augmented/mmlupro_gpt4_scratch.json',
    output_dir='datasets/filtered/gpt4/'
)
"
```

Creates: `3H0M.json`, `0H3M.json`, `3H3M.json`, `1H0M.json`, `2H0M.json`, etc.

---

## Step 5: Run Experiments

```bash
python scripts/run_experiment.py \
    --dataset datasets/filtered/gpt4/3H3M.json \
    --model gpt-4.1 \
    --output-dir results/gpt4_eval_gpt4_gen_3H3M \
    --limit 100
```

### Concurrent Experiments

```bash
# Evaluator × Generator × Config = unique output dir
EVALUATORS=("gpt-4.1" "claude-sonnet-4-5")
GENERATORS=("gpt4" "claude")
CONFIGS=("3H0M" "0H3M" "3H3M")

for EVAL in "${EVALUATORS[@]}"; do
    EVAL_SHORT="${EVAL//\//_}"
    for GEN in "${GENERATORS[@]}"; do
        for CFG in "${CONFIGS[@]}"; do
            python scripts/run_experiment.py \
                --dataset "datasets/filtered/${GEN}/${CFG}.json" \
                --model "$EVAL" \
                --output-dir "results/${EVAL_SHORT}_${GEN}_${CFG}" &
        done
    done
done
wait
```

---

## Step 6: Analyze Results

```bash
# Behavioral signatures
python -c "
from analysis import analyze_experiment
results = analyze_experiment('results/gpt4_eval_gpt4_gen_3H3M/')
print(f'Accuracy: {results[\"accuracy\"]:.2%}')
"

# Category breakdown
python -c "
from analysis import plot_category_breakdown
plot_category_breakdown(
    'results/gpt4_eval_gpt4_gen_3H3M/results.json',
    output_path='results/plots/category_breakdown.png'
)
"
```

---

## Step 7: Generate Visualizations

```bash
# RQ plots
python -c "
from analysis import plot_all_rq
plot_all_rq(base_dir='results/', output_dir='results/plots/')
"

# Branching analysis (1H vs 2H vs 3H)
python -c "
from analysis import plot_human_distractor_branching
plot_human_distractor_branching(base_dir='results/', output_dir='results/plots/')
"

# Difficulty scaling
python -c "
from analysis import plot_all_difficulty
plot_all_difficulty(results_dir='results/', output_dir='results/plots/difficulty/')
"
```

---

## Concurrent Naming Convention

To avoid conflicts when running jobs in parallel:

```
datasets/augmented/{dataset}_{generator}_{mode}.json
datasets/filtered/{generator}/{config}.json
results/{evaluator}_{generator}_{config}/
```

**Examples:**
```
datasets/augmented/mmlupro_gpt4_scratch.json
datasets/augmented/arc_claude_condhuman.json
results/gpt4_gpt4_3H3M/
results/claude_gemini_0H3M/
```

---

## Available Models

```bash
python scripts/generate_distractors.py --list-models
```

| Provider | Models |
|----------|--------|
| OpenAI | `gpt-4.1`, `gpt-5-mini`, `gpt-5.2` |
| Anthropic | `claude-opus-4-6`, `claude-sonnet-4-5`, `claude-haiku-4-5` |
| Google | `gemini-3-pro-preview`, `gemini-3-flash-preview` |
| DeepSeek | `deepseek-chat`, `deepseek-reasoner` |
| Local | `qwen3-8b` (vLLM) |

---

## Project Structure

```
augmented-mcqa/
├── scripts/
│   ├── process_all.py           # Process all datasets
│   ├── generate_distractors.py  # Generate synthetic distractors
│   ├── run_experiment.py        # Run evaluation experiments
│   └── test_run.sh              # End-to-end test
├── data/
│   ├── downloader.py            # Download from HuggingFace
│   ├── sorter.py                # Sort MMLU-Pro distractors
│   ├── arc_processor.py         # Process ARC dataset
│   ├── supergpqa_processor.py   # Process SuperGPQA
│   ├── augmentor.py             # Generate distractors
│   └── filter.py                # Create filtered subsets
├── models/                      # Model clients (single source of truth)
├── experiments/                 # Experiment config and runners
├── evaluation/                  # Answer extraction, scoring
├── analysis/                    # Analysis and visualization
├── datasets/                    # Downloaded/processed data
└── results/                     # Experiment outputs
```

---

## License

Private research repository.
