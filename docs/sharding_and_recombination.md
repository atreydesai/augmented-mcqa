# Sharding and Recombination Guide

This guide is the operational playbook for Final5 evaluation when scaling past the base 18 job groups.

## 1) Concept Model

Final5 eval uses two dimensions of splitting:

- Base groups: `3 generator models * 3 eval models * 2 modes = 18`
- Config shards (`num_gpus`): split config list across array tasks
- Entry sub-shards (`entry_shards`): split question rows within each config

### Formulas

Given:

- base groups = `18`
- config shards = `num_gpus`
- entry sub-shards = `entry_shards`

Then:

- job groups = `18 * entry_shards`
- array tasks submitted = `18 * entry_shards * num_gpus`

Example (`num_gpus=8`, `entry_shards=4`):

- job groups = `18 * 4 = 72`
- array tasks = `18 * 4 * 8 = 576`

## 2) End-to-End Commands

### A. Build datasets (generation)

```bash
uv run python scripts/regenerate_experiments.py \
  --processed-dataset datasets/processed/unified_processed_v2 \
  --output-root datasets/augmented
```

### B. Build SLURM bundle

```bash
uv run python scripts/build_eval_slurm_bundle.py \
  --manifest datasets/augmented/<final5_regeneration_manifest>.json \
  --num-gpus 8 \
  --entry-shards 4
```

Bundle output default:

```text
jobs/generated/<timestamp>/
```

### C. Submit everything

```bash
bash jobs/generated/<timestamp>/submit_all.sh
```

Dry-run submission preview:

```bash
bash jobs/generated/<timestamp>/submit_all.sh --dry-run
```

### D. Re-run failed tasks

Find the relevant sbatch file and resubmit only failed array IDs:

```bash
sbatch --array=1,4,7 jobs/generated/<timestamp>/<specific>.sbatch
```

### E. Merge partials to canonical results

```bash
uv run python scripts/merge_eval_subshards.py \
  --bundle-manifest jobs/generated/<timestamp>/bundle_manifest.json \
  --strict
```

## 3) Practical Scale Recipes

### Small smoke (quick validation)

- `num_gpus=1`
- `entry_shards=1`
- optionally run with eval `--limit 2`

```bash
uv run python scripts/build_eval_slurm_bundle.py \
  --manifest datasets/augmented/<manifest>.json \
  --num-gpus 1 \
  --entry-shards 1
```

### Medium run

- `num_gpus=4`
- `entry_shards=2`

```bash
uv run python scripts/build_eval_slurm_bundle.py \
  --manifest datasets/augmented/<manifest>.json \
  --num-gpus 4 \
  --entry-shards 2
```

### Full run (common)

- `num_gpus=8`
- `entry_shards=4`

```bash
uv run python scripts/build_eval_slurm_bundle.py \
  --manifest datasets/augmented/<manifest>.json \
  --num-gpus 8 \
  --entry-shards 4
```

### Max split (queue pressure control)

- keep per-task runtime lower by increasing `entry_shards`
- example: `num_gpus=8`, `entry_shards=12`

```bash
uv run python scripts/build_eval_slurm_bundle.py \
  --manifest datasets/augmented/<manifest>.json \
  --num-gpus 8 \
  --entry-shards 12
```

## 4) Verification Commands

### Check bundle math

```bash
jq '.total_base_groups, .total_job_groups, .total_array_tasks' jobs/generated/<timestamp>/bundle_manifest.json
```

### Check expected config roots

```bash
jq '.config_roots | length' jobs/generated/<timestamp>/bundle_manifest.json
```

### Count canonical merged result files

```bash
find results -path '*/results.json' | rg '/(human_from_scratch|model_from_scratch|augment_human|augment_model|augment_ablation)/results.json$' | wc -l
```

Expected canonical config files after full merge:

- `3 generators * 3 eval models * 2 modes * 3 datasets * 5 settings = 270`

### Inspect missing partial shard files before merge

```bash
uv run python scripts/merge_eval_subshards.py \
  --bundle-manifest jobs/generated/<timestamp>/bundle_manifest.json
```

Then inspect `merged_summary.json` for `missing_partials` or per-config `missing_entry_shards`.

### Row-level sanity check (example)

```bash
python - <<'PY'
import json, pathlib
p = pathlib.Path('results/<generator>/<eval_model>/<mode>/<dataset>/<setting>/results.json')
obj = json.loads(p.read_text())
print('total', obj['summary']['total'])
print('accuracy', obj['summary']['accuracy'])
PY
```

### Aggregate row-count sanity checks (global + per setting)

```bash
python - <<'PY'
import json
from collections import Counter
from pathlib import Path

setting_totals = Counter()
global_total = 0

for p in Path('results').glob('*/*/*/*/*/results.json'):
    setting = p.parent.name
    obj = json.loads(p.read_text())
    n = int(obj.get('summary', {}).get('total', 0))
    setting_totals[setting] += n
    global_total += n

print('global_total_rows', global_total)
print('by_setting')
for k in sorted(setting_totals):
    print(f'  {k}: {setting_totals[k]}')
PY
```

Expected global totals for full Final5 eval **when all 3 datasets contribute 1000 rows**:

- `270000` rows
- per-setting expected rows: `54000` each

If a dataset has fewer available rows after preprocessing/filtering (for example GPQA < 1000),
actual totals will be lower. Use `bundle_manifest.json` `expected_eval_rows` as the run target,
and compare it against merged output totals.

## 5) Troubleshooting

### Missing partial shard files

Symptoms:

- merge summary reports `missing_entry_shards`

Actions:

1. Identify config and missing shard indexes from `merged_summary.json`.
2. Re-run only failed array IDs for the corresponding sbatch file:

```bash
sbatch --array=<failed_ids_csv> jobs/generated/<timestamp>/<specific>.sbatch
```

3. Re-run strict merge.

### Duplicate partial rows

Symptoms:

- merge metadata reports `duplicate_question_idx_rows`

Behavior:

- merge dedupes deterministically by `question_idx`
- in `--strict`, conflicting duplicates fail merge

Action:

- inspect offending config partials and re-run shard(s), then merge again.

### Non-matching shard params

Symptoms:

- strict merge error: expected `entry_shards` mismatch

Cause:

- mixed outputs from different bundle settings

Action:

- keep bundle outputs isolated by timestamp directory
- clear/restart that config path, rerun with one consistent `entry_shards` value.

### Resume/restart strategy

- Re-submitting is safe with `--skip-existing` in generated sbatch scripts.
- Targeted restart pattern:

```bash
sbatch --array=<failed_ids_csv> jobs/generated/<timestamp>/<specific>.sbatch
uv run python scripts/merge_eval_subshards.py --bundle-manifest jobs/generated/<timestamp>/bundle_manifest.json --strict
```
