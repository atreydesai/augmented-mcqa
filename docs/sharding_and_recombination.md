# Sharding and Recombination Guide (Final5 V2)

Final5 V2 uses explicit per-pair work units and Arrow-native outputs.

## 1) Concept Model

- Pair groups: `3 generator models * 3 eval models = 9`
- Modes per pair: `2` (`full_question`, `choices_only`)
- Dataset parts per pair are dynamic:
  - `parts[dataset] = max(1, ceil(rows(dataset) / target_rows_per_subsplit))`
- Each work unit evaluates one `(mode, dataset, dataset_part_idx)` and always runs all 5 settings.

For current row counts (`arc=1000`, `mmlu_pro=1000`, `gpqa=448`) with target `500`:

- parts: `arc=2`, `mmlu_pro=2`, `gpqa=1`
- work units per pair: `2 * (2+2+1) = 10`
- total work units across all pairs: `9 * 10 = 90`

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
  --target-rows-per-subsplit 500
```

Bundle output default:

```text
jobs/generated/<timestamp>/
```

### C. Submit everything

```bash
bash jobs/generated/<timestamp>/submit_all.sh
```

### D. Re-run failed tasks

```bash
sbatch jobs/generated/<timestamp>/<specific>.sbatch
```

### E. Merge partials to canonical outputs

```bash
uv run python scripts/merge_eval_subshards.py \
  --bundle-manifest jobs/generated/<timestamp>/bundle_manifest.json \
  --strict
```

### F. Single-GPU smoke validation

```bash
scripts/run_final5_remote_smoke.sh
```

This smoke path uses a tiny dataset and still exercises the same sharding model
(`mode x dataset_part`) while running all 5 Final5 settings in each work unit.

## 3) Output Layout

Canonical per-config outputs:

```text
results/<generator>/<eval_model>/<mode>/<dataset>/<setting>/summary.json
results/<generator>/<eval_model>/<mode>/<dataset>/<setting>/rows/
results/<generator>/<eval_model>/<mode>/<dataset>/<setting>/merge_metadata.json
```

Partial outputs:

```text
results/<generator>/<eval_model>/<mode>/<dataset>/<setting>/_partials/entry_shard_<i>_of_<n>/summary.json
results/<generator>/<eval_model>/<mode>/<dataset>/<setting>/_partials/entry_shard_<i>_of_<n>/rows/
```

## 4) Verification Commands

### Check bundle-level counts

```bash
jq '.total_pairs, .total_sbatch_files, .total_work_units' jobs/generated/<timestamp>/bundle_manifest.json
```

### Inspect per-pair work units

```bash
jq '.' jobs/generated/<timestamp>/<one-job>.work_units.json
```

### Count canonical summaries

```bash
find results -path '*/summary.json' | rg '/(human_from_scratch|model_from_scratch|augment_human|augment_model|augment_ablation)/summary.json$' | wc -l
```

### Quick merged summary check

```bash
python - <<'PY'
import json, pathlib
p = pathlib.Path('results/<generator>/<eval_model>/<mode>/<dataset>/<setting>/summary.json')
obj = json.loads(p.read_text())
print('total', obj['summary']['total'])
print('accuracy', obj['summary']['accuracy'])
print('rows_path', obj.get('rows_path'))
PY
```

## 5) Troubleshooting

### Missing partial shard files

Symptoms:

- merge summary reports `missing_entry_shards`

Actions:

1. Identify config and missing shard indexes from `merged_summary.json`.
2. Re-run only failed array IDs for the corresponding sbatch file.
3. Re-run strict merge.

### Duplicate partial rows

Symptoms:

- `merge_metadata.json` reports `duplicate_question_idx_rows`

Behavior:

- merge dedupes by `question_idx` with deterministic first-wins order.
- in `--strict`, conflicting duplicates fail merge.

### Non-matching shard params

Symptoms:

- strict merge error: expected `entry_shards` mismatch

Cause:

- mixed outputs from different bundle settings.

Action:

- isolate outputs by bundle directory and rerun with one consistent bundle manifest.
