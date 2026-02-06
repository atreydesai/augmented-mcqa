#!/usr/bin/env python3
"""Analyze MMLU and MMLU-Pro dataset structures."""

from datasets import load_dataset
import json

output = []

def log(msg):
    print(msg)
    output.append(str(msg))

# MMLU Analysis
log('='*60)
log('MMLU Structure (cais/mmlu)')
log('='*60)

try:
    # MMLU has many configs, let's check one
    ds = load_dataset('cais/mmlu', 'all', split='test[:3]', trust_remote_code=True)
    log(f'Columns: {ds.column_names}')
    log(f'\nFirst 2 entries:')
    for i in range(min(2, len(ds))):
        log(f'\n--- Entry {i} ---')
        for k, v in ds[i].items():
            val_str = repr(v)
            if len(val_str) > 400:
                val_str = val_str[:400] + '...'
            log(f'  {k}: {val_str}')
except Exception as e:
    log(f'Error loading MMLU: {e}')
    # Try alternative name
    try:
        ds = load_dataset('lukaemon/mmlu', 'abstract_algebra', split='test[:3]', trust_remote_code=True)
        log(f'Columns (lukaemon/mmlu): {ds.column_names}')
        log(f'\nFirst 2 entries:')
        for i in range(min(2, len(ds))):
            log(f'\n--- Entry {i} ---')
            for k, v in ds[i].items():
                log(f'  {k}: {repr(v)[:400]}')
    except Exception as e2:
        log(f'Error with alternative: {e2}')

# MMLU-Pro Analysis
log('\n' + '='*60)
log('MMLU-Pro Structure (TIGER-Lab/MMLU-Pro)')
log('='*60)

try:
    ds = load_dataset('TIGER-Lab/MMLU-Pro', split='test[:5]', trust_remote_code=True)
    log(f'Columns: {ds.column_names}')
    log(f'\nFirst 3 entries:')
    for i in range(min(3, len(ds))):
        log(f'\n--- Entry {i} ---')
        for k, v in ds[i].items():
            val_str = repr(v)
            if len(val_str) > 400:
                val_str = val_str[:400] + '...'
            log(f'  {k}: {val_str}')
    
    # Option count distribution
    log('\n' + '='*60)
    log('MMLU-Pro Option Count Distribution (first 500)')
    log('='*60)
    ds500 = load_dataset('TIGER-Lab/MMLU-Pro', split='test[:500]', trust_remote_code=True)
    option_counts = {}
    for e in ds500:
        n = len(e.get('options', []))
        option_counts[n] = option_counts.get(n, 0) + 1
    
    log(f'Distribution: {dict(sorted(option_counts.items()))}')
    total = sum(option_counts.values())
    for n, cnt in sorted(option_counts.items()):
        log(f'  {n} options: {cnt} ({cnt/total*100:.1f}%)')
        
    # Check categories
    log('\n' + '='*60)
    log('MMLU-Pro Categories (first 500)')
    log('='*60)
    categories = {}
    for e in ds500:
        cat = e.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    for cat, cnt in sorted(categories.items(), key=lambda x: -x[1])[:10]:
        log(f'  {cat}: {cnt}')
        
except Exception as e:
    import traceback
    log(f'Error loading MMLU-Pro: {e}')
    log(traceback.format_exc())

# Write output to file
with open('/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/mmlu_analysis.txt', 'w') as f:
    f.write('\n'.join(output))

print('\nOutput written to mmlu_analysis.txt')
