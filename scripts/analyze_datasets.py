#!/usr/bin/env python3
"""Analyze HuggingFace dataset structures for ARC and SuperGPQA."""

from datasets import load_dataset
import json

output = []

def log(msg):
    print(msg)
    output.append(msg)

# ARC Analysis
log('='*60)
log('ARC-Easy Structure')
log('='*60)
try:
    arc_easy = load_dataset('allenai/ai2_arc', 'ARC-Easy', split='test', trust_remote_code=True)
    log(f'Columns: {arc_easy.column_names}')
    log(f'Total entries: {len(arc_easy)}')
    log('\nFirst 2 entries:')
    for i in range(2):
        log(f'\n--- Entry {i} ---')
        for k, v in arc_easy[i].items():
            log(f'  {k}: {repr(v)}')
except Exception as e:
    log(f'Error loading ARC-Easy: {e}')

log('\n' + '='*60)
log('ARC-Challenge Structure')
log('='*60)
try:
    arc_chal = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test', trust_remote_code=True)
    log(f'Columns: {arc_chal.column_names}')
    log(f'Total entries: {len(arc_chal)}')
    log('\nFirst 2 entries:')
    for i in range(2):
        log(f'\n--- Entry {i} ---')
        for k, v in arc_chal[i].items():
            log(f'  {k}: {repr(v)}')
except Exception as e:
    log(f'Error loading ARC-Challenge: {e}')

# SuperGPQA Analysis
log('\n' + '='*60)
log('SuperGPQA Structure')
log('='*60)
try:
    ds = load_dataset('m-a-p/SuperGPQA', split='train[:10]', trust_remote_code=True)
    log(f'Columns: {ds.column_names}')
    log('\nFirst 3 entries:')
    for i in range(3):
        log(f'\n--- Entry {i} ---')
        for k, v in ds[i].items():
            val_str = repr(v)
            if len(val_str) > 300:
                val_str = val_str[:300] + '...'
            log(f'  {k}: {val_str}')
    
    # Option count distribution
    log('\n' + '='*60)
    log('SuperGPQA Option Count Distribution (first 500)')
    log('='*60)
    ds500 = load_dataset('m-a-p/SuperGPQA', split='train[:500]', trust_remote_code=True)
    option_counts = {}
    for e in ds500:
        n = 0
        for k in ['A','B','C','D','E','F','G','H','I','J']:
            if k in e and e[k] is not None and e[k] != '':
                n += 1
        option_counts[n] = option_counts.get(n, 0) + 1
    
    log(f'Distribution: {dict(sorted(option_counts.items()))}')
    total = sum(option_counts.values())
    for n, count in sorted(option_counts.items()):
        log(f'  {n} options: {count} ({count/total*100:.1f}%)')
        
except Exception as e:
    log(f'Error loading SuperGPQA: {e}')

# Write output to file
with open('/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/dataset_analysis.txt', 'w') as f:
    f.write('\n'.join(output))

print('\nAnalysis complete. Output written to dataset_analysis.txt')
