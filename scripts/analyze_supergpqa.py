#!/usr/bin/env python3
"""Analyze SuperGPQA by downloading JSONL directly."""

import json
import os
from huggingface_hub import hf_hub_download

output = []

def log(msg):
    print(msg)
    output.append(str(msg))

log('='*60)
log('SuperGPQA Structure (Direct JSONL)')
log('='*60)

try:
    # Download the JSONL file directly
    file_path = hf_hub_download(
        repo_id='m-a-p/SuperGPQA',
        filename='SuperGPQA-all.jsonl',
        repo_type='dataset',
        force_download=True
    )
    log(f'Downloaded to: {file_path}')
    
    # Read and analyze
    entries = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 1000:  # Only load first 1000 for analysis
                entries.append(json.loads(line))
    
    log(f'Loaded {len(entries)} entries')
    
    if entries:
        log(f'\nColumns: {list(entries[0].keys())}')
        
        log('\nFirst 3 entries:')
        for i in range(min(3, len(entries))):
            log(f'\n--- Entry {i} ---')
            for k, v in entries[i].items():
                val_str = repr(v)
                if len(val_str) > 400:
                    val_str = val_str[:400] + '...'
                log(f'  {k}: {val_str}')
        
        # Option count distribution
        log('\n' + '='*60)
        log('SuperGPQA Option Count Distribution')
        log('='*60)
        
        option_counts = {}
        for entry in entries:
            n = 0
            for k in ['A','B','C','D','E','F','G','H','I','J']:
                if k in entry and entry[k] is not None and str(entry[k]).strip() != '':
                    n += 1
            option_counts[n] = option_counts.get(n, 0) + 1
        
        log(f'Distribution: {dict(sorted(option_counts.items()))}')
        total = sum(option_counts.values())
        for n, cnt in sorted(option_counts.items()):
            log(f'  {n} options: {cnt} ({cnt/total*100:.1f}%)')
        
        # Find 10-option example
        log('\n10-option question example:')
        for entry in entries:
            n_opts = sum(1 for k in ['A','B','C','D','E','F','G','H','I','J'] 
                         if k in entry and entry[k] is not None and str(entry[k]).strip() != '')
            if n_opts == 10:
                for k, v in entry.items():
                    val_str = repr(v)
                    if len(val_str) > 400:
                        val_str = val_str[:400] + '...'
                    log(f'  {k}: {val_str}')
                break
        
except Exception as e:
    import traceback
    log(f'Error: {e}')
    log(traceback.format_exc())

# Write output to file
with open('/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/supergpqa_analysis.txt', 'w') as f:
    f.write('\n'.join(output))

print('Analysis complete. Output written to supergpqa_analysis.txt')
