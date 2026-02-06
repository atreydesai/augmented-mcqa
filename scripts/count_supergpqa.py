#!/usr/bin/env python3
"""Get proper option count for SuperGPQA using options list."""

import json
from huggingface_hub import hf_hub_download

output = []
def log(msg):
    print(msg)
    output.append(str(msg))

log('='*60)
log('SuperGPQA Option Count (from options list)')
log('='*60)

try:
    file_path = hf_hub_download(
        repo_id='m-a-p/SuperGPQA',
        filename='SuperGPQA-all.jsonl',
        repo_type='dataset'
    )
    
    # Read all entries
    option_counts = {}
    total = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            n = len(entry.get('options', []))
            option_counts[n] = option_counts.get(n, 0) + 1
            total += 1
    
    log(f'Total entries: {total}')
    log(f'\nOption count distribution:')
    for n, cnt in sorted(option_counts.items()):
        log(f'  {n} options: {cnt} ({cnt/total*100:.1f}%)')
    
    # Count 10-option questions specifically
    ten_option = option_counts.get(10, 0)
    log(f'\n10-option questions: {ten_option} ({ten_option/total*100:.1f}%)')
    
    # Get fields for 10-option questions
    log('\nSample 10-option question fields:')
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if len(entry.get('options', [])) == 10:
                log(f"  discipline: {entry.get('discipline')}")
                log(f"  field: {entry.get('field')}")
                log(f"  subfield: {entry.get('subfield')}")
                log(f"  difficulty: {entry.get('difficulty')}")
                break

except Exception as e:
    import traceback
    log(f'Error: {e}')
    log(traceback.format_exc())

with open('/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/supergpqa_counts.txt', 'w') as f:
    f.write('\n'.join(output))

print('\nOutput written to supergpqa_counts.txt')
