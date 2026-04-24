import json
import pandas as pd
from collections import defaultdict, Counter

data_path = "/fs/clip-projects/rlab/atrey/qgqa/augmented-mcqa/results/atrey_writing_flaw_rows_all.jsonl"

def categorize(row):
    config = row.get("config", "")
    model = row.get("model", "")
    
    if "human" in config:
        return "Human"
    elif "gpt" in model.lower():
        return "GPT"
    elif "gemini" in model.lower():
        return "Gemini"
    elif "qwen" in model.lower():
        return "Qwen"
    else:
        return model

records = []
flaw_counts = defaultdict(Counter)
total_counts = defaultdict(int)

with open(data_path, 'r') as f:
    for line in f:
        row = json.loads(line)
        cat = categorize(row)
        total_counts[cat] += 1
        
        md = row.get("writing_flaw", {}).get("metadata", {})
        flaws = md.get("writing_flaws", [])
        
        for flaw in flaws:
            rule_name = flaw.get("name")
            flaw_counts[cat][rule_name] += 1

print("Total questions per generator:")
for k, v in total_counts.items():
    print(f"{k}: {v}")

print("\nFlaw rates (fails per question):")
# Get all unique rules
all_rules = set()
for counts in flaw_counts.values():
    all_rules.update(counts.keys())

for rule in sorted(list(all_rules)):
    print(f"\nRule: {rule}")
    for cat in ["Human", "GPT", "Gemini", "Qwen"]:
        if total_counts[cat] > 0:
            rate = flaw_counts[cat][rule] / total_counts[cat]
            print(f"  {cat}: {rate:.4f} ({flaw_counts[cat][rule]} fails)")
