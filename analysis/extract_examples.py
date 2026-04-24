import json
import random

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

gpt_equal_length_fails = []
gpt_single_best_answer_fails = []

with open(data_path, 'r') as f:
    for line in f:
        row = json.loads(line)
        cat = categorize(row)
        
        if cat == "GPT":
            md = row.get("writing_flaw", {}).get("metadata", {})
            flaws = md.get("writing_flaws", [])
            for flaw in flaws:
                if flaw.get("name") == "equal_length_options":
                    gpt_equal_length_fails.append((row, flaw.get("explanation")))
                if flaw.get("name") == "single_best_answer":
                    gpt_single_best_answer_fails.append((row, flaw.get("explanation")))

# Print a couple of examples
print("=== GPT EQUAL LENGTH FAILS ===")
for row, exp in gpt_equal_length_fails[:3]:
    print(f"Q: {row['question']}")
    print(f"Choices: {row['choices']}")
    print(f"Explanation: {exp}\n")

print("=== GPT SINGLE BEST ANSWER FAILS ===")
for row, exp in gpt_single_best_answer_fails[:3]:
    print(f"Q: {row['question']}")
    print(f"Choices: {row['choices']}")
    print(f"Explanation: {exp}\n")
