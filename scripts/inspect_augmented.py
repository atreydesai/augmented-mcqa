import sys
import os
from pathlib import Path
from datasets import load_from_disk
import argparse

# Add project root to path to allow importing from data.hub_utils
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from data.hub_utils import push_dataset_to_hub
except ImportError:
    # Fallback if imports are still tricky
    push_dataset_to_hub = None
    print("Warning: Could not import push_dataset_to_hub")

def main():
    parser = argparse.ArgumentParser(description="Inspect and optionally push augmented dataset to HF Hub.")
    parser.add_argument("--path", type=str, default="datasets/augmented/test_human_fixed_region", help="Path to dataset on disk")
    parser.add_argument("--push", action="store_true", help="Push to Hugging Face Hub")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items to inspect per split")
    args = parser.parse_args()

    ds_path = args.path
    if not os.path.exists(ds_path):
        print(f"Error: Path {ds_path} does not exist.")
        return

    print(f"Loading dataset from: {ds_path}")
    ds_dict = load_from_disk(ds_path)
    print(f"Splits: {list(ds_dict.keys())}")

    # Inspect entries of each split
    for split_name, ds in ds_dict.items():
        print(f"\n{'='*80}")
        print(f"Split: {split_name}")
        print(f"{'='*80}")
        if len(ds) == 0:
            print("Empty split.")
            continue
            
        inspect_count = args.limit if args.limit is not None else 1
        for i in range(min(inspect_count, len(ds))):
            row = ds[i]
            cols = ds.column_names
            
            if inspect_count > 1:
                print(f"\n--- ENTRY {i+1} ---")
        
        print(f"\n[BASIC INFO]")
        print(f"Question: {row.get('question', '')[:200]}...")
        print(f"Answer: {row.get('answer', '')}")
        if 'choices_human' in cols:
            print(f"Choices Human: {row['choices_human']}")
            
        # prefixes to check
        for prefix in ['qa', 'qadh', 'qadm']:
            suffix = {'qa': 'scratch', 'qadh': 'dhuman', 'qadm': 'dmodel'}[prefix]
            model_col = f"cond_model_q_a_{suffix}"
            
            if (f"{prefix}_full_question" in cols or model_col in cols) and (row.get(f"{prefix}_full_question") is not None or row.get(model_col) is not None):
                print(f"\n--- PREFIX: {prefix} ({suffix}) ---")
                
                # 1. qa_full_question
                if f"{prefix}_full_question" in cols:
                    print(f"\n[{prefix}_full_question]")
                    print(row[f"{prefix}_full_question"])
                
                # 2. qa_model_input
                if f"{prefix}_model_input" in cols:
                    print(f"\n[{prefix}_model_input]")
                    print(f"{row[f'{prefix}_model_input'][:300]}...")
                
                # 3. qa_model_output
                if f"{prefix}_model_output" in cols:
                    print(f"\n[{prefix}_model_output]")
                    print(f"{row[f'{prefix}_model_output'][:300]}...")
                
                # 4. cond_model_q_a_scratch
                if model_col in cols:
                    print(f"\n[{model_col}]")
                    print(row[model_col])
                
                # 5. qa_options_randomized
                if f"{prefix}_options_randomized" in cols:
                    print(f"\n[{prefix}_options_randomized]")
                    print(row[f"{prefix}_options_randomized"])
                    print(f"Length: {len(row[f'{prefix}_options_randomized']) if row[f'{prefix}_options_randomized'] else 0}")
                
                # 6. qa_correct_answer_letter
                if f"{prefix}_correct_answer_letter" in cols:
                    print(f"\n[{prefix}_correct_answer_letter]")
                    print(row[f"{prefix}_correct_answer_letter"])

    # Push to hub if requested
    if args.push:
        if push_dataset_to_hub:
            dataset_name = Path(ds_path).name
            print(f"\nPushing dataset '{dataset_name}' to Hub...")
            push_dataset_to_hub(ds_dict, dataset_name=dataset_name)
        else:
            print("\nError: push_dataset_to_hub not available.")

if __name__ == "__main__":
    main()
