
import sys
import os
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.arc_processor import load_arc_dataset
from data.gpqa_processor import load_gpqa_dataset
from data.mmlu_pro_processor import process_mmlu_pro

def analyze_dataset(name, entries, is_dict=False):
    print(f"\n{'='*20} {name} {'='*20}")
    if not entries:
        print("No entries found.")
        return

    first_entry = entries[0] if not is_dict else entries['train'][0]
    
    print(f"Type: {type(first_entry)}")
    print(f"Keys: {list(first_entry.keys())}")
    
    # Print sample values for key columns
    keys_to_check = [
        "question", "options", "answer", "gold_answer", "answer_index", 
        "category", "src", "subfield", "difficulty", "choices_human", 
        "lines", "choices", "answerKey"
    ]
    
    for key in keys_to_check:
        if key in first_entry:
            val = first_entry[key]
            if isinstance(val, list) and len(val) > 2:
                print(f"{key}: List[{len(val)}] -> {val[:2]}...")
            else:
                print(f"{key}: {val}")
        else:
            print(f"{key}: <MISSING>")

def main():
    print("Starting Dataset Analysis...")
    
    # ARC Easy
    try:
        print("\nLoading ARC-Easy...")
        arc_easy = load_arc_dataset("easy", limit=5)
        analyze_dataset("ARC-Easy", arc_easy)
    except Exception as e:
        print(f"Error loading ARC-Easy: {e}")

    # ARC Challenge
    try:
        print("\nLoading ARC-Challenge...")
        arc_challenge = load_arc_dataset("challenge", limit=5)
        analyze_dataset("ARC-Challenge", arc_challenge)
    except Exception as e:
        print(f"Error loading ARC-Challenge: {e}")

    # GPQA
    try:
        print("\nLoading GPQA...")
        gpqa = load_gpqa_dataset(limit=5)
        analyze_dataset("GPQA", gpqa)
    except Exception as e:
        print(f"Error loading GPQA: {e}")

    # MMLU-Pro (using existing processed if available, or raw)
    try:
        print("\nLoading MMLU-Pro...")
        # Try to load formatted entries if possible, we use the processor but limit it
        # Note: process_mmlu_pro returns a DatasetDict
        from config import RAW_DATASETS_DIR
        input_path = RAW_DATASETS_DIR / "mmlu_pro"
        mmlu_all = RAW_DATASETS_DIR / "mmlu_all"
        
        # We need to ensure these paths exist or the script will fail
        if not input_path.exists():
            print(f"MMLU Pro raw path {input_path} does not exist. Skipping.")
        else:
             # Just load a small sample independently to avoid full processing
            mmlu_pro = process_mmlu_pro(
                mmlu_pro_path=input_path,
                mmlu_path=mmlu_all, 
                limit=5,
                report_whitespace_bugs=False,
                output_path=Path("temp_mmlu_analysis")
            )
            analyze_dataset("MMLU-Pro", mmlu_pro, is_dict=True)
            
            # Clean up temp
            import shutil
            if os.path.exists("temp_mmlu_analysis"):
                shutil.rmtree("temp_mmlu_analysis")
                
    except Exception as e:
        print(f"Error loading MMLU-Pro: {e}")

if __name__ == "__main__":
    main()
