from datasets import load_from_disk
import json

ds = load_from_disk(
    "/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/datasets/mmlu_pro_processed"
)["test"]

print(json.dumps(ds[0], indent=2))
