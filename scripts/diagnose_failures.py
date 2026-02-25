from datasets import load_from_disk

p = "/Users/ndesai-air/Documents/GitHub/augmented-mcqa/datasets/augmented/final5_full_20260225_004316_gpt-5.2-2025-12-11"
ds = load_from_disk(p)

required = [
    ("model_from_scratch", 3),
    ("augment_human", 6),
    ("augment_model", 9),
    ("augment_ablation", 9),
]

for split in ds.keys():
    failed = []
    for i, row in enumerate(ds[split]):
        missing = [k for k, n in required if len(row.get(k) or []) < n]
        if missing:
            failed.append({
                "idx": i,
                "id": row.get("id"),
                "question_id": row.get("question_id"),
                "missing": missing,
                "question": row.get("question", "")[:140],
            })

    print(f"\n{split}: failed_rows={len(failed)}")
    for r in failed:
        print(r)