from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict

from utils.logs import read_log


def load_results(results_path: Path) -> Dict[str, Any]:
    if results_path.suffix != ".eval":
        raise ValueError(f"Expected Inspect `.eval` log, got {results_path}")

    log = read_log(results_path)
    results: list[dict[str, Any]] = []
    for sample in log.samples:
        if not sample.scores:
            continue
        score = next(iter(sample.scores.values()))
        metadata = dict(getattr(score, "metadata", {}) or {})
        results.append(
            {
                "question_idx": metadata.get("question_idx"),
                "is_correct": bool(getattr(score, "value", False)),
                "prediction_type": metadata.get("prediction_type", "?"),
                "category": metadata.get("category", ""),
            }
        )
    return {
        "config": getattr(log.eval, "metadata", {}) or {},
        "results": results,
    }


def compute_behavioral_signature(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {"G": 0, "H": 0, "M": 0, "?": 0}    
    for r in results:
        ptype = r.get("prediction_type", "?")
        if ptype in counts:
            counts[ptype] += 1
        else:
            counts["?"] += 1
    
    total = sum(counts.values())
    
    rates = {}
    if total > 0:
        rates = {k: v / total for k, v in counts.items()}
    
    return {
        "counts": counts,
        "rates": rates,
        "total": total,
    }


def compute_gold_rate(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    
    correct = sum(1 for r in results if r.get("is_correct", False))
    return correct / len(results)


def compute_hierarchical_signature(
    results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    by_category: Dict[str, List[Dict]] = defaultdict(list)
    
    for r in results:
        cat = r.get("category", "unknown")
        by_category[cat].append(r)
    
    signatures = {}
    for cat, cat_results in by_category.items():
        signatures[cat] = compute_behavioral_signature(cat_results)
        signatures[cat]["gold_rate"] = compute_gold_rate(cat_results)
    
    return signatures


def analyze_experiment(results_path: Path) -> Dict[str, Any]:
    data = load_results(results_path)
    
    # Get individual results
    if "results" in data:
        results = data["results"]
    elif isinstance(data, list):
        results = data
    else:
        results = []
    
    analysis = {
        "overall": {
            "gold_rate": compute_gold_rate(results),
            "signature": compute_behavioral_signature(results),
            "total_entries": len(results),
        },
        "by_category": compute_hierarchical_signature(results),
    }
    
    # Add config if available
    if "config" in data:
        analysis["config"] = data["config"]
    
    return analysis


def compare_experiments(
    experiment_paths: List[Path],
    names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if names is None:
        names = [f"exp_{i}" for i in range(len(experiment_paths))]
    
    comparisons = {}
    
    for name, path in zip(names, experiment_paths):
        analysis = analyze_experiment(path)
        comparisons[name] = {
            "gold_rate": analysis["overall"]["gold_rate"],
            "signature": analysis["overall"]["signature"],
            "total": analysis["overall"]["total_entries"],
        }
    
    return {
        "experiments": comparisons,
        "summary": {
            "gold_rates": {
                name: comp["gold_rate"]
                for name, comp in comparisons.items()
            },
        },
    }

def format_signature_table(
    experiments: Dict[str, Dict[str, Any]],
    include_counts: bool = False,
) -> str:
    lines = []
    
    # Header
    if include_counts:
        header = "| Experiment | Gold% | H% | M% | G | H | M |"
        sep = "|------------|-------|----|----|---|---|---|"
    else:
        header = "| Experiment | Gold% | H% | M% |"
        sep = "|------------|-------|----|----|"
    
    lines.append(header)
    lines.append(sep)
    
    for name, data in experiments.items():
        sig = data.get("signature", {})
        rates = sig.get("rates", {})
        counts = sig.get("counts", {})
        
        gold = rates.get("G", 0) * 100
        h_rate = rates.get("H", 0) * 100
        m_rate = rates.get("M", 0) * 100
        
        if include_counts:
            line = f"| {name} | {gold:.1f} | {h_rate:.1f} | {m_rate:.1f} | {counts.get('G', 0)} | {counts.get('H', 0)} | {counts.get('M', 0)} |"
        else:
            line = f"| {name} | {gold:.1f} | {h_rate:.1f} | {m_rate:.1f} |"
        
        lines.append(line)
    
    return "\n".join(lines)
