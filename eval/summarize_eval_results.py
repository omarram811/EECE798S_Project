# eval/summarize_eval_results.py

"""
summarize_eval_results.py

Reads all JSON result files from eval/results/*_results.json
and produces:
  - A per-model CSV summary at eval/results/summary_metrics.csv

This script is intentionally lightweight (no pandas dependency).
You can open the CSV directly in Excel.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
SUMMARY_CSV = RESULTS_DIR / "summary_metrics.csv"


def load_result_files() -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")
    for path in RESULTS_DIR.glob("*_results.json"):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        key = path.stem  # e.g., "openai_gpt-5-nano_results"
        out[key] = data
    if not out:
        raise RuntimeError(f"No *_results.json files found in {RESULTS_DIR}")
    return out


def aggregate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}
    # We'll aggregate by unique (provider, model) in this file, but by construction
    # eval_rag_models.py uses one provider/model per result file.
    provider = records[0].get("provider")
    model = records[0].get("model")

    correct = 0
    exact = 0
    any_match_in_topk = 0
    top1_match = 0
    f1_vals = []
    sim_vals = []
    latencies = []
    total_tokens = 0
    total_cost = 0.0

    for r in records:
        m = r.get("metrics", {})
        retr = r.get("retrieval", {})
        if m.get("is_correct"):
            correct += 1
        if m.get("exact_match_normalized"):
            exact += 1
        if retr.get("any_match_in_topk"):
            any_match_in_topk += 1
        if retr.get("top1_matches_gold"):
            top1_match += 1
        if m.get("f1") is not None:
            f1_vals.append(float(m["f1"]))
        if m.get("semantic_similarity") is not None:
            sim_vals.append(float(m["semantic_similarity"]))
        if m.get("latency_seconds") is not None:
            latencies.append(float(m["latency_seconds"]))
        if m.get("total_tokens") is not None:
            total_tokens += int(m["total_tokens"])
        if m.get("total_cost") is not None:
            total_cost += float(m["total_cost"])

    def avg(xs: List[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    out = {
        "provider": provider,
        "model": model,
        "n_records": n,
        "task_accuracy_is_correct": correct / n,
        "exact_match_rate": exact / n,
        "retrieval_top1_accuracy": top1_match / n,
        "retrieval_topk_recall": any_match_in_topk / n,
        "avg_f1": avg(f1_vals),
        "avg_semantic_similarity": avg(sim_vals),
        "avg_latency_seconds": avg(latencies),
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "avg_cost_per_question": (total_cost / n) if n and total_cost else 0.0,
    }
    return out


def main():
    all_files = load_result_files()
    summaries: List[Dict[str, Any]] = []

    for name, records in all_files.items():
        metrics = aggregate_metrics(records)
        metrics["result_file"] = name
        summaries.append(metrics)

    # Write CSV
    fieldnames = [
        "result_file",
        "provider",
        "model",
        "n_records",
        "task_accuracy_is_correct",
        "exact_match_rate",
        "retrieval_top1_accuracy",
        "retrieval_topk_recall",
        "avg_f1",
        "avg_semantic_similarity",
        "avg_latency_seconds",
        "total_tokens",
        "total_cost",
        "avg_cost_per_question",
    ]

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    print(f"Wrote summary metrics to {SUMMARY_CSV}")
    for row in summaries:
        avg_f1 = row.get("avg_f1")
        avg_f1_str = f"{avg_f1:.3f}" if avg_f1 is not None else "nan"
        print(
            f"- {row['provider']} / {row['model']}: "
            f"acc={row['task_accuracy_is_correct']:.3f}, "
            f"retr_top1={row['retrieval_top1_accuracy']:.3f}, "
            f"avg_f1={avg_f1_str}"
        )


if __name__ == "__main__":
    main()
