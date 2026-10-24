"""
Runs the corpus-specific retrieval smoke set through all three retrieval
strategies and prints a side-by-side comparison table. Use this for regression
checks when the corpus or retrieval implementation changes; it is not a
scholarly benchmark.

Usage:
    python rag/evaluation/compare_strategies.py [--out comparison.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_retrieval import PROJECT_ROOT, run  # type: ignore

STRATEGIES = ["semantic_only", "metadata_only", "hybrid"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(PROJECT_ROOT / "rag" / "evaluation" / "strategy_comparison.json"))
    args = parser.parse_args()

    reports = {s: run(s) for s in STRATEGIES}

    print("=" * 60)
    print("RETRIEVAL STRATEGY COMPARISON")
    print("=" * 60)
    header = f"{'Strategy':16}{'Hit@1':>10}{'Hit@3':>10}{'Hit@5':>10}{'Recall@5':>12}"
    print(header)
    for s in STRATEGIES:
        o = reports[s]["overall"]
        print(
            f"{s:16}{o['hit@1']*100:9.1f}%{o['hit@3']*100:9.1f}%"
            f"{o['hit@5']*100:9.1f}%{o['recall@5']*100:11.1f}%"
        )

    print("\nBy category, Hit@5:")
    categories = sorted(reports["hybrid"]["by_category"].keys())
    cat_header = f"{'Category':20}" + "".join(f"{s:>16}" for s in STRATEGIES)
    print(cat_header)
    for cat in categories:
        row = f"{cat:20}"
        for s in STRATEGIES:
            row += f"{reports[s]['by_category'][cat]['hit@5']*100:15.1f}%"
        print(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({s: reports[s]["overall"] for s in STRATEGIES}, fh, indent=2)
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
