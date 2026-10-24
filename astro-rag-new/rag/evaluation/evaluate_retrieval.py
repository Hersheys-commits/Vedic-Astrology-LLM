"""
Runs the corpus-specific retrieval smoke set (`retrieval_gold.jsonl`) against
one retrieval strategy and reports Hit@K / Precision@K / Recall@K overall and
by category. The smoke set is generated from real chunks in the supplied OCR
corpus; it is not a manually adjudicated scholarly relevance benchmark.

Usage:
    python rag/evaluation/evaluate_retrieval.py [--strategy hybrid|metadata_only|semantic_only] [--out report.json]

Run from anywhere; it locates the project root itself.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.corpus_store import reload_store  # noqa: E402
from app.retrieval.hybrid import retrieve_for_fact  # noqa: E402

GOLD_PATH = PROJECT_ROOT / "rag" / "evaluation" / "retrieval_gold.jsonl"
KS = [1, 3, 5]


def load_gold() -> list[dict]:
    tests = []
    with GOLD_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                tests.append(json.loads(line))
    return tests


def evaluate_one(expected: list[str], retrieved_ids: list[str], k: int) -> dict:
    top_k = retrieved_ids[:k]
    relevant_retrieved = [c for c in top_k if c in expected]
    hit = 1.0 if relevant_retrieved else 0.0
    precision = len(relevant_retrieved) / k if k else 0.0
    recall = len(relevant_retrieved) / len(expected) if expected else 0.0
    reciprocal_rank = (
        1.0 / (retrieved_ids.index(relevant_retrieved[0]) + 1)
        if relevant_retrieved
        else 0.0
    )
    return {
        "hit": hit,
        "precision": precision,
        "recall": recall,
        "reciprocal_rank": reciprocal_rank,
    }


def run(strategy: str) -> dict:
    store = reload_store()
    tests = load_gold()

    per_query = []
    for t in tests:
        max_k = max(KS)
        results = retrieve_for_fact(
            t["chart_facts"], strategy=strategy, top_k=max_k, store=store,
            query_context=t.get("query"),
        )
        retrieved_ids = [r["chunk_id"] for r in results]
        metrics_by_k = {k: evaluate_one(t["expected_chunks"], retrieved_ids, k) for k in KS}
        per_query.append({
            "id": t["id"],
            "category": t["category"],
            "query": t["query"],
            "expected_chunks": t["expected_chunks"],
            "retrieved_chunks": retrieved_ids,
            "metrics": metrics_by_k,
        })

    def aggregate(subset: list[dict]) -> dict:
        agg = {}
        for k in KS:
            n = len(subset) or 1
            agg[f"hit@{k}"] = round(sum(q["metrics"][k]["hit"] for q in subset) / n, 4)
            agg[f"precision@{k}"] = round(sum(q["metrics"][k]["precision"] for q in subset) / n, 4)
            agg[f"recall@{k}"] = round(sum(q["metrics"][k]["recall"] for q in subset) / n, 4)
        n = len(subset) or 1
        agg["mrr"] = round(
            sum(
                max(q["metrics"][k]["reciprocal_rank"] for k in KS)
                for q in subset
            )
            / n,
            4,
        )
        agg["n_queries"] = len(subset)
        return agg

    by_category = {}
    for cat in sorted({q["category"] for q in per_query}):
        by_category[cat] = aggregate([q for q in per_query if q["category"] == cat])

    report = {
        "strategy": strategy,
        "overall": aggregate(per_query),
        "by_category": by_category,
        "queries": per_query,
    }
    return report


def print_report(report: dict) -> None:
    print("=" * 50)
    print("RAG RETRIEVAL EVALUATION")
    print(f"strategy: {report['strategy']}")
    print("=" * 50)
    o = report["overall"]
    print(f"\nQueries: {o['n_queries']}\n")
    print(f"{'':13}{'@1':>8}{'@3':>8}{'@5':>8}")
    print(f"{'Hit Rate':13}{o['hit@1']*100:7.1f}%{o['hit@3']*100:7.1f}%{o['hit@5']*100:7.1f}%")
    print(f"{'Precision':13}{o['precision@1']*100:7.1f}%{o['precision@3']*100:7.1f}%{o['precision@5']*100:7.1f}%")
    print(f"{'Recall':13}{o['recall@1']*100:7.1f}%{o['recall@3']*100:7.1f}%{o['recall@5']*100:7.1f}%")
    print(f"{'MRR':13}{o['mrr']*100:7.1f}%")

    print("\n" + "-" * 30)
    print("By Category (Hit@5)")
    print("-" * 30)
    for cat, m in report["by_category"].items():
        print(f"{cat:20}{m['hit@5']*100:6.1f}%   (n={m['n_queries']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="hybrid", choices=["hybrid", "metadata_only", "semantic_only"])
    parser.add_argument("--out", default=str(PROJECT_ROOT / "rag" / "evaluation" / "retrieval_report.json"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    report = run(args.strategy)
    if not args.quiet:
        print_report(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    if not args.quiet:
        print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
