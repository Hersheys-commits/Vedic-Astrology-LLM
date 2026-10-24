#!/usr/bin/env python3
"""Compare two Astro RAG manual-run JSON files for reproducibility and pipeline stability.

Usage:
  python scripts/audit_manual_runs.py previous.json current.json

This is a diagnostic only: text similarity is not a correctness score. It compares the
recorded question, chart facts, retrieved passages, status and visible answer text.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("results", data.get("answers", []))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array (or results array) in {path}")
    return [row for row in data if isinstance(row, dict)]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--out", type=Path, help="Optional JSON summary path")
    args = parser.parse_args()

    previous = load_rows(args.previous)
    current = load_rows(args.current)
    key = lambda row: normalize(row.get("question") or row.get("index"))
    prev = {key(row): row for row in previous}
    curr = {key(row): row for row in current}
    common = sorted(set(prev) & set(curr))
    sims = [
        SequenceMatcher(None, normalize(prev[q].get("answer")), normalize(curr[q].get("answer"))).ratio()
        for q in common
    ]
    report = {
        "previous_file": args.previous.name,
        "current_file": args.current.name,
        "previous_rows": len(previous),
        "current_rows": len(current),
        "matching_question_count": len(common),
        "question_sets_identical": set(prev) == set(curr),
        "exact_answer_matches": sum(
            normalize(prev[q].get("answer")) == normalize(curr[q].get("answer"))
            for q in common
        ),
        "answer_similarity": {
            "method": "difflib.SequenceMatcher after whitespace normalization; not a correctness metric",
            "mean": round(statistics.mean(sims), 4) if sims else None,
            "median": round(statistics.median(sims), 4) if sims else None,
            "min": round(min(sims), 4) if sims else None,
            "max": round(max(sims), 4) if sims else None,
        },
        "http_200_previous": sum(row.get("http_status") == 200 for row in previous),
        "http_200_current": sum(row.get("http_status") == 200 for row in current),
        "completed_previous": sum(row.get("generation_status") == "completed" for row in previous),
        "completed_current": sum(row.get("generation_status") == "completed" for row in current),
        "identical_chart_facts_used": sum(
            prev[q].get("chart_facts_used") == curr[q].get("chart_facts_used") for q in common
        ),
        "identical_retrieved_passages": sum(
            prev[q].get("retrieved_passages") == curr[q].get("retrieved_passages") for q in common
        ),
        "identical_chart_snapshots": sum(
            prev[q].get("chart_snapshot") == curr[q].get("chart_snapshot") for q in common
        ),
        "row_differences": [
            {
                "question": q,
                "index_previous": prev[q].get("index"),
                "index_current": curr[q].get("index"),
                "answer_similarity": round(
                    SequenceMatcher(None, normalize(prev[q].get("answer")), normalize(curr[q].get("answer"))).ratio(), 4
                ),
                "chart_facts_same": prev[q].get("chart_facts_used") == curr[q].get("chart_facts_used"),
                "retrieval_same": prev[q].get("retrieved_passages") == curr[q].get("retrieved_passages"),
                "answer_exact_match": normalize(prev[q].get("answer")) == normalize(curr[q].get("answer")),
            }
            for q in common
        ],
        "caveat": "If effective prompt and model/export fingerprints are not recorded in both runs, answer divergence cannot be attributed solely to sampling or model nondeterminism.",
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
