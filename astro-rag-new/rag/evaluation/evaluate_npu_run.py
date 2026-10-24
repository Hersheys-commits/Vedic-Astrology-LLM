"""Produce a reproducibility and observable-quality report for a saved NPU run."""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

REASONING_RE = re.compile(
    r"(?:^|\n)\s*(?:\*\*)?(?:Reasoning|Chart Factors Examined|"
    r"Internal Verification)\s*:",
    re.IGNORECASE,
)
CITATION_RE = re.compile(
    r"\bPassage\s+(?P<number>\d+)(?:\s*[—–-]\s*(?P<source>[^|,.;)\n]+))?"
)


def _source_name(passage: dict[str, Any]) -> str:
    metadata = passage.get("metadata") or {}
    source = metadata.get("source")
    if isinstance(source, dict):
        return str(source.get("book") or source.get("title") or source.get("book_id") or "")
    return str(source or metadata.get("book_id") or "")


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _answer_metrics(record: dict[str, Any]) -> dict[str, Any]:
    answer = str(record.get("answer") or "")
    passages = record.get("retrieved_passages") or []
    citation_refs = []
    source_checks = []
    for match in CITATION_RE.finditer(answer):
        number = int(match.group("number"))
        valid_index = 1 <= number <= len(passages)
        citation_refs.append(valid_index)
        cited_source = (match.group("source") or "").strip()
        if valid_index and cited_source:
            expected_source = _source_name(passages[number - 1])
            source_checks.append(
                bool(expected_source)
                and (
                    _normalize(expected_source) in _normalize(cited_source)
                    or _normalize(cited_source) in _normalize(expected_source)
                )
            )
        elif cited_source:
            source_checks.append(False)
    return {
        "reasoning_leak": bool(REASONING_RE.search(answer)),
        "word_count": len(answer.split()),
        "citation_count": len(citation_refs),
        "valid_citation_count": sum(citation_refs),
        "source_label_count": len(source_checks),
        "valid_source_label_count": sum(source_checks),
    }


def evaluate_run(
    records: list[dict[str, Any]],
    baseline: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    elapsed = [
        float(record["elapsed_seconds"])
        for record in records
        if isinstance(record.get("elapsed_seconds"), (int, float))
    ]
    record_metrics = [_answer_metrics(record) for record in records]
    total_citations = sum(item["citation_count"] for item in record_metrics)
    total_sources = sum(item["source_label_count"] for item in record_metrics)

    report: dict[str, Any] = {
        "methodology": {
            "citation_validity": "Passage numbers must refer to the ordered retrieved_passages array.",
            "source_validity": "A cited source title is compared with that passage's supplied metadata title.",
            "reasoning_leakage": "Deterministic heading/tag scan of saved final answer text.",
            "interpretive_quality": "Not scored; the saved NPU run has no adjudicated answer labels.",
            "streaming_final_equality": "Not scored; saved result files contain final answers, not rendered delta transcripts.",
        },
        "run": {
            "question_count": len(records),
            "request_error_count": sum("error" in record for record in records),
            "incomplete_generation_count": sum(
                record.get("generation_status") == "incomplete"
                for record in records
            ),
            "provider_mismatch_count": sum(
                bool(record.get("provider_mismatch")) for record in records
            ),
            "mean_latency_seconds": round(statistics.mean(elapsed), 2) if elapsed else None,
            "median_latency_seconds": round(statistics.median(elapsed), 2) if elapsed else None,
            "reasoning_leak_count": sum(item["reasoning_leak"] for item in record_metrics),
            "mean_answer_words": round(
                statistics.mean(item["word_count"] for item in record_metrics), 1
            ) if record_metrics else None,
            "citation_count": total_citations,
            "valid_citation_rate": (
                round(sum(item["valid_citation_count"] for item in record_metrics) / total_citations, 4)
                if total_citations
                else None
            ),
            "source_label_count": total_sources,
            "valid_source_label_rate": (
                round(sum(item["valid_source_label_count"] for item in record_metrics) / total_sources, 4)
                if total_sources
                else None
            ),
            "prompt_hash_coverage": sum(
                bool((record.get("generation_usage") or {}).get("prompt_sha256"))
                for record in records
            ),
            "model_fingerprint_coverage": sum(
                bool((record.get("generation_usage") or {}).get("model_fingerprint"))
                for record in records
            ),
            "generation_config_coverage": sum(
                bool((record.get("generation_usage") or {}).get("generation_config"))
                for record in records
            ),
            "retrieval_signature_hash_coverage": sum(
                bool((record.get("generation_usage") or {}).get("retrieval_signature_sha256"))
                for record in records
            ),
        },
        "not_scored": [
            "chart placement accuracy",
            "house-lord accuracy",
            "occupancy-versus-lordship confusion",
            "dignity accuracy",
            "condition-satisfaction accuracy",
            "unsupported-claim and hallucination rates",
            "answer completeness and question relevance",
            "useful abstention precision",
            "streaming/final text equality",
        ],
    }

    if baseline is not None:
        before = {record.get("index"): record for record in baseline}
        pairs = [
            (record, before[record.get("index")])
            for record in records
            if record.get("index") in before
        ]

        def compare_hashes(get_hash):
            comparable = []
            for current, previous in pairs:
                current_hash = get_hash(current)
                previous_hash = get_hash(previous)
                if (
                    isinstance(current_hash, str)
                    and current_hash.strip()
                    and isinstance(previous_hash, str)
                    and previous_hash.strip()
                ):
                    comparable.append((current_hash.strip(), previous_hash.strip()))
            return {
                "comparable_count": len(comparable),
                "same_count": sum(current == previous for current, previous in comparable),
            }

        chart_hashes = compare_hashes(lambda record: record.get("chart_sha256"))
        prompt_hashes = compare_hashes(
            lambda record: (record.get("generation_usage") or {}).get("prompt_sha256")
        )
        model_fingerprints = compare_hashes(
            lambda record: (record.get("generation_usage") or {}).get("model_fingerprint")
        )
        retrieval_hashes = compare_hashes(
            lambda record: (record.get("generation_usage") or {}).get(
                "retrieval_signature_sha256"
            )
        )
        report["comparison"] = {
            "paired_question_count": len(pairs),
            "same_question_count": sum(
                current.get("question") == previous.get("question")
                for current, previous in pairs
            ),
            "chart_hash_comparable_count": chart_hashes["comparable_count"],
            "same_chart_hash_count": chart_hashes["same_count"],
            "prompt_hash_comparable_count": prompt_hashes["comparable_count"],
            "same_prompt_hash_count": prompt_hashes["same_count"],
            "model_fingerprint_comparable_count": model_fingerprints["comparable_count"],
            "same_model_fingerprint_count": model_fingerprints["same_count"],
            "retrieval_signature_comparable_count": retrieval_hashes["comparable_count"],
            "same_retrieval_signature_count": retrieval_hashes["same_count"],
            "exact_answer_match_count": sum(
                current.get("answer") == previous.get("answer")
                for current, previous in pairs
            ),
        }
    return report


def _load_run(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"{path}: expected a JSON array of result objects")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Saved NPU result JSON.")
    parser.add_argument("--baseline", type=Path, help="Optional earlier run for paired comparison.")
    parser.add_argument("--out", type=Path, help="Optional JSON report output path.")
    args = parser.parse_args()
    try:
        current = _load_run(args.run)
        baseline = _load_run(args.baseline) if args.baseline else None
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    report = evaluate_run(current, baseline)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
