"""Run the curated 20-case chart -> retrieval -> answer evaluation.

Full run (requires both services, OPENAI_API_KEY and a separate
EVAL_JUDGE_MODEL):
    python rag/evaluation/evaluate_end_to_end.py --split dev

Offline-generation retrieval run (requires both services, no model calls):
    python rag/evaluation/evaluate_end_to_end.py --mode retrieval-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import settings  # noqa: E402
from app.facts.intent import resolve_facts  # noqa: E402
from app.retrieval.corpus_store import get_store  # noqa: E402
from app.retrieval.hybrid import retrieve_for_facts  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "rag" / "evaluation" / "end_to_end_gold.jsonl"
REPORT_DIR = PROJECT_ROOT / "rag" / "evaluation" / "reports"
JUDGE_FIELDS = (
    "chart_fact_consistency",
    "evidence_support",
    "citation_validity",
    "question_relevance",
    "uncertainty_and_safety",
)


def load_cases(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [case.get("id") for case in cases]
    if len(cases) != 20:
        raise ValueError(f"Expected exactly 20 evaluation cases; found {len(cases)}")
    if len(set(ids)) != len(ids):
        raise ValueError("Evaluation case IDs must be unique")
    if sum(case.get("split") == "development" for case in cases) != 12:
        raise ValueError("Expected 12 development cases")
    if sum(case.get("split") == "holdout" for case in cases) != 8:
        raise ValueError("Expected 8 holdout cases")
    profiles = [
        (case["birth"]["birth_date"], case["birth"]["birth_time"], case["birth"]["location"]["latitude"], case["birth"]["location"]["longitude"])
        for case in cases
    ]
    if len(set(profiles)) != len(profiles):
        raise ValueError("Each evaluation case must use a distinct birth profile")
    if len({case["birth"]["birth_date"] for case in cases}) != 20:
        raise ValueError("Each evaluation case must use a distinct birth date")
    if len({case["birth"]["birth_time"] for case in cases}) != 20:
        raise ValueError("Each evaluation case must use a distinct birth time")
    if len({(case["birth"]["location"]["latitude"], case["birth"]["location"]["longitude"]) for case in cases}) != 20:
        raise ValueError("Each evaluation case must use a distinct birth location")
    for case in cases:
        for field in ("question", "birth", "settings", "as_of_date", "expected_facts", "evidence", "expected_answer_points", "must_not_claim"):
            if field not in case:
                raise ValueError(f"{case.get('id', '<unknown>')}: missing required field {field}")
        try:
            date.fromisoformat(case["birth"]["birth_date"])
            time.fromisoformat(case["birth"]["birth_time"])
            date.fromisoformat(case["as_of_date"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{case['id']}: birth date, birth time and as_of_date must be ISO formatted") from exc
        location = case["birth"]["location"]
        if not (-90 <= location["latitude"] <= 90 and -180 <= location["longitude"] <= 180):
            raise ValueError(f"{case['id']}: location coordinates are outside valid ranges")
        for evidence in case["evidence"]:
            for field in ("chunk_id", "book_id", "chapter", "pdf_page"):
                if field not in evidence:
                    raise ValueError(f"{case['id']}: evidence is missing {field}")
    return cases


def fact_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def check_expected_facts(actual: list[dict[str, Any]], case: dict[str, Any]) -> dict[str, Any]:
    expected = case.get("expected_facts", [])
    missing = [exp for exp in expected if not any(fact_matches(found, exp) for found in actual)]
    expected_houses = set(case.get("expected_houses", []))
    actual_houses = {f.get("house") for f in actual if f.get("type") == "house_lookup"}
    missing_houses = sorted(expected_houses - actual_houses)
    needs_dasha = bool(case.get("requires_current_dasha"))
    has_dasha = any(f.get("type") == "dasha" for f in actual)
    return {
        "pass": not missing and not missing_houses and (not needs_dasha or has_dasha),
        "missing_expected_facts": missing,
        "expected_houses": sorted(expected_houses),
        "missing_houses": missing_houses,
        "requires_current_dasha": needs_dasha,
        "has_dasha_fact": has_dasha,
    }


def normalize_ask_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten the API's ChartFactUsed {type, detail} response into fact records."""
    normalized = []
    for fact in facts:
        detail = fact.get("detail") if isinstance(fact, dict) else None
        if isinstance(detail, dict):
            normalized.append({**detail, "type": fact.get("type", detail.get("type"))})
        elif isinstance(fact, dict):
            normalized.append(fact)
    return normalized


def retrieval_metrics(expected_ids: list[str], retrieved_ids: list[str], ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, Any]:
    if not expected_ids:
        return {"not_scored": "case has no expected relevant corpus passage", "hit": {}, "recall": {}, "precision": {}, "reciprocal_rank": 0.0}
    expected = set(expected_ids)
    first_rank = next((i for i, chunk_id in enumerate(retrieved_ids, start=1) if chunk_id in expected), None)
    return {
        "hit": {str(k): int(any(chunk_id in expected for chunk_id in retrieved_ids[:k])) for k in ks},
        "recall": {str(k): round(len(expected & set(retrieved_ids[:k])) / len(expected), 4) for k in ks},
        "precision": {str(k): round(len(expected & set(retrieved_ids[:k])) / k, 4) for k in ks},
        "reciprocal_rank": round(1 / first_rank, 4) if first_rank else 0.0,
    }


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE)
    try:
        result = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            raise ValueError("Judge response did not contain a JSON object")
        result = json.loads(match.group(0))
    if not isinstance(result, dict):
        raise ValueError("Judge response must be a JSON object")
    for field in JUDGE_FIELDS:
        score = result.get(field)
        if not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"Judge field {field} must be an integer from 1 to 5")
    if not isinstance(result.get("critical_errors", []), list):
        raise ValueError("Judge critical_errors must be a list")
    return result


def _chart_payload(case: dict[str, Any]) -> dict[str, Any]:
    birth = case["birth"]
    return {
        "birth_date": birth["birth_date"],
        "birth_time": birth["birth_time"],
        "location": birth["location"],
        "settings": case["settings"],
        "as_of_date": case["as_of_date"],
    }


def get_chart(case: dict[str, Any]) -> dict[str, Any]:
    url = f"{settings.chart_engine_url.rstrip('/')}/chart"
    response = requests.post(url, json=_chart_payload(case), timeout=settings.chart_engine_timeout_seconds)
    response.raise_for_status()
    return response.json()


def _evidence_rows(case: dict[str, Any], store) -> list[dict[str, Any]]:
    rows = []
    for expected in case["evidence"]:
        chunk = store.chunk(expected["chunk_id"])
        if not chunk:
            raise ValueError(f"{case['id']}: expected chunk not found: {expected['chunk_id']}")
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", {}) or {}
        for key in ("book_id", "chapter_no"):
            expected_key = "book_id" if key == "book_id" else "chapter"
            if expected.get(expected_key) is not None and expected.get(expected_key) != metadata.get(key):
                raise ValueError(f"{case['id']}: {expected['chunk_id']} {expected_key} metadata mismatch")
        if expected.get("pdf_page") is not None and source.get("pdf_page_start") != expected["pdf_page"]:
            raise ValueError(f"{case['id']}: {expected['chunk_id']} PDF page metadata mismatch")
        rows.append({
            **expected,
            "text": chunk.get("text", ""),
            "chapter_title": metadata.get("chapter_title"),
            "chunk_type": metadata.get("chunk_type"),
        })
    return rows


def _call_judge(case: dict[str, Any], answer: str, actual_facts: list[dict[str, Any]], passages: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for answer judging")
    if not settings.eval_judge_model:
        raise RuntimeError("EVAL_JUDGE_MODEL is required for a full answer-quality run")
    if settings.eval_judge_model == settings.openai_model:
        raise RuntimeError("EVAL_JUDGE_MODEL must be different from OPENAI_MODEL")

    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    evidence = [
        {"chunk_id": p["chunk_id"], "source": p.get("metadata", {}).get("source"), "chapter": p.get("metadata", {}).get("chapter_no"), "text": p.get("text", "")}
        for p in passages
    ]
    judge_input = {
        "question": case["question"],
        "expected_answer_points": case["expected_answer_points"],
        "must_not_claim": case["must_not_claim"],
        "expect_abstention": case.get("expect_abstention", False),
        "gold_source_evidence": _evidence_rows(case, get_store()),
        "chart_facts_used": actual_facts,
        "retrieved_passages": evidence,
        "generated_answer": answer,
    }
    schema = {
        "type": "object",
        "properties": {
            **{field: {"type": "integer", "minimum": 1, "maximum": 5} for field in JUDGE_FIELDS},
            "critical_errors": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "required": [*JUDGE_FIELDS, "critical_errors", "rationale"],
        "additionalProperties": False,
    }
    response = client.responses.create(
        model=settings.eval_judge_model,
        instructions=(
            "Evaluate this generated Vedic astrology answer against the supplied chart facts, retrieved evidence, "
            "gold source passages, expected points, and prohibited claims. You are an evaluator, not an astrologer. "
            "Do not treat an answer as correct merely because it sounds plausible. Check each astrological claim against "
            "the supplied text and chart. The `expect_abstention` cases should decline unsupported or high-stakes "
            "conclusions. Score each dimension from 1 (fails) to 5 (strong). A critical error includes a fabricated "
            "chart fact, unsupported decisive prediction, invalid attribution, or violating a prohibited claim. Keep rationale under 50 words and return only the schema."
        ),
        input=json.dumps(judge_input, ensure_ascii=False),
        text={"format": {"type": "json_schema", "name": "answer_evaluation", "strict": True, "schema": schema}},
        max_output_tokens=3000,
    )
    result = parse_json_object(response.output_text or "")
    usage = response.usage.model_dump() if getattr(response, "usage", None) else None
    return result, usage


def _citation_indexes(answer: str, passage_count: int) -> dict[str, Any]:
    cited = [int(n) for n in re.findall(r"\bPassage\s+(\d+)\b", answer, flags=re.IGNORECASE)]
    return {
        "passage_numbers": cited,
        "all_references_resolve": all(1 <= n <= passage_count for n in cited) if cited else None,
    }


def _run_retrieval_only(cases: list[dict[str, Any]], store) -> list[dict[str, Any]]:
    results = []
    for case in cases:
        chart = get_chart(case)
        facts, fact_source = resolve_facts(case["question"], chart, mode="rules")
        passages = retrieve_for_facts(facts, case["question"], strategy="hybrid", overall_top_k=5, store=store)
        actual_ids = [p["chunk_id"] for p in passages]
        fact_check = check_expected_facts(facts, case)
        retrieved = [{"chunk_id": p["chunk_id"], "score": p["score"], "match_type": p["match_type"], "metadata": p["metadata"], "text": p["text"]} for p in passages]
        results.append({
            "id": case["id"], "split": case["split"], "category": case["category"], "question": case["question"],
            "chart_id": chart.get("chart_id"), "chart_validation": chart.get("validation"), "intent_source": fact_source,
            "chart_facts_used": facts, "fact_check": fact_check, "retrieved_passages": retrieved,
            "retrieval_metrics": retrieval_metrics([e["chunk_id"] for e in case["evidence"]], actual_ids),
            "answer": None, "judge": None,
        })
    return results


def _run_full(
    cases: list[dict[str, Any]],
    store,
    progress_path: Path | None = None,
    resume_results: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    missing = [name for name, configured in (
        ("OPENAI_API_KEY", bool(settings.openai_api_key)),
        ("EVAL_JUDGE_MODEL", bool(settings.eval_judge_model)),
    ) if not configured]
    if missing:
        raise RuntimeError(
            "Full answer evaluation requires configuration for: " + ", ".join(missing)
            + ". Set these in the private Astro RAG .env file."
        )
    if settings.eval_judge_model == settings.openai_model:
        raise RuntimeError("EVAL_JUDGE_MODEL must be different from OPENAI_MODEL")

    rag_url = settings.evaluation_rag_url.rstrip("/")
    health = requests.get(f"{rag_url}/health", timeout=10)
    health.raise_for_status()
    if health.json().get("generation_mode") != "openai":
        raise RuntimeError("Astro RAG is in mock mode; configure OPENAI_API_KEY before full generation evaluation")

    results = []
    resume_by_id = {result["id"]: result for result in (resume_results or [])}
    for case in cases:
        previous = resume_by_id.get(case["id"])
        if previous and previous.get("answer") is not None:
            chart_id = previous.get("chart_id")
            chart_validation = previous.get("chart_validation")
            chart_summary = previous.get("chart_summary")
            intent_source = previous.get("intent_source")
            generation_mode = previous.get("generation_mode")
            generation_model = previous.get("generation_model")
            generation_status = previous.get("generation_status")
            generation_usage = previous.get("generation_usage")
            facts = previous.get("chart_facts_used", [])
            passages = previous.get("retrieved_passages", [])
            answer = previous["answer"]
        else:
            chart = get_chart(case)
            payload = {
                **_chart_payload(case),
                "question": case["question"],
                "strategy": "hybrid",
                "top_k": 5,
                "intent_mode": case.get("intent_mode", "rules"),
            }
            response = requests.post(f"{rag_url}/ask", json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            chart_id = chart.get("chart_id")
            chart_validation = chart.get("validation")
            chart_summary = data.get("chart_snapshot")
            intent_source = data.get("intent_source")
            generation_mode = data.get("generation_mode")
            generation_model = data.get("model")
            generation_status = data.get("generation_status")
            generation_usage = data.get("generation_usage")
            facts = data["chart_facts_used"]
            passages = data["retrieved_passages"]
            answer = data["answer"]
        facts = normalize_ask_facts(facts)
        judge_error = None
        try:
            judge, judge_usage = _call_judge(case, answer, facts, passages)
        except Exception as exc:
            judge, judge_usage = None, None
            judge_error = f"{type(exc).__name__}: {str(exc)[:500]}"
        expected_ids = [e["chunk_id"] for e in case["evidence"]]
        actual_ids = [p["chunk_id"] for p in passages]
        fact_check = check_expected_facts(facts, case)
        citations = _citation_indexes(answer, len(passages))
        judge_pass = bool(judge) and all(judge[field] >= 4 for field in JUDGE_FIELDS) and not judge.get("critical_errors")
        results.append({
            "id": case["id"], "split": case["split"], "category": case["category"], "question": case["question"],
            "chart_id": chart_id, "chart_validation": chart_validation,
            "chart_summary": chart_summary, "intent_source": intent_source,
            "generation_mode": generation_mode, "generation_model": generation_model,
            "generation_status": generation_status, "generation_usage": generation_usage,
            "chart_facts_used": facts, "fact_check": fact_check, "retrieved_passages": passages,
            "retrieval_metrics": retrieval_metrics(expected_ids, actual_ids), "answer": answer,
            "answer_citations": citations, "judge": judge, "judge_model": settings.eval_judge_model,
            "judge_usage": judge_usage, "judge_error": judge_error,
            "case_pass": fact_check["pass"] and judge_pass and citations["all_references_resolve"] is not False,
        })
        if progress_path:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(json.dumps({
                "suite": "astro-rag-end-to-end-v1", "mode": "full", "split": case["split"],
                "status": "running", "cases_completed": len(results), "cases": results,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in results if r.get("retrieval_metrics", {}).get("hit")]
    total = len(results) or 1
    agg: dict[str, Any] = {
        "cases": len(results),
        "fact_check_pass_rate": round(sum(bool(r.get("fact_check", {}).get("pass")) for r in results) / total, 4),
        "retrieval_cases_scored": len(scored),
        "hit_at_k": {}, "recall_at_k": {}, "mean_reciprocal_rank": 0.0,
    }
    for k in ("1", "3", "5"):
        agg["hit_at_k"][k] = round(sum(r["retrieval_metrics"]["hit"][k] for r in scored) / len(scored), 4) if scored else None
        agg["recall_at_k"][k] = round(sum(r["retrieval_metrics"]["recall"][k] for r in scored) / len(scored), 4) if scored else None
    agg["mean_reciprocal_rank"] = round(sum(r["retrieval_metrics"]["reciprocal_rank"] for r in scored) / len(scored), 4) if scored else None
    judged = [r for r in results if r.get("judge")]
    agg["judge_error_cases"] = sum(bool(r.get("judge_error")) for r in results)
    if judged:
        agg["judge_mean_scores"] = {
            field: round(sum(r["judge"][field] for r in judged) / len(judged), 3) for field in JUDGE_FIELDS
        }
    answer_results = [r for r in results if r.get("judge") or r.get("judge_error")]
    if answer_results:
        agg["answer_case_pass_rate"] = round(sum(bool(r.get("case_pass")) for r in answer_results) / len(answer_results), 4)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("development", "holdout", "all"), default="development")
    parser.add_argument("--mode", choices=("full", "retrieval-only"), default="full")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--case-id", type=str, default=None, help="Run one case for focused debugging")
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume full mode from a partial JSON report, reusing completed answers")
    args = parser.parse_args()

    cases = load_cases()
    if args.split != "all":
        cases = [case for case in cases if case["split"] == args.split]
    if args.case_id:
        cases = [case for case in cases if case["id"] == args.case_id]
        if not cases:
            parser.error(f"No evaluation case found for --case-id {args.case_id!r} in the selected split")
    store = get_store()
    out = args.out or REPORT_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{args.split}_{args.mode}.json"
    progress_path = out.with_suffix(".partial.json")
    resume_results = None
    if args.resume_from:
        if args.mode != "full":
            parser.error("--resume-from is only supported in full mode")
        resume_report = json.loads(args.resume_from.read_text(encoding="utf-8"))
        resume_results = resume_report.get("cases", [])
    results = _run_retrieval_only(cases, store) if args.mode == "retrieval-only" else _run_full(cases, store, progress_path, resume_results)
    corpus_hash = hashlib.sha256((PROJECT_ROOT / "rag" / "processed" / "corpus.jsonl").read_bytes()).hexdigest()
    dataset_hash = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
    prompt_hash = hashlib.sha256((PROJECT_ROOT / "app" / "generation" / "prompt.py").read_bytes()).hexdigest()
    report = {
        "suite": "astro-rag-end-to-end-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "split": args.split,
        "generation_model": settings.openai_model if args.mode == "full" else None,
        "judge_model": settings.eval_judge_model if args.mode == "full" else None,
        "chart_engine_url": settings.chart_engine_url,
        "rag_url": settings.evaluation_rag_url if args.mode == "full" else None,
        "corpus_sha256": corpus_hash,
        "dataset_sha256": dataset_hash,
        "generation_prompt_sha256": prompt_hash if args.mode == "full" else None,
        "overall": _aggregate(results),
        "by_split": {
            split: _aggregate([result for result in results if result.get("split") == split])
            for split in ("development", "holdout")
            if any(result.get("split") == split for result in results)
        },
        "cases": results,
        "evaluation_note": "Automated judge scores are diagnostic. Review source evidence and high-impact failures before accepting changes.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress_path.exists():
        progress_path.unlink()
    print(json.dumps({"overall": report["overall"], "by_split": report["by_split"], "report": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
