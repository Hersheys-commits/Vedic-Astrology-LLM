from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.evaluation.evaluate_end_to_end import (
    check_expected_facts,
    load_cases,
    normalize_ask_facts,
    parse_json_object,
    retrieval_metrics,
    _aggregate,
    _run_full,
)


def test_curated_dataset_has_twenty_unique_birth_profiles_and_splits():
    cases = load_cases()
    assert len(cases) == 20
    assert sum(case["split"] == "development" for case in cases) == 12
    assert sum(case["split"] == "holdout" for case in cases) == 8
    assert len({case["id"] for case in cases}) == 20


def test_dataset_validator_rejects_duplicate_profiles(tmp_path: Path):
    cases = load_cases()
    cases[1]["birth"] = cases[0]["birth"]
    path = tmp_path / "bad.jsonl"
    path.write_text("\n".join(json.dumps(case) for case in cases), encoding="utf-8")
    with pytest.raises(ValueError, match="distinct birth profile"):
        load_cases(path)


def test_fact_matcher_checks_expected_chart_facts_and_house_lookups():
    case = {
        "expected_facts": [{"type": "planet_house", "planet": "Mars", "house": 10}],
        "expected_houses": [2, 10],
        "requires_current_dasha": True,
    }
    facts = [
        {"type": "planet_house", "planet": "Mars", "house": 10},
        {"type": "house_lookup", "house": 10},
        {"type": "dasha", "mahadasha": "Venus"},
    ]
    result = check_expected_facts(facts, case)
    assert result["pass"] is False
    assert result["missing_houses"] == [2]
    assert result["has_dasha_fact"] is True


def test_ask_fact_normalizer_flattens_chart_fact_response():
    assert normalize_ask_facts([{"type": "planet_house", "detail": {"planet": "Mars", "house": 10}}]) == [
        {"type": "planet_house", "planet": "Mars", "house": 10}
    ]


def test_retrieval_metrics_hit_recall_precision_and_rank():
    result = retrieval_metrics(["gold-a", "gold-b"], ["other", "gold-b", "gold-a"])
    assert result["hit"] == {"1": 0, "3": 1, "5": 1}
    assert result["recall"] == {"1": 0.0, "3": 1.0, "5": 1.0}
    assert result["precision"] == {"1": 0.0, "3": 0.6667, "5": 0.4}
    assert result["reciprocal_rank"] == 0.5


def test_retrieval_metrics_marks_unsupported_question_unscored():
    result = retrieval_metrics([], ["unrelated"])
    assert result["not_scored"]
    assert result["hit"] == {}


def test_judge_parser_accepts_json_and_rejects_invalid_scores():
    valid = {field: 4 for field in (
        "chart_fact_consistency", "evidence_support", "citation_validity",
        "question_relevance", "uncertainty_and_safety",
    )}
    valid.update(critical_errors=[], rationale="Supported.")
    assert parse_json_object("```json\n" + json.dumps(valid) + "\n```")["evidence_support"] == 4
    valid["evidence_support"] = 6
    with pytest.raises(ValueError, match="integer from 1 to 5"):
        parse_json_object(json.dumps(valid))


def test_aggregate_formats_retrieval_metrics_and_case_counts():
    report = _aggregate([
        {"retrieval_metrics": retrieval_metrics(["gold"], ["gold"]), "fact_check": {"pass": True}},
        {"retrieval_metrics": retrieval_metrics([], ["other"]), "fact_check": {"pass": False}},
    ])
    assert report["cases"] == 2
    assert report["retrieval_cases_scored"] == 1
    assert report["hit_at_k"]["5"] == 1.0
    assert report["mean_reciprocal_rank"] == 1.0
    assert report["fact_check_pass_rate"] == 0.5


def test_full_runner_resume_reuses_answer_without_regenerating(monkeypatch, tmp_path: Path):
    import rag.evaluation.evaluate_end_to_end as evaluator

    case = load_cases()[0]
    previous = {
        "id": case["id"], "answer": "Supported. Passage 1.", "chart_id": "chart-1",
        "chart_validation": {"valid": True}, "chart_summary": {}, "intent_source": "rules",
        "generation_mode": "openai", "generation_model": "generator", "generation_status": "complete",
        "generation_usage": {"total_tokens": 20},
        "chart_facts_used": [{"type": "planet_house", "detail": {"type": "planet_house", "planet": "Mars", "house": 10}}],
        "retrieved_passages": [{"chunk_id": "gold", "text": "text", "metadata": {}}],
    }
    monkeypatch.setattr(evaluator.settings, "openai_api_key", "test-key")
    monkeypatch.setattr(evaluator.settings, "eval_judge_model", "judge-model")
    monkeypatch.setattr(evaluator.settings, "openai_model", "generator")

    class Health:
        @staticmethod
        def json():
            return {"generation_mode": "openai"}

        @staticmethod
        def raise_for_status():
            return None

    monkeypatch.setattr(evaluator.requests, "get", lambda *args, **kwargs: Health())
    monkeypatch.setattr(evaluator.requests, "post", lambda *args, **kwargs: pytest.fail("resume should not regenerate"))
    monkeypatch.setattr(evaluator, "get_chart", lambda case: pytest.fail("resume should not recalculate chart"))
    monkeypatch.setattr(evaluator, "_call_judge", lambda *args: ({
        **{field: 5 for field in evaluator.JUDGE_FIELDS}, "critical_errors": [], "rationale": "Supported."
    }, None))

    results = _run_full([case], store=object(), progress_path=tmp_path / "partial.json", resume_results=[previous])
    assert results[0]["fact_check"]["pass"] is True
    assert results[0]["case_pass"] is True
    assert results[0]["generation_model"] == "generator"


def test_ask_request_accepts_optional_as_of_date_and_chart_client_forwards_it(monkeypatch):
    from app.chart_client import client as chart_client
    from app.models.requests import AskRequest, ChartCalcSettings, LocationInput

    request = AskRequest(
        question="What is my current dasha?",
        birth_date="2000-01-02",
        birth_time="12:30:00",
        location=LocationInput(latitude=28.6, longitude=77.2),
    )
    assert request.as_of_date is None
    request.as_of_date = "2026-10-01"
    sent = {}

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"ok": True}

    def fake_post(url, json, timeout):
        sent.update(json)
        return Response()

    monkeypatch.setattr(chart_client.requests, "post", fake_post)
    chart_client.fetch_chart(
        "2000-01-02", "12:30:00",
        LocationInput(latitude=28.6, longitude=77.2),
        ChartCalcSettings(), request.as_of_date,
    )
    assert sent["as_of_date"] == "2026-10-01"

