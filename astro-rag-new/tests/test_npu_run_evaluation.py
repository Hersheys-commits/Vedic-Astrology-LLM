from rag.evaluation.evaluate_npu_run import evaluate_run


def _record(index=1, answer="A supported answer."):
    return {
        "index": index,
        "question": "Question",
        "chart_sha256": "chart-hash",
        "answer": answer,
        "elapsed_seconds": 10.0,
        "generation_status": "completed",
        "provider_mismatch": False,
        "retrieved_passages": [
            {"metadata": {"source": {"book": "Brihat Jataka"}}}
        ],
        "generation_usage": {
            "prompt_sha256": "prompt-hash",
            "model_fingerprint": "model-hash",
            "retrieval_signature_sha256": "retrieval-hash",
            "generation_config": {"do_sample": False},
        },
    }


def test_npu_run_evaluator_measures_citations_leaks_latency_and_metadata_coverage():
    report = evaluate_run([
        _record(answer="Passage 1 — Brihat Jataka supports this."),
        _record(
            index=2,
            answer="A response.\n**Reasoning:** private analysis.",
        ),
    ])

    assert report["run"]["question_count"] == 2
    assert report["run"]["mean_latency_seconds"] == 10
    assert report["run"]["reasoning_leak_count"] == 1
    assert report["run"]["valid_citation_rate"] == 1
    assert report["run"]["prompt_hash_coverage"] == 2
    assert "unsupported-claim and hallucination rates" in report["not_scored"]


def test_npu_run_comparison_only_counts_matching_question_indices():
    current = [_record()]
    baseline = [_record(answer="Different answer.")]

    comparison = evaluate_run(current, baseline)["comparison"]

    assert comparison["paired_question_count"] == 1
    assert comparison["chart_hash_comparable_count"] == 1
    assert comparison["prompt_hash_comparable_count"] == 1
    assert comparison["same_prompt_hash_count"] == 1
    assert comparison["same_model_fingerprint_count"] == 1
    assert comparison["exact_answer_match_count"] == 0


def test_npu_run_comparison_does_not_treat_missing_hashes_as_equal():
    current = _record()
    baseline = _record()
    current["chart_sha256"] = " "
    baseline.pop("chart_sha256")
    current["generation_usage"] = {}
    baseline["generation_usage"] = {}

    comparison = evaluate_run([current], [baseline])["comparison"]

    assert comparison["chart_hash_comparable_count"] == 0
    assert comparison["same_chart_hash_count"] == 0
    assert comparison["prompt_hash_comparable_count"] == 0
    assert comparison["same_prompt_hash_count"] == 0
    assert comparison["model_fingerprint_comparable_count"] == 0
    assert comparison["same_model_fingerprint_count"] == 0
    assert comparison["retrieval_signature_comparable_count"] == 0
    assert comparison["same_retrieval_signature_count"] == 0
