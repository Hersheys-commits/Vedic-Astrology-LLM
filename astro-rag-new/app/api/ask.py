from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterator
from time import perf_counter
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.chart_client.client import fetch_chart
from app.facts import chart_reader as cr
from app.facts.intent import resolve_facts
from app.generation.prompt import build_prompt
from app.generation.qwen_client import (
    _clean_model_output,
    _stream_visible_text,
    verify_and_repair_chart_claims,
)
from app.generation.response_generator import generate_response, stream_response
from app.models.requests import AskRequest
from app.models.responses import AskResponse, ChartFactUsed, RetrievedPassage
from app.retrieval.corpus_store import get_store
from app.retrieval.hybrid import retrieve_for_facts
from app.utils.errors import RAGError

router = APIRouter()
logger = logging.getLogger(__name__)


def _retrieval_signature(passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": passage.get("chunk_id"),
            "score": passage.get("score"),
            "match_type": passage.get("match_type"),
        }
        for passage in passages
    ]


def _generation_usage(generator: Any, passages: list[dict[str, Any]]) -> dict[str, Any]:
    usage = generator.last_diagnostics.get("usage")
    result = dict(usage) if isinstance(usage, dict) else {}
    signature = _retrieval_signature(passages)
    result["retrieval_signature"] = signature
    result["retrieval_signature_sha256"] = hashlib.sha256(
        json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return result


def _prepare_ask_context(req: AskRequest) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    str,
    list[dict[str, Any]],
    str,
    str,
]:
    """Run the shared chart, fact, retrieval, and prompt preparation pipeline."""
    if req.chart_json is not None:
        chart = req.chart_json
    else:
        chart = fetch_chart(
            birth_date=req.birth_date,
            birth_time=req.birth_time,
            location=req.location,
            calc_settings=req.settings,
            as_of_date=req.as_of_date,
        )
    logger.info("Ask request: chart ready; resolving question facts")

    effective_intent_mode = req.intent_mode
    if effective_intent_mode is None and req.response_model in {
        "astro-qwen-lora",
        "astro-qwen-npu",
    }:
        effective_intent_mode = "rules"
    chart_facts, intent_source = resolve_facts(
        req.question, chart, mode=effective_intent_mode
    )
    logger.info(
        "Ask request: resolved %d chart facts; retrieving source passages",
        len(chart_facts),
    )

    passages = retrieve_for_facts(
        facts=chart_facts,
        question=req.question,
        strategy=req.strategy,
        overall_top_k=req.top_k,
        store=get_store(),
        chart=chart,
    )
    logger.info(
        "Ask request: retrieved %d passages; building answer prompt",
        len(passages),
    )
    logger.info("Ask request retrieval signature: %s", _retrieval_signature(passages))
    system_prompt, user_prompt = build_prompt(
        chart, chart_facts, passages, req.question
    )
    return chart, chart_facts, intent_source, passages, system_prompt, user_prompt


def _sse_event(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    started_at = perf_counter()
    logger.info("Ask request started (answer provider: %s)", req.response_model)
    chart, chart_facts, intent_source, passages, system_prompt, user_prompt = (
        _prepare_ask_context(req)
    )

    logger.info("Ask request: answer generation started (provider: %s)", req.response_model)
    answer, generator = generate_response(
        req.response_model, system_prompt, user_prompt
    )
    answer = verify_and_repair_chart_claims(answer, chart, passages)
    logger.info(
        "Ask request completed (provider: %s, elapsed: %.1fs)",
        generator.mode,
        perf_counter() - started_at,
    )

    debug = None
    if req.include_debug:
        debug = {"system_prompt": system_prompt, "user_prompt": user_prompt}

    return AskResponse(
        answer=answer,
        question=req.question,
        chart_facts_used=[ChartFactUsed(type=f["type"], detail=f) for f in chart_facts],
        intent_source=intent_source,
        retrieved_passages=[RetrievedPassage(**p) for p in passages],
        retrieval_strategy=req.strategy,
        chart_snapshot=cr.chart_summary(chart),
        model=generator.model,
        generation_mode=generator.mode,
        debug=debug,
        generation_status=generator.last_diagnostics.get("status"),
        generation_incomplete_reason=generator.last_diagnostics.get("incomplete_reason"),
        generation_usage=_generation_usage(generator, passages),
    )


@router.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Stream early progress, context metadata, answer deltas, and final response."""
    started_at = perf_counter()
    logger.info("Ask stream request started (answer provider: %s)", req.response_model)

    def event_generator() -> Iterator[str]:
        try:
            # Yield before chart-engine/network calls and retrieval. This lets the
            # browser receive an SSE frame immediately instead of waiting for the
            # entire preparation pipeline to finish.
            yield _sse_event(
                "progress",
                {
                    "stage": "preparing",
                    "message": "Preparing chart facts and retrieving references…",
                },
            )
            chart, chart_facts, intent_source, passages, system_prompt, user_prompt = (
                _prepare_ask_context(req)
            )
            fact_models = [ChartFactUsed(type=f["type"], detail=f) for f in chart_facts]
            passage_models = [RetrievedPassage(**passage) for passage in passages]
            chart_snapshot = cr.chart_summary(chart)
            debug = (
                {"system_prompt": system_prompt, "user_prompt": user_prompt}
                if req.include_debug
                else None
            )
            yield _sse_event(
                "progress",
                {
                    "stage": "generating",
                    "message": "References ready; generating the answer…",
                },
            )

            deltas, generator = stream_response(
                req.response_model, system_prompt, user_prompt
            )
            yield _sse_event(
                "metadata",
                {
                    "question": req.question,
                    "chart_facts_used": [fact.model_dump() for fact in fact_models],
                    "intent_source": intent_source,
                    "retrieved_passages": [
                        passage.model_dump() for passage in passage_models
                    ],
                    "retrieval_strategy": req.strategy,
                    "chart_snapshot": chart_snapshot,
                    "model": generator.model,
                    "generation_mode": generator.mode,
                },
            )

            answer_chunks: list[str] = []
            emitted_answer = ""
            for delta in deltas:
                if delta:
                    answer_chunks.append(delta)
                    raw_so_far = "".join(answer_chunks)
                    if (
                        re.search(r"<think\b", raw_so_far, flags=re.IGNORECASE)
                        and not re.search(
                            r"</think\s*>", raw_so_far, flags=re.IGNORECASE
                        )
                    ) or (
                        re.search(
                            r"<tool_call\b", raw_so_far, flags=re.IGNORECASE
                        )
                        and not re.search(
                            r"</tool_call\s*>", raw_so_far, flags=re.IGNORECASE
                        )
                    ):
                        continue
                    safe_raw, _ = _stream_visible_text(raw_so_far)
                    visible = _clean_model_output(safe_raw)
                    if len(visible) > len(emitted_answer):
                        visible_delta = visible[len(emitted_answer):]
                        emitted_answer = visible
                        yield _sse_event("delta", {"delta": visible_delta})

            complete_raw_answer, _ = _stream_visible_text(
                "".join(answer_chunks),
                final=True,
            )
            answer = verify_and_repair_chart_claims(
                _clean_model_output(complete_raw_answer),
                chart,
                passages,
            )
            response = AskResponse(
                answer=answer,
                question=req.question,
                chart_facts_used=fact_models,
                intent_source=intent_source,
                retrieved_passages=passage_models,
                retrieval_strategy=req.strategy,
                chart_snapshot=chart_snapshot,
                model=generator.model,
                generation_mode=generator.mode,
                debug=debug,
                generation_status=generator.last_diagnostics.get("status"),
                generation_incomplete_reason=generator.last_diagnostics.get(
                    "incomplete_reason"
                ),
                generation_usage=_generation_usage(generator, passages),
            )
            logger.info(
                "Ask stream request completed (provider: %s, elapsed: %.1fs)",
                generator.mode,
                perf_counter() - started_at,
            )
            yield _sse_event("replace", {"answer": answer})
            yield _sse_event("done", response.model_dump())
        except RAGError as exc:
            yield _sse_event("error", exc.to_dict())
        except Exception:
            logger.exception("Unhandled error during /ask/stream")
            yield _sse_event(
                "error",
                {
                    "error": {
                        "code": "stream_failed",
                        "message": "Streaming generation failed. Check the server log.",
                    }
                },
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
