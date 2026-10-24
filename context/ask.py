from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from time import perf_counter
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.chart_client.client import fetch_chart
from app.facts import chart_reader as cr
from app.facts.intent import resolve_facts
from app.generation.prompt import build_prompt
from app.generation.qwen_client import _clean_model_output, verify_and_repair_chart_claims
from app.generation.response_generator import generate_response, stream_response
from app.models.requests import AskRequest
from app.models.responses import AskResponse, ChartFactUsed, RetrievedPassage
from app.retrieval.corpus_store import get_store
from app.retrieval.hybrid import retrieve_for_facts
from app.utils.errors import RAGError

router = APIRouter()
logger = logging.getLogger(__name__)


def _prepare_ask_context(req: AskRequest) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    str,
    list[dict[str, Any]],
    str,
    str,
]:
    """Shared deterministic chart + fact resolution + retrieval + prompt step."""
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

    # Pin intent_mode to deterministic 'rules' for local Qwen/NPU models when not explicitly set
    effective_intent_mode = req.intent_mode or ('rules' if req.response_model in ('astro-qwen-lora', 'astro-qwen-npu') else None)
    chart_facts, intent_source = resolve_facts(req.question, chart, mode=effective_intent_mode)
    logger.info("Ask request: resolved %d chart facts; retrieving source passages", len(chart_facts))

    store = get_store()
    passages = retrieve_for_facts(
        facts=chart_facts,
        question=req.question,
        strategy=req.strategy,
        overall_top_k=req.top_k,
        store=store,
    )
    logger.info("Ask request: retrieved %d passages; building answer prompt", len(passages))

    system_prompt, user_prompt = build_prompt(chart, chart_facts, passages, req.question)
    return chart, chart_facts, intent_source, passages, system_prompt, user_prompt


def _sse_event(event: str, payload: dict[str, Any]) -> str:
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
    if chart and (chart.get('planets') or chart.get('houses')):
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
        generation_usage=generator.last_diagnostics.get("usage"),
    )


@router.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """
    Server-Sent Events (SSE) streaming endpoint for `/ask`.
    Emits:
      - `event: metadata` immediately after retrieval (with chart facts & passages)
      - `event: delta` for each generated token chunk (`{"delta": "..."}`)
      - `event: done` with the complete `AskResponse` JSON object
      - `event: error` if generation fails mid-stream
    """
    started_at = perf_counter()
    logger.info("Ask stream request started (answer provider: %s)", req.response_model)
    chart, chart_facts, intent_source, passages, system_prompt, user_prompt = (
        _prepare_ask_context(req)
    )
    facts_models = [ChartFactUsed(type=f["type"], detail=f) for f in chart_facts]
    passage_models = [RetrievedPassage(**p) for p in passages]
    snapshot = cr.chart_summary(chart)
    debug = (
        {"system_prompt": system_prompt, "user_prompt": user_prompt}
        if req.include_debug
        else None
    )

    def _event_generator() -> Iterator[str]:
        try:
            token_iter, generator = stream_response(
                req.response_model, system_prompt, user_prompt
            )
            yield _sse_event(
                "metadata",
                {
                    "question": req.question,
                    "chart_facts_used": [f.model_dump() for f in facts_models],
                    "intent_source": intent_source,
                    "retrieved_passages": [p.model_dump() for p in passage_models],
                    "retrieval_strategy": req.strategy,
                    "chart_snapshot": snapshot,
                    "model": generator.model,
                    "generation_mode": generator.mode,
                },
            )
            chunks: list[str] = []
            for delta in token_iter:
                if delta:
                    chunks.append(delta)
                    yield _sse_event("delta", {"delta": delta})

            final_answer = verify_and_repair_chart_claims(
                _clean_model_output("".join(chunks)), chart, passages
            )
            logger.info(
                "Ask stream request completed (provider: %s, elapsed: %.1fs)",
                generator.mode,
                perf_counter() - started_at,
            )
            response_obj = AskResponse(
                answer=final_answer,
                question=req.question,
                chart_facts_used=facts_models,
                intent_source=intent_source,
                retrieved_passages=passage_models,
                retrieval_strategy=req.strategy,
                chart_snapshot=snapshot,
                model=generator.model,
                generation_mode=generator.mode,
                debug=debug,
                generation_status=generator.last_diagnostics.get("status"),
                generation_incomplete_reason=generator.last_diagnostics.get("incomplete_reason"),
                generation_usage=generator.last_diagnostics.get("usage"),
            )
            yield _sse_event("done", response_obj.model_dump())
        except RAGError as exc:
            yield _sse_event("error", exc.to_dict())
        except Exception as exc:
            logger.exception("Unhandled error during /ask/stream")
            yield _sse_event(
                "error",
                {"error": {"code": "stream_failed", "message": str(exc)}},
            )

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )