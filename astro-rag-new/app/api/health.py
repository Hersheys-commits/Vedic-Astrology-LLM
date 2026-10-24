from __future__ import annotations

from fastapi import APIRouter

from app.chart_client.client import ping_chart_engine
from app.config.settings import settings
from app.generation.llm_client import get_llm_client
from app.models.responses import HealthResponse
from app.retrieval.corpus_store import get_store

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    store = get_store()
    stats = store.stats()
    categories = {k.replace("distinct_", ""): v for k, v in stats.items() if k != "total_chunks"}
    llm = get_llm_client()

    return HealthResponse(
        status="ok" if store.is_loaded() else "corpus_not_loaded",
        corpus_chunks=stats.get("total_chunks", 0),
        corpus_categories=categories,
        chart_engine_url=settings.chart_engine_url,
        chart_engine_reachable=ping_chart_engine(),
        generation_mode=llm.mode,
        model=llm.model,
    )
