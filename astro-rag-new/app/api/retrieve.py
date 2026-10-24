from __future__ import annotations

from fastapi import APIRouter

from app.models.requests import RetrieveRequest
from app.models.responses import RetrievedPassage, RetrieveResponse
from app.retrieval.corpus_store import get_store
from app.retrieval.hybrid import retrieve_for_fact, retrieve_for_facts

router = APIRouter()


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    """
    Phase-4 style retrieval testing: pass either `chart_facts` (a single
    fact, e.g. {"type": "planet_house", "planet": "Mars", "house": 10}),
    `query` (free text), or both. No chart-engine call, no LLM call --
    isolates retrieval quality from everything downstream, per the
    architecture doc's recommendation to test retrieval separately from
    generation.
    """
    store = get_store()

    if req.chart_facts and req.query:
        results = retrieve_for_facts(
            facts=[req.chart_facts], question=req.query, strategy=req.strategy,
            overall_top_k=req.top_k, store=store,
        )
    elif req.chart_facts:
        results = retrieve_for_fact(req.chart_facts, strategy=req.strategy, top_k=req.top_k, store=store)
    else:
        results = retrieve_for_facts(
            facts=[], question=req.query, strategy=req.strategy, overall_top_k=req.top_k, store=store,
        )

    return RetrieveResponse(
        query=req.query,
        chart_facts=req.chart_facts,
        strategy=req.strategy,
        results=[RetrievedPassage(**r) for r in results],
    )
