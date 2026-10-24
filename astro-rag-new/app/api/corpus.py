from __future__ import annotations

from fastapi import APIRouter

from app.retrieval.corpus_store import get_store

router = APIRouter()


@router.get("/corpus/stats")
def corpus_stats() -> dict:
    store = get_store()
    return store.stats()
