from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class RetrievedPassage(BaseModel):
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    match_type: str  # "metadata" | "semantic" | "metadata+semantic"
    matched_fact: Optional[dict[str, Any]] = None


class ChartFactUsed(BaseModel):
    type: str
    detail: dict[str, Any]


class AskResponse(BaseModel):
    answer: str
    question: str
    chart_facts_used: list[ChartFactUsed]
    intent_source: str  # "llm" | "rules" | "rules_after_llm_failure"
    retrieved_passages: list[RetrievedPassage]
    retrieval_strategy: str
    chart_snapshot: dict[str, Any]
    model: str
    generation_mode: str  # "openai" | "mock" | "transformers" | "openvino_npu"
    debug: Optional[dict[str, Any]] = None
    generation_status: Optional[str] = None
    generation_incomplete_reason: Optional[str] = None
    generation_usage: Optional[dict[str, Any]] = None


class RetrieveResponse(BaseModel):
    query: Optional[str]
    chart_facts: Optional[dict[str, Any]]
    strategy: str
    results: list[RetrievedPassage]


class HealthResponse(BaseModel):
    status: str
    corpus_chunks: int
    corpus_categories: dict[str, int]
    chart_engine_url: str
    chart_engine_reachable: Optional[bool]
    generation_mode: str
    model: str
