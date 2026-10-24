from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

Strategy = Literal["hybrid", "metadata_only", "semantic_only"]
IntentMode = Literal["auto", "rules"]          # next to Strategy
ResponseModel = Literal["gpt-5-mini", "astro-qwen-lora", "astro-qwen-npu"]


class LocationInput(BaseModel):
    mode: str = "map_click"
    latitude: float
    longitude: float
    display_name: Optional[str] = None


class ChartCalcSettings(BaseModel):
    ayanamsa: str = "lahiri"
    house_system: str = "whole_sign"
    node_type: str = "mean"


class AskRequest(BaseModel):
    """
    Either supply birth details (this service will call the chart-engine
    itself), or pass an already-computed chart via `chart_json` to skip that
    call -- useful for testing, or when a caller already has the chart from
    a previous turn in a conversation.
    """

    birth_date: Optional[str] = None
    birth_time: Optional[str] = None
    location: Optional[LocationInput] = None
    settings: ChartCalcSettings = Field(default_factory=ChartCalcSettings)
    as_of_date: Optional[str] = Field(
        default=None,
        description="Date used for time-sensitive chart fields such as the running dasha.",
    )

    chart_json: Optional[dict[str, Any]] = None

    question: str

    top_k: int = Field(default=5, ge=1, le=10)
    strategy: Strategy = "hybrid"
    intent_mode: Optional[IntentMode] = None   # None -> INTENT_MODE from settings
    response_model: ResponseModel = "gpt-5-mini"
    include_debug: bool = False

    @model_validator(mode="after")
    def _check_birth_info_or_chart(self):
        if self.chart_json is None:
            missing = [
                name
                for name, val in (
                    ("birth_date", self.birth_date),
                    ("birth_time", self.birth_time),
                    ("location", self.location),
                )
                if val is None
            ]
            if missing:
                raise ValueError(
                    "Provide either `chart_json`, or all of birth_date/birth_time/location "
                    f"(missing: {', '.join(missing)})"
                )
        if not self.question or not self.question.strip():
            raise ValueError("`question` must not be empty")
        return self


class RetrieveRequest(BaseModel):
    """Direct retrieval testing, no chart call and no LLM call."""

    query: Optional[str] = None
    chart_facts: Optional[dict[str, Any]] = None
    strategy: Strategy = "hybrid" 
    top_k: int = Field(default=5, ge=1, le=10)

    @model_validator(mode="after")
    def _need_query_or_facts(self):
        if not self.query and not self.chart_facts:
            raise ValueError("Provide `query`, `chart_facts`, or both")
        return self


class GeocodeRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=200)
    limit: int = Field(default=5, ge=1, le=10)
