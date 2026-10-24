"""
Talks to the chart-engine project (a separate service, default localhost:8000)
over HTTP. This is the ONLY file in this service that knows the chart-engine's
request shape -- everything downstream just works with the parsed chart dict.
"""
from __future__ import annotations

from typing import Any

import requests

from app.config.settings import settings
from app.models.requests import ChartCalcSettings, LocationInput
from app.utils.errors import RAGError


def fetch_chart(
    birth_date: str,
    birth_time: str,
    location: LocationInput,
    calc_settings: ChartCalcSettings,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    payload = {
        "birth_date": birth_date,
        "birth_time": birth_time,
        "location": location.model_dump(),
        "settings": calc_settings.model_dump(),
    }
    if as_of_date is not None:
        payload["as_of_date"] = as_of_date
    url = f"{settings.chart_engine_url.rstrip('/')}/chart"
    try:
        resp = requests.post(url, json=payload, timeout=settings.chart_engine_timeout_seconds)
    except requests.RequestException as exc:
        raise RAGError(
            "CHART_ENGINE_UNREACHABLE",
            f"Could not reach chart engine at {url}: {exc}",
            status_code=502,
        ) from exc

    if resp.status_code != 200:
        raise RAGError(
            "CHART_ENGINE_ERROR",
            f"Chart engine returned HTTP {resp.status_code}: {resp.text[:500]}",
            status_code=502,
        )

    try:
        return resp.json()
    except ValueError as exc:
        raise RAGError(
            "CHART_ENGINE_BAD_RESPONSE",
            f"Chart engine response was not valid JSON: {exc}",
            status_code=502,
        ) from exc


def ping_chart_engine() -> bool:
    """Best-effort health check, used by GET /health. Never raises."""
    url = f"{settings.chart_engine_url.rstrip('/')}/health"
    try:
        resp = requests.get(url, timeout=3.0)
        return resp.status_code == 200
    except requests.RequestException:
        return False
