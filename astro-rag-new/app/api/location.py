from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any

import requests
from fastapi import APIRouter
from pydantic import BaseModel

from app.config.settings import settings
from app.models.requests import GeocodeRequest
from app.utils.errors import RAGError

router = APIRouter(prefix="/location", tags=["location"])

_CACHE_TTL_SECONDS = 60 * 60
_CACHE_MAX_ITEMS = 128
_cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
_cache_lock = threading.Lock()
_upstream_lock = threading.Lock()
_next_upstream_request_at = 0.0


class GeocodeResult(BaseModel):
    display_name: str
    latitude: float
    longitude: float
    type: str | None = None


class GeocodeResponse(BaseModel):
    results: list[GeocodeResult]
    attribution: str = "Geocoding data © OpenStreetMap contributors"


def _cache_key(query: str, limit: int) -> str:
    return f"{limit}:{' '.join(query.casefold().split())}"


def _cached(key: str) -> dict[str, Any] | None:
    now = time.monotonic()
    with _cache_lock:
        item = _cache.get(key)
        if item is None:
            return None
        expires_at, result = item
        if expires_at <= now:
            del _cache[key]
            return None
        _cache.move_to_end(key)
        return result


def _store(key: str, result: dict[str, Any]) -> None:
    with _cache_lock:
        _cache[key] = (time.monotonic() + _CACHE_TTL_SECONDS, result)
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX_ITEMS:
            _cache.popitem(last=False)


@router.post("/geocode", response_model=GeocodeResponse)
def geocode(request: GeocodeRequest) -> GeocodeResponse:
    """Forward an explicit place search to Chart Engine and cache its result."""
    query = request.query.strip()
    if len(query) < 2:
        raise RAGError("INVALID_LOCATION_QUERY", "Enter at least two characters to search.")

    key = _cache_key(query, request.limit)
    result = _cached(key)
    if result is None:
        url = f"{settings.chart_engine_url.rstrip('/')}/location/geocode"
        try:
            # The configured default geocoder is public Nominatim. Serialize
            # cache misses and stay within its one-request-per-second limit.
            global _next_upstream_request_at
            with _upstream_lock:
                delay = _next_upstream_request_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                _next_upstream_request_at = time.monotonic() + 1.0
                response = requests.post(
                    url,
                    json={"query": query, "limit": request.limit},
                    timeout=settings.chart_engine_timeout_seconds,
                )
        except requests.RequestException as exc:
            raise RAGError(
                "LOCATION_LOOKUP_UNAVAILABLE",
                "Could not reach the location search service. Check that Chart Engine is running and try again.",
                status_code=502,
            ) from exc

        if response.status_code != 200:
            try:
                payload = response.json()
                detail = payload.get("error", {}).get("message") or payload.get("detail")
            except (ValueError, AttributeError):
                detail = None
            raise RAGError(
                "LOCATION_LOOKUP_FAILED",
                detail or "The location search service could not complete this search.",
                status_code=502,
            )

        try:
            result = response.json()
            parsed = GeocodeResponse.model_validate(result)
        except (ValueError, TypeError) as exc:
            raise RAGError(
                "LOCATION_LOOKUP_BAD_RESPONSE",
                "The location search service returned an invalid response.",
                status_code=502,
            ) from exc
        if not parsed.results:
            raise RAGError("LOCATION_NOT_FOUND", f"No location matched {query!r}.", status_code=404)
        _store(key, parsed.model_dump())
        result = parsed.model_dump()

    return GeocodeResponse.model_validate(result)


def clear_geocode_cache() -> None:
    """Clear cached lookups. Intended for isolated unit tests."""
    with _cache_lock:
        _cache.clear()
