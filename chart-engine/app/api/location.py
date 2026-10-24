"""Geocoding endpoints (Nominatim by default).

Review the provider's rate limits and attribution requirements before any
production use, and set GEOCODER_USER_AGENT to a real contact address.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter

from app.config.settings import GEOCODER_TIMEOUT, GEOCODER_URL, GEOCODER_USER_AGENT
from app.models.requests import GeocodeRequest, ReverseGeocodeRequest
from app.models.responses import GeocodeResponse, GeocodeResult
from app.utils.errors import ChartError

router = APIRouter(prefix="/location", tags=["location"])

_HEADERS = {"User-Agent": GEOCODER_USER_AGENT}


@router.post("/geocode", response_model=GeocodeResponse, summary="Address -> candidate coordinates")
def geocode(request: GeocodeRequest) -> GeocodeResponse:
    params = {"q": request.query, "format": "jsonv2", "limit": request.limit, "addressdetails": 1}
    try:
        response = httpx.get(
            f"{GEOCODER_URL}/search", params=params, headers=_HEADERS, timeout=GEOCODER_TIMEOUT
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise ChartError("LOCATION_NOT_FOUND", f"Geocoding request failed: {exc}")

    results = [
        GeocodeResult(
            display_name=item["display_name"],
            latitude=float(item["lat"]),
            longitude=float(item["lon"]),
            type=item.get("type"),
        )
        for item in payload
    ]
    if not results:
        raise ChartError("LOCATION_NOT_FOUND", f"No location matched {request.query!r}.")
    return GeocodeResponse(results=results)


@router.post("/reverse", summary="Coordinates -> display name")
def reverse(request: ReverseGeocodeRequest) -> dict:
    params = {"lat": request.latitude, "lon": request.longitude, "format": "jsonv2"}
    try:
        response = httpx.get(
            f"{GEOCODER_URL}/reverse", params=params, headers=_HEADERS, timeout=GEOCODER_TIMEOUT
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise ChartError("LOCATION_NOT_FOUND", f"Reverse geocoding request failed: {exc}")
    return {
        "display_name": payload.get("display_name"),
        "latitude": request.latitude,
        "longitude": request.longitude,
        "attribution": "Geocoding data \u00a9 OpenStreetMap contributors",
    }


@router.post("/timezone", summary="Coordinates -> IANA timezone (no network call)")
def timezone_lookup(request: ReverseGeocodeRequest) -> dict:
    from app.astronomy.timezone import resolve_timezone

    tz = resolve_timezone(request.latitude, request.longitude)
    return {"timezone": tz, "source": "timezonefinder"}
