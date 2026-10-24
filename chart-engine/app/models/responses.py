"""Response contracts.

The chart itself is returned as a plain dict: the schema is versioned inside the
payload (`schema_version`) and is intentionally allowed to grow without breaking
older clients. These models document the stable envelope.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class GeocodeResult(BaseModel):
    display_name: str
    latitude: float
    longitude: float
    type: Optional[str] = None


class GeocodeResponse(BaseModel):
    results: list[GeocodeResult]
    attribution: str = "Geocoding data \u00a9 OpenStreetMap contributors"


class HealthResponse(BaseModel):
    status: str
    ephemeris: str
    ephemeris_mode: str
    engine_version: str
    schema_version: str
    pyswisseph_version: str


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


ChartResponse = dict[str, Any]
