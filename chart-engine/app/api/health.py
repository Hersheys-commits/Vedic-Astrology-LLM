"""GET /health - deployment and ephemeris diagnostics."""

from __future__ import annotations

from fastapi import APIRouter

from app.astronomy.swisseph import ephemeris_mode, swiss_ephemeris_version
from app.config.settings import ENGINE_VERSION, SCHEMA_VERSION
from app.models.responses import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    mode = ephemeris_mode()
    return HealthResponse(
        status="ok",
        ephemeris="available",
        ephemeris_mode=mode,
        engine_version=ENGINE_VERSION,
        schema_version=SCHEMA_VERSION,
        pyswisseph_version=swiss_ephemeris_version(),
    )
