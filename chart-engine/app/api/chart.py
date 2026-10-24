"""POST /chart - the only endpoint the RAG layer needs."""

from __future__ import annotations

from fastapi import APIRouter

from app.engine.calculator import generate_chart
from app.models.requests import ChartRequest

router = APIRouter(tags=["chart"])


@router.post("/chart", summary="Compute a deterministic Vedic chart")
def post_chart(request: ChartRequest) -> dict:
    return generate_chart(
        birth_date=request.birth_date,
        birth_time=request.birth_time,
        latitude=request.location.latitude,
        longitude=request.location.longitude,
        location_name=request.location.display_name,
        location_source=request.location.mode,
        timezone_id=request.timezone,
        dst_choice=request.dst_choice,
        settings=request.settings,
        as_of_date=request.as_of_date,
    )
