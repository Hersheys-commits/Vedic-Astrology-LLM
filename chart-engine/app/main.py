"""FastAPI application wiring."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import chart, health, location
from app.astronomy.swisseph import init_ephemeris
from app.config.settings import ENGINE_VERSION
from app.utils.errors import ChartError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(
    title="Vedic Chart Engine",
    version=ENGINE_VERSION,
    description=(
        "Deterministic Vedic astrology chart engine. Computes facts only: no LLM, "
        "no interpretation, no randomness. Output is designed to be consumed by a "
        "downstream RAG/generation layer."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chart.router)
app.include_router(location.router)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.exception_handler(ChartError)
def chart_error_handler(_: Request, exc: ChartError) -> JSONResponse:
    return JSONResponse(status_code=exc.http_status, content=exc.to_dict())


@app.on_event("startup")
def _startup() -> None:
    mode = init_ephemeris()
    logging.getLogger("chart_engine").info("ephemeris_initialised mode=%s", mode)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Minimal Leaflet test UI so the engine can be driven without a React app."""
    return FileResponse(str(STATIC_DIR / "index.html"))
