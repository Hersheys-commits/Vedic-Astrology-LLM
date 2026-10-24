from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import ask, corpus, health, location, retrieve
from app.config.settings import settings
from app.retrieval.corpus_store import get_store
from app.utils.errors import RAGError

app = FastAPI(
    title="Astro RAG + Baseline Generation",
    description=(
        "Phase 3-5 service: takes birth details + a question, calls the "
        "chart-engine for a structured chart, retrieves relevant doctrine "
        "passages from a tagged corpus, and generates a grounded answer."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _force_json_utf8(request: Request, call_next):
    response = await call_next(request)
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("application/json") and "charset=" not in content_type.lower():
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response


@app.on_event("startup")
def _startup() -> None:
    # Load the corpus once at startup, same rationale as the chart-engine's
    # init_ephemeris() call -- don't pay that cost on the first real request.
    get_store()


@app.exception_handler(RAGError)
def _handle_rag_error(_: Request, exc: RAGError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


app.include_router(ask.router, tags=["ask"])
app.include_router(location.router)
app.include_router(retrieve.router, tags=["retrieval"])
app.include_router(corpus.router, tags=["corpus"])
app.include_router(health.router, tags=["health"])

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
