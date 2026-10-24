"""
Central configuration for the RAG + baseline-generation service.

Everything here is overridable via environment variables or a `.env` file
(see `.env.example`). Nothing else in the codebase should read `os.environ`
directly -- go through `settings`.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Service ---
    app_port: int = 8001
    app_host: str = "0.0.0.0"

    # --- Upstream chart-engine ---
    chart_engine_url: str = "http://localhost:8000"
    chart_engine_timeout_seconds: float = 15.0
    evaluation_rag_url: str = "http://localhost:8001"

    # --- OpenAI generation / intent ---
    openai_api_key: str = ""
    openai_model: str = "gpt-5-mini"
    eval_judge_model: str = ""
    generation_max_tokens: int = 5000
    openvino_npu_max_new_tokens: int = 1024
    openai_reasoning_effort: str = "low"

    # --- Corpus / retrieval ---
    corpus_path: Path = PROJECT_ROOT / "rag" / "processed" / "corpus.jsonl"
    openvino_npu_model_dir: Path = PROJECT_ROOT / "models" / "astro-qwen-npu"
    openvino_npu_max_prompt_len: int = 1536
    retrieval_top_k: int = 5
    retrieval_strategy: str = "hybrid"
    per_fact_top_k: int = 5
    semantic_fallback_min_hits: int = 3

    # --- Question understanding ---
    intent_mode: str = "auto"      # auto = OpenAI if a key is configured; rules = never call LLM
    intent_model: str = ""         # empty = OPENAI_MODEL
    intent_max_tokens: int = 1500

    # --- Misc ---
    cors_allow_origins: list[str] = ["*"]


settings = Settings()
