"""Application settings, loaded from environment variables (prefix ``POLYRAG_``) or ``.env``."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LLMProvider(StrEnum):
    ANTHROPIC = "anthropic"  # Claude API (ANTHROPIC_API_KEY)
    ANTHROPIC_VERTEX = "anthropic-vertex"  # Claude on Google Cloud Vertex AI (ADC)
    LANGCHAIN = "langchain"  # any LangChain chat model, e.g. "google_genai:gemini-2.5-flash"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="POLYRAG_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Data -------------------------------------------------------------------------------
    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"
    index_dir: Path = PROJECT_ROOT / ".polyrag"
    golden_set: Path = PROJECT_ROOT / "eval" / "golden_set.jsonl"
    reports_dir: Path = PROJECT_ROOT / "reports"

    # --- LLM --------------------------------------------------------------------------------
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    # Synthesis model and the model used by the router / graders / SQL agent / verifier.
    llm_model: str = "claude-opus-5-5"
    llm_fast_model: str = "claude-opus-5-5"
    judge_model: str = "claude-opus-5-5"
    # Server-side refusal fallback (Claude API only, not available on Vertex AI).
    llm_refusal_fallback: bool = True
    gcp_project: str | None = None
    gcp_region: str = "global"

    # --- Retrieval --------------------------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Off by default: on the golden set it adds ~1 s per query on CPU without improving
    # recall (see reports/retrieval.md). Enable with Xenova/ms-marco-MiniLM-L-6-v2.
    reranker_model: str | None = None
    qdrant_url: str | None = None  # None -> embedded Qdrant stored in index_dir
    qdrant_api_key: SecretStr | None = None
    collection: str = "polyrag_chunks"
    chunk_size: int = 900
    chunk_overlap: int = 120
    candidates_k: int = 20  # per retriever, before fusion
    top_k: int = 5  # chunks kept per agent after fusion / reranking
    rrf_k: int = 60

    # --- Agents -----------------------------------------------------------------------------
    max_retrieval_rounds: int = Field(default=2, ge=1, le=4)
    max_sql_attempts: int = Field(default=3, ge=1, le=5)
    max_synthesis_attempts: int = Field(default=2, ge=1, le=3)
    sql_max_rows: int = 200

    # --- API --------------------------------------------------------------------------------
    api_key: SecretStr | None = None  # if set, required in the X-API-Key header
    auto_ingest: bool = True  # build the index at startup when it is missing
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    log_level: str = "INFO"
    log_json: bool = False

    @property
    def qdrant_path(self) -> Path:
        return self.index_dir / "qdrant"

    @property
    def chunks_path(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    @property
    def database_path(self) -> Path:
        return self.index_dir / "voltaria.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()


# USD per million tokens (input, output). Used for cost reporting only.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}
