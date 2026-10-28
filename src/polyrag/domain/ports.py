"""Ports (interfaces) between the application core and the infrastructure.

Adapters live in ``polyrag.infrastructure``; tests plug in-memory fakes into the same ports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from polyrag.domain.models import Chunk, ScoredChunk, SearchFilter

SchemaT = TypeVar("SchemaT", bound=BaseModel)

Effort = Literal["low", "medium", "high"]


class LLMError(RuntimeError):
    """Raised when a model call fails in a way the agents should handle (refusal, bad output)."""


class LLMUnavailableError(RuntimeError):
    """Raised when the model provider cannot serve requests at all (missing credentials,
    exhausted usage limit). Unlike ``LLMError`` an agent cannot recover from it, so it
    propagates to the caller instead of degrading the answer."""


class LLMConfigurationError(LLMUnavailableError):
    """The provider is not configured (e.g. no credentials found)."""


class QueryError(ValueError):
    """Raised when a SQL query is rejected by the guardrails or fails to execute.

    The message is fed back to the SQL agent so that it can correct its query."""


@runtime_checkable
class LLM(Protocol):
    """Minimal chat-model contract used by the agents."""

    model: str

    async def complete(
        self, *, system: str, prompt: str, effort: Effort = "medium", max_tokens: int = 4096
    ) -> str: ...

    async def structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[SchemaT],
        effort: Effort = "low",
        max_tokens: int = 4096,
    ) -> SchemaT: ...

    def stream(
        self, *, system: str, prompt: str, effort: Effort = "medium", max_tokens: int = 4096
    ) -> AsyncIterator[str]: ...


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    def reset(self, dimension: int) -> None: ...

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    def search(
        self, vector: Sequence[float], *, limit: int, where: SearchFilter | None = None
    ) -> list[ScoredChunk]: ...

    def count(self) -> int: ...


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]: ...


class QueryResult(BaseModel):
    columns: list[str]
    rows: list[list[object]]
    truncated: bool = False

    def to_markdown(self, max_rows: int = 30) -> str:
        if not self.columns:
            return "(no columns)"
        header = "| " + " | ".join(self.columns) + " |"
        separator = "| " + " | ".join("---" for _ in self.columns) + " |"
        body = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in self.rows[:max_rows]]
        lines = [header, separator, *body]
        if not self.rows:
            lines.append("(0 rows)")
        if self.truncated or len(self.rows) > max_rows:
            lines.append(f"(truncated, {len(self.rows)} rows shown at most)")
        return "\n".join(lines)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}" if abs(value) < 1 else f"{value:,.2f}"
    return str(value)


class SqlDatabase(Protocol):
    @property
    def dialect(self) -> str: ...

    def describe(self) -> str:
        """Schema (DDL) plus a few sample rows, used to ground SQL generation."""
        ...

    def execute(self, sql: str, *, max_rows: int = 200) -> QueryResult: ...
