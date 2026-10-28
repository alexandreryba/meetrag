"""Core domain objects.

These types are framework-agnostic: nothing here imports LangGraph, FastAPI or a vendor SDK,
so the retrieval and agent logic can be tested and reused independently of the infrastructure.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Source(StrEnum):
    """Knowledge sources the assistant can route a question to."""

    MEETINGS = "meetings"  # meeting transcripts: who said what, when
    POLICIES = "policies"  # internal HR, IT-security and expense policies
    PRODUCTS = "products"  # commercial offers and product documentation
    SUPPORT = "support"  # customer-support procedures and playbooks
    ANALYTICS = "analytics"  # structured data (business + meeting records), queried with SQL

    @property
    def is_document_source(self) -> bool:
        return self is not Source.ANALYTICS


DOCUMENT_SOURCES: tuple[Source, ...] = tuple(s for s in Source if s.is_document_source)


class Document(BaseModel):
    """A raw document before chunking."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    source: Source
    title: str
    text: str
    path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable unit of text, with the heading path (documents) or the time range
    (transcripts) it belongs to. ``metadata`` carries filterable attributes such as the
    meeting date or the people speaking in the chunk."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    source: Source
    title: str
    section: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def speakers(self) -> list[str]:
        return list(self.metadata.get("speakers", []))

    @property
    def contextualized_text(self) -> str:
        """Text used for indexing: the heading path is prepended so that a chunk such as
        "Up to 3 days per week" still carries the "Remote work policy" context."""
        header = f"{self.title} > {self.section}" if self.section else self.title
        if kind := self.metadata.get("meeting_type"):
            # Canonical English meeting type: an English question about a "steering
            # committee" must also match a French "Comité de pilotage" transcript.
            header = f"{header} · {kind} meeting"
        return f"{header}\n\n{self.text}"


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float


class MeetingFilter(BaseModel):
    """Restrictions on meeting transcripts, as understood from the question by the router."""

    meeting_type: str | None = Field(
        default=None, description="Only this meeting type, e.g. 'steering committee'"
    )
    date_from: str | None = Field(default=None, description="Earliest meeting date, YYYY-MM-DD")
    date_to: str | None = Field(default=None, description="Latest meeting date, YYYY-MM-DD")
    speakers: list[str] = Field(
        default_factory=list, description="People whose own words are asked about"
    )
    latest_only: bool = Field(
        default=False, description="Only the most recent meeting matching the other criteria"
    )

    @property
    def is_empty(self) -> bool:
        return not (
            self.meeting_type or self.date_from or self.date_to or self.speakers or self.latest_only
        )


class SearchFilter(BaseModel):
    """Restrictions applied by every retriever (keyword and vector) before ranking."""

    model_config = ConfigDict(frozen=True)

    source: Source | None = None
    doc_ids: frozenset[str] | None = None  # e.g. the meetings resolved from a date range
    speakers: frozenset[str] | None = None  # keep chunks where one of these people speaks

    def matches(self, chunk: Chunk) -> bool:
        if self.source is not None and chunk.source != self.source:
            return False
        if self.doc_ids is not None and chunk.doc_id not in self.doc_ids:
            return False
        return self.speakers is None or bool(self.speakers.intersection(chunk.speakers))


EvidenceKind = Literal["document", "table"]


class Evidence(BaseModel):
    """A piece of evidence produced by a specialist agent and handed to the synthesizer."""

    kind: EvidenceKind
    source: Source
    title: str
    content: str
    reference: str  # chunk id for documents, SQL statement for tables
    score: float | None = None


class Citation(BaseModel):
    index: int
    source: Source
    title: str
    reference: str
    snippet: str


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0

    def add(self, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.llm_calls += 1
        self.cost_usd += cost_usd


class TraceEvent(BaseModel):
    """A human-readable step of the agent run, surfaced in the API and the demo UI."""

    agent: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class AnswerResult(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    routed_sources: list[Source]
    sources_used: list[Source]
    grounded: bool
    abstained: bool
    evidence: list[Evidence]
    trace: list[TraceEvent]
    usage: TokenUsage
    latency_ms: float
