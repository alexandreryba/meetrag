"""Graph state and the structured outputs exchanged with the models."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field

from meetrag.domain.models import Citation, Evidence, MeetingFilter, Source, TraceEvent


class RouteStep(BaseModel):
    source: Source = Field(description="Knowledge source to consult")
    query: str = Field(description="Self-contained query for this source")
    meeting_filter: MeetingFilter | None = Field(
        default=None, description="Meetings source only: restrict to some meetings or speakers"
    )


class RoutePlan(BaseModel):
    reasoning: str = Field(description="One sentence explaining the routing decision")
    steps: list[RouteStep] = Field(
        description="Sources to consult; empty when the question is out of scope"
    )


class GradeResult(BaseModel):
    relevant_ids: list[str] = Field(description="Ids of the useful passages, e.g. ['P1', 'P3']")
    sufficient: bool = Field(description="True if the useful passages fully answer the query")
    better_query: str | None = Field(
        default=None, description="Improved search query when the passages are insufficient"
    )


class SqlDraft(BaseModel):
    sql: str = Field(description="A single read-only SQLite SELECT statement")
    rationale: str = Field(description="One sentence on how the query answers the question")


class Verdict(BaseModel):
    grounded: bool = Field(description="True if every claim is supported by the evidence")
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Claims that the evidence does not support"
    )


class GraphState(TypedDict, total=False):
    question: str
    plan: RoutePlan
    # Specialist agents run in parallel; reducers merge their outputs.
    evidence: Annotated[list[Evidence], operator.add]
    trace: Annotated[list[TraceEvent], operator.add]
    answer: str
    citations: list[Citation]
    abstained: bool
    grounded: bool
    feedback: str | None
    synthesis_attempts: int


class AgentTask(TypedDict):
    """Payload sent (with ``Send``) to one specialist agent."""

    question: str
    step: RouteStep
