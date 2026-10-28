"""Document agent: corrective retrieval over one document source.

retrieve -> grade the passages -> (if insufficient) rewrite the query and retrieve again.
Grading removes distractors before synthesis; rewriting recovers from vocabulary mismatch.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from polyrag.agents.prompts import GRADER_SYSTEM
from polyrag.agents.state import GradeResult
from polyrag.domain.models import Evidence, ScoredChunk, SearchFilter, Source, TraceEvent
from polyrag.domain.ports import LLMError

if TYPE_CHECKING:
    from polyrag.agents.state import AgentTask
    from polyrag.domain.ports import LLM
    from polyrag.retrieval.hybrid import HybridRetriever
    from polyrag.retrieval.meetings import MeetingCatalog

logger = logging.getLogger(__name__)


def _format_passages(candidates: list[ScoredChunk]) -> str:
    return "\n\n".join(
        f'<passage id="P{i}" title="{c.chunk.title}" section="{c.chunk.section}">\n'
        f"{c.chunk.text}\n</passage>"
        for i, c in enumerate(candidates, start=1)
    )


class DocumentAgent:
    name = "document_agent"

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        llm: LLM,
        catalog: MeetingCatalog | None = None,
        top_k: int = 5,
        max_rounds: int = 2,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.catalog = catalog
        self.top_k = top_k
        self.max_rounds = max_rounds

    async def __call__(self, state: AgentTask) -> dict[str, Any]:
        step = state["step"]
        agent = f"{step.source.value}_agent"
        query = step.query
        kept: dict[str, ScoredChunk] = {}
        trace: list[TraceEvent] = []
        where = SearchFilter(source=step.source)
        if step.source is Source.MEETINGS and self.catalog is not None:
            where, note = self.catalog.resolve(step.meeting_filter)
            if note:
                trace.append(TraceEvent(agent=agent, message=f"Meeting filter: {note}"))

        for round_number in range(1, self.max_rounds + 1):
            hits = await asyncio.to_thread(
                self.retriever.retrieve, query, top_k=self.top_k, where=where
            )
            candidates = [hit for hit in hits if hit.chunk.chunk_id not in kept]
            if not candidates:
                trace.append(TraceEvent(agent=agent, message=f"No new passages for '{query}'"))
                break
            try:
                grade = await self.llm.structured(
                    system=GRADER_SYSTEM,
                    prompt=f"Query: {query}\n\n{_format_passages(candidates)}",
                    schema=GradeResult,
                    effort="low",
                )
            except LLMError as exc:
                logger.warning("grader failed (%s); keeping ungraded passages", exc)
                kept.update((c.chunk.chunk_id, c) for c in candidates)
                break

            ids = {f"P{i}": c for i, c in enumerate(candidates, start=1)}
            relevant = [ids[pid] for pid in grade.relevant_ids if pid in ids]
            kept.update((c.chunk.chunk_id, c) for c in relevant)
            trace.append(
                TraceEvent(
                    agent=agent,
                    message=(
                        f"Round {round_number}: {len(relevant)}/{len(candidates)} relevant "
                        f"passages for '{query}'"
                    ),
                    data={
                        "sufficient": grade.sufficient,
                        "kept": [c.chunk.chunk_id for c in relevant],
                    },
                )
            )
            if grade.sufficient or not grade.better_query or round_number == self.max_rounds:
                break
            query = grade.better_query
            trace.append(TraceEvent(agent=agent, message=f"Rewriting query to '{query}'"))

        evidence = [
            Evidence(
                kind="document",
                source=c.chunk.source,
                title=f"{c.chunk.title} > {c.chunk.section}" if c.chunk.section else c.chunk.title,
                content=c.chunk.text,
                reference=c.chunk.chunk_id,
                score=c.score,
            )
            for c in sorted(kept.values(), key=lambda c: -c.score)
        ]
        return {"evidence": evidence, "trace": trace}
