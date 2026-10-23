"""Application service: runs the graph and shapes its output for the API, CLI and eval."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

from polyrag.agents.synthesis import unique_evidence
from polyrag.domain.models import AnswerResult, TokenUsage
from polyrag.observability import track_usage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langgraph.graph.state import CompiledStateGraph

    from polyrag.agents.state import GraphState


class Assistant:
    def __init__(self, graph: CompiledStateGraph[Any, Any, Any, Any]) -> None:
        self.graph = graph

    @staticmethod
    def _initial_state(question: str) -> GraphState:
        return {"question": question.strip(), "evidence": [], "trace": [], "synthesis_attempts": 0}

    @staticmethod
    def _to_result(
        question: str, state: GraphState, usage: TokenUsage, started: float
    ) -> AnswerResult:
        plan = state.get("plan")
        evidence = unique_evidence(state.get("evidence", []))
        citations = state.get("citations", [])
        cited_sources = list(dict.fromkeys(c.source for c in citations))
        return AnswerResult(
            question=question,
            answer=state.get("answer", ""),
            citations=citations,
            routed_sources=list(dict.fromkeys(s.source for s in plan.steps)) if plan else [],
            sources_used=cited_sources or list(dict.fromkeys(e.source for e in evidence)),
            grounded=state.get("grounded", False),
            abstained=state.get("abstained", False),
            evidence=evidence,
            trace=state.get("trace", []),
            usage=usage,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    async def ask(self, question: str) -> AnswerResult:
        started = time.perf_counter()
        with track_usage() as usage:
            state = cast("GraphState", await self.graph.ainvoke(self._initial_state(question)))
        return self._to_result(question, state, usage, started)

    async def stream(self, question: str) -> AsyncIterator[dict[str, Any]]:
        """Yield ``trace`` events as agents progress, ``token`` events while the answer is
        written (``reset`` when a draft is rewritten), then one ``final`` event."""
        started = time.perf_counter()
        with track_usage() as usage:
            final: GraphState | None = None
            async for item in self.graph.astream(
                self._initial_state(question), stream_mode=["updates", "custom", "values"]
            ):
                mode, chunk = cast("tuple[str, Any]", item)
                if mode == "custom":
                    yield chunk
                elif mode == "updates":
                    for update in chunk.values():
                        for event in (update or {}).get("trace", []):
                            yield {"type": "trace", "event": event.model_dump(mode="json")}
                elif mode == "values":
                    final = chunk
            result = self._to_result(question, final or {}, usage, started)
        yield {"type": "final", "result": result.model_dump(mode="json")}
