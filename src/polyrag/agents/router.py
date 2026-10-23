"""Router agent: plans which sources to consult and decomposes the question."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from polyrag.agents.prompts import ROUTER_SYSTEM
from polyrag.agents.state import RoutePlan, RouteStep
from polyrag.domain.models import DOCUMENT_SOURCES, TraceEvent
from polyrag.domain.ports import LLMError

if TYPE_CHECKING:
    from polyrag.agents.state import GraphState
    from polyrag.domain.ports import LLM

logger = logging.getLogger(__name__)


class RouterAgent:
    name = "router"

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        question = state["question"]
        try:
            plan = await self.llm.structured(
                system=ROUTER_SYSTEM,
                prompt=f"Question: {question}",
                schema=RoutePlan,
                effort="low",
            )
        except LLMError as exc:
            # Degrade gracefully: search every document source with the raw question.
            logger.warning("router failed (%s); falling back to all document sources", exc)
            plan = RoutePlan(
                reasoning="Routing failed; searching all document sources.",
                steps=[RouteStep(source=s, query=question) for s in DOCUMENT_SOURCES],
            )
        sources = ", ".join(step.source.value for step in plan.steps) or "none"
        event = TraceEvent(
            agent=self.name,
            message=f"Routing to: {sources}",
            data={
                "reasoning": plan.reasoning,
                "steps": [step.model_dump(mode="json") for step in plan.steps],
            },
        )
        return {"plan": plan, "trace": [event]}
