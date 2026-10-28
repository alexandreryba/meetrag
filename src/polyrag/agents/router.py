"""Router agent: plans which sources to consult and decomposes the question."""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING, Any

from polyrag.agents.prompts import ROUTER_SYSTEM
from polyrag.agents.state import RoutePlan, RouteStep
from polyrag.domain.models import DOCUMENT_SOURCES, TraceEvent
from polyrag.domain.ports import LLMError

if TYPE_CHECKING:
    from collections.abc import Callable

    from polyrag.agents.state import GraphState
    from polyrag.domain.ports import LLM
    from polyrag.retrieval.meetings import MeetingCatalog

logger = logging.getLogger(__name__)


class RouterAgent:
    name = "router"

    def __init__(
        self,
        llm: LLM,
        *,
        catalog: MeetingCatalog | None = None,
        today: Callable[[], date] = date.today,
    ) -> None:
        self.llm = llm
        self.catalog = catalog
        self.today = today

    def _prompt(self, question: str) -> str:
        context = [f"Today's date: {self.today().isoformat()}"]
        if self.catalog is not None:
            context.append(f"Meeting catalog: {self.catalog.describe()}")
        return "\n".join([*context, "", f"Question: {question}"])

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        question = state["question"]
        try:
            plan = await self.llm.structured(
                system=ROUTER_SYSTEM,
                prompt=self._prompt(question),
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
