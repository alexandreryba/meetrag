"""Synthesizer (cited answer, token streaming) and verifier (groundedness check)."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from langgraph.config import get_stream_writer

from meetrag.agents.prompts import ABSTAIN_MESSAGE, SYNTHESIZER_SYSTEM, VERIFIER_SYSTEM
from meetrag.agents.state import Verdict
from meetrag.domain.models import Citation, Evidence, TraceEvent
from meetrag.domain.ports import LLMError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from meetrag.agents.state import GraphState
    from meetrag.domain.ports import LLM

logger = logging.getLogger(__name__)

_CITATION = re.compile(r"\[(\d{1,3})\]")


def unique_evidence(evidence: Sequence[Evidence]) -> list[Evidence]:
    """Deduplicate evidence (two agents may retrieve the same chunk), keeping order stable
    so that citation numbers mean the same thing for the synthesizer and the verifier."""
    seen: set[tuple[str, str]] = set()
    unique: list[Evidence] = []
    for item in evidence:
        key = (item.kind, item.reference)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def format_evidence(evidence: Sequence[Evidence]) -> str:
    return "\n\n".join(
        f"[{i}] source={item.source.value} | {item.title}\n{item.content}"
        for i, item in enumerate(evidence, start=1)
    )


def extract_citations(answer: str, evidence: Sequence[Evidence]) -> list[Citation]:
    indices = sorted({int(n) for n in _CITATION.findall(answer) if 1 <= int(n) <= len(evidence)})
    return [
        Citation(
            index=i,
            source=evidence[i - 1].source,
            title=evidence[i - 1].title,
            reference=evidence[i - 1].reference,
            snippet=evidence[i - 1].content[:280],
        )
        for i in indices
    ]


def is_abstention(answer: str) -> bool:
    return answer.strip().strip('"').startswith(ABSTAIN_MESSAGE.rstrip("."))


class Synthesizer:
    name = "synthesizer"

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        writer = get_stream_writer()
        attempt = state.get("synthesis_attempts", 0) + 1
        evidence = unique_evidence(state.get("evidence", []))
        if not evidence:
            writer({"type": "token", "text": ABSTAIN_MESSAGE})
            return {
                "answer": ABSTAIN_MESSAGE,
                "citations": [],
                "abstained": True,
                "synthesis_attempts": attempt,
                "trace": [TraceEvent(agent=self.name, message="No evidence: abstaining")],
            }

        prompt = f"Question: {state['question']}\n\nEvidence:\n{format_evidence(evidence)}"
        if feedback := state.get("feedback"):
            prompt += (
                "\n\nA fact-checker found these claims of your previous draft unsupported by "
                f"the evidence:\n{feedback}\nWrite a corrected answer without them."
            )
            writer({"type": "reset"})

        parts: list[str] = []
        try:
            async for text in self.llm.stream(
                system=SYNTHESIZER_SYSTEM, prompt=prompt, effort="medium"
            ):
                parts.append(text)
                writer({"type": "token", "text": text})
        except LLMError as exc:
            logger.warning("synthesis failed: %s", exc)
            writer({"type": "reset"})
            writer({"type": "token", "text": ABSTAIN_MESSAGE})
            parts = [ABSTAIN_MESSAGE]

        answer = "".join(parts).strip()
        citations = extract_citations(answer, evidence)
        return {
            "answer": answer,
            "citations": citations,
            "abstained": is_abstention(answer),
            "synthesis_attempts": attempt,
            "feedback": None,
            "trace": [
                TraceEvent(
                    agent=self.name,
                    message=f"Answer drafted with {len(citations)} citation(s)",
                    data={"attempt": attempt, "evidence_items": len(evidence)},
                )
            ],
        }


class Verifier:
    name = "verifier"

    def __init__(self, llm: LLM, *, max_synthesis_attempts: int = 2) -> None:
        self.llm = llm
        self.max_synthesis_attempts = max_synthesis_attempts

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        if state.get("abstained") or not state.get("answer"):
            return {"grounded": True, "feedback": None}
        evidence = unique_evidence(state.get("evidence", []))
        try:
            verdict = await self.llm.structured(
                system=VERIFIER_SYSTEM,
                prompt=(
                    f"Question: {state['question']}\n\nEvidence:\n{format_evidence(evidence)}"
                    f"\n\nAnswer to check:\n{state['answer']}"
                ),
                schema=Verdict,
                effort="low",
            )
        except LLMError as exc:
            message = f"Verification unavailable: {exc}"
            return {"grounded": False, "trace": [TraceEvent(agent=self.name, message=message)]}

        if verdict.grounded:
            event = TraceEvent(agent=self.name, message="All claims are supported by evidence")
            return {"grounded": True, "feedback": None, "trace": [event]}

        retry = state.get("synthesis_attempts", 1) < self.max_synthesis_attempts
        claims = "\n".join(f"- {claim}" for claim in verdict.unsupported_claims)
        event = TraceEvent(
            agent=self.name,
            message=(
                f"{len(verdict.unsupported_claims)} unsupported claim(s)"
                + ("; asking for a corrected answer" if retry else "")
            ),
            data={"unsupported_claims": verdict.unsupported_claims},
        )
        return {"grounded": False, "feedback": claims if retry else None, "trace": [event]}
