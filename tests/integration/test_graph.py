from __future__ import annotations

import itertools

from meetrag.agents.prompts import ABSTAIN_MESSAGE
from meetrag.agents.state import SqlDraft, Verdict
from meetrag.config import Settings
from meetrag.container import Container
from meetrag.domain.models import Source
from tests.conftest import Indexed, ScriptedLLM


async def test_document_question_is_answered_with_citations(container: Container) -> None:
    result = await container.assistant.ask("How many days per week can I work remote?")
    assert Source.POLICIES in result.routed_sources
    assert result.evidence and all(
        e.kind == "document" for e in result.evidence if e.source is Source.POLICIES
    )
    assert result.citations[0].index == 1
    assert result.grounded and not result.abstained
    agents = [event.agent for event in result.trace]
    assert agents[0] == "router" and "synthesizer" in agents and "verifier" in agents


async def test_out_of_scope_question_abstains_without_calling_the_synthesizer(
    container: Container, llm: ScriptedLLM
) -> None:
    result = await container.assistant.ask("Who won the 2022 football World Cup?")
    assert result.abstained
    assert result.answer == ABSTAIN_MESSAGE
    assert result.routed_sources == []
    assert not any(kind == "stream" for kind, _ in llm.calls)


async def test_multi_source_question_fans_out_to_sql_and_documents(container: Container) -> None:
    result = await container.assistant.ask(
        "What is the churn rate, and what does the retention playbook say?"
    )
    assert set(result.routed_sources) == {Source.SUPPORT, Source.ANALYTICS}
    kinds = {e.kind for e in result.evidence}
    assert kinds == {"document", "table"}
    table = next(e for e in result.evidence if e.kind == "table")
    assert "active_customers" in table.content


async def test_sql_agent_self_corrects_after_a_rejected_query(
    settings: Settings, indexed: Indexed
) -> None:
    drafts = itertools.chain(
        [SqlDraft(sql="DROP TABLE customers", rationale="bad")],
        itertools.repeat(SqlDraft(sql="SELECT COUNT(*) AS n FROM support_tickets", rationale="ok")),
    )
    llm = ScriptedLLM(handlers={SqlDraft: lambda _: next(drafts)})
    embedder, store = indexed
    container = Container(
        settings, embedder=embedder, store=store, reranker=None, llm=llm, fast_llm=llm
    )
    result = await container.assistant.ask("How many tickets in total?")
    sql_prompts = [prompt for kind, prompt in llm.calls if kind == "SqlDraft"]
    assert len(sql_prompts) == 2
    assert "previous query failed" in sql_prompts[1]
    assert any("SELECT COUNT(*) AS n FROM support_tickets" in e.reference for e in result.evidence)


async def test_verifier_triggers_one_rewrite_then_stops(
    settings: Settings, indexed: Indexed
) -> None:
    llm = ScriptedLLM(
        handlers={Verdict: lambda _: Verdict(grounded=False, unsupported_claims=["made-up claim"])}
    )
    embedder, store = indexed
    container = Container(
        settings, embedder=embedder, store=store, reranker=None, llm=llm, fast_llm=llm
    )
    result = await container.assistant.ask("What is the hotel cap in Paris?")
    streams = [prompt for kind, prompt in llm.calls if kind == "stream"]
    assert len(streams) == settings.max_synthesis_attempts
    assert "made-up claim" in streams[1]
    assert result.grounded is False


async def test_streaming_emits_trace_tokens_and_a_final_result(container: Container) -> None:
    events = [
        event async for event in container.assistant.stream("What is the price of Voltaria Verte?")
    ]
    types = [event["type"] for event in events]
    assert types[0] == "trace" and types[-1] == "final"
    assert "token" in types
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert tokens.strip() == events[-1]["result"]["answer"]


async def test_meeting_filter_restricts_retrieval_to_the_latest_steering_committee(
    settings: Settings, indexed: Indexed
) -> None:
    from meetrag.agents.state import RoutePlan, RouteStep
    from meetrag.domain.models import MeetingFilter

    def route_to_last_committee(prompt: str) -> RoutePlan:
        assert "Meeting catalog: 10 meetings" in prompt  # the router sees the catalog
        step = RouteStep(
            source=Source.MEETINGS,
            query="decisions of the steering committee",
            meeting_filter=MeetingFilter(meeting_type="steering committee", latest_only=True),
        )
        return RoutePlan(reasoning="meetings", steps=[step])

    llm = ScriptedLLM(handlers={RoutePlan: route_to_last_committee})
    embedder, store = indexed
    container = Container(
        settings, embedder=embedder, store=store, reranker=None, llm=llm, fast_llm=llm
    )
    result = await container.assistant.ask("What was decided at the last steering committee?")
    assert result.evidence
    assert {e.reference.split("#")[0] for e in result.evidence} == {
        "meetings/2026-10-06-steering-committee-3"
    }
    assert any("Meeting filter: 1 meeting(s)" in event.message for event in result.trace)
