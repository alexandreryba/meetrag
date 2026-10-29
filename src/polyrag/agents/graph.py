"""LangGraph orchestration.

START -> router -> (parallel fan-out with Send) document agents / SQL agent
      -> synthesizer -> verifier -> END, or back to the synthesizer if ungrounded
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from polyrag.agents.analytics import SqlAgent
from polyrag.agents.documents import DocumentAgent
from polyrag.agents.router import RouterAgent

# Imported at runtime (not under TYPE_CHECKING): LangGraph reads these annotations.
from polyrag.agents.state import AgentTask, GraphState
from polyrag.agents.synthesis import Synthesizer, Verifier
from polyrag.domain.models import Source

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from polyrag.config import Settings
    from polyrag.domain.ports import LLM, SqlDatabase
    from polyrag.retrieval.hybrid import HybridRetriever
    from polyrag.retrieval.meetings import MeetingCatalog


def dispatch(state: GraphState) -> list[Send] | str:
    """Fan out one specialist task per routing step; they run concurrently."""
    steps = state["plan"].steps
    if not steps:
        return Synthesizer.name
    return [
        Send(
            SqlAgent.name if step.source is Source.ANALYTICS else DocumentAgent.name,
            AgentTask(question=state["question"], step=step),
        )
        for step in steps
    ]


def after_verification(state: GraphState) -> str:
    """Loop back once to the synthesizer when the verifier found unsupported claims."""
    return Synthesizer.name if state.get("feedback") else END


def build_graph(
    *,
    retriever: HybridRetriever,
    database: SqlDatabase,
    llm: LLM,
    fast_llm: LLM,
    settings: Settings,
    catalog: MeetingCatalog | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    graph = StateGraph(GraphState)
    graph.add_node(RouterAgent.name, RouterAgent(fast_llm, catalog=catalog))
    graph.add_node(
        DocumentAgent.name,
        DocumentAgent(
            retriever=retriever,
            llm=fast_llm,
            catalog=catalog,
            top_k=settings.top_k,
            max_rounds=settings.max_retrieval_rounds,
        ),
        input_schema=AgentTask,
    )
    graph.add_node(
        SqlAgent.name,
        SqlAgent(
            database=database,
            llm=fast_llm,
            max_attempts=settings.max_sql_attempts,
            max_rows=settings.sql_max_rows,
        ),
        input_schema=AgentTask,
    )
    graph.add_node(Synthesizer.name, Synthesizer(llm))
    graph.add_node(
        Verifier.name,
        Verifier(fast_llm, max_synthesis_attempts=settings.max_synthesis_attempts),
    )

    graph.add_edge(START, RouterAgent.name)
    graph.add_conditional_edges(
        RouterAgent.name, dispatch, [DocumentAgent.name, SqlAgent.name, Synthesizer.name]
    )
    graph.add_edge(DocumentAgent.name, Synthesizer.name)
    graph.add_edge(SqlAgent.name, Synthesizer.name)
    graph.add_edge(Synthesizer.name, Verifier.name)
    graph.add_conditional_edges(Verifier.name, after_verification)
    return graph.compile(name="polyrag")
