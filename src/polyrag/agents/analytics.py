"""SQL agent: text-to-SQL with guardrails and self-correction from execution errors."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from polyrag.agents.prompts import SQL_SYSTEM
from polyrag.agents.state import SqlDraft
from polyrag.domain.models import Evidence, Source, TraceEvent
from polyrag.domain.ports import LLMError, QueryError

if TYPE_CHECKING:
    from polyrag.agents.state import AgentTask
    from polyrag.domain.ports import LLM, SqlDatabase


class SqlAgent:
    name = "sql_agent"

    def __init__(
        self, *, database: SqlDatabase, llm: LLM, max_attempts: int = 3, max_rows: int = 200
    ) -> None:
        self.database = database
        self.llm = llm
        self.max_attempts = max_attempts
        self.max_rows = max_rows
        self._schema: str | None = None

    async def _describe(self) -> str:
        if self._schema is None:
            self._schema = await asyncio.to_thread(self.database.describe)
        return self._schema

    async def __call__(self, state: AgentTask) -> dict[str, Any]:
        question = state["step"].query
        trace: list[TraceEvent] = []
        try:
            schema = await self._describe()
        except QueryError as exc:
            return {"trace": [TraceEvent(agent=self.name, message=f"Database unavailable: {exc}")]}

        feedback = ""
        for attempt in range(1, self.max_attempts + 1):
            prompt = (
                f"Database schema ({self.database.dialect}), data snapshot as of 2026-09-30:\n"
                f"{schema}\n\nQuestion: {question}{feedback}"
            )
            try:
                draft = await self.llm.structured(
                    system=SQL_SYSTEM, prompt=prompt, schema=SqlDraft, effort="low"
                )
            except LLMError as exc:
                trace.append(TraceEvent(agent=self.name, message=f"SQL generation failed: {exc}"))
                break
            try:
                result = await asyncio.to_thread(
                    self.database.execute, draft.sql, max_rows=self.max_rows
                )
            except QueryError as exc:
                trace.append(
                    TraceEvent(
                        agent=self.name,
                        message=f"Attempt {attempt}: query rejected, retrying",
                        data={"sql": draft.sql, "error": str(exc)},
                    )
                )
                feedback = (
                    f"\n\nYour previous query failed.\nQuery:\n{draft.sql}\nError: {exc}\n"
                    "Fix the query."
                )
                continue

            trace.append(
                TraceEvent(
                    agent=self.name,
                    message=f"Query returned {len(result.rows)} row(s)",
                    data={"sql": draft.sql, "attempt": attempt},
                )
            )
            evidence = Evidence(
                kind="table",
                source=Source.ANALYTICS,
                title="Analytics database (snapshot 2026-09-30)",
                content=(
                    f"Question: {question}\nSQL:\n{draft.sql}\nResult:\n{result.to_markdown()}"
                ),
                reference=draft.sql,
            )
            return {"evidence": [evidence], "trace": trace}

        trace.append(TraceEvent(agent=self.name, message="No valid SQL query after retries"))
        return {"trace": trace}
