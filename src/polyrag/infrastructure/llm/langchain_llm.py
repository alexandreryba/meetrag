"""Adapter for any LangChain chat model (Gemini, OpenAI, Mistral, Ollama...).

Install the extra and the provider package, e.g. ``uv sync --extra langchain`` then
``uv pip install langchain-google-genai``, and set
``POLYRAG_LLM_PROVIDER=langchain`` / ``POLYRAG_LLM_MODEL=google_genai:gemini-2.5-flash``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from polyrag.domain.ports import Effort, LLMError, SchemaT
from polyrag.observability import record_usage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


class LangChainLLM:
    def __init__(self, model: str) -> None:
        try:
            from langchain.chat_models import init_chat_model
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError("Install the 'langchain' extra: uv sync --extra langchain") from exc
        self.model = model
        self._chat: Any = init_chat_model(model)

    def _record(self, message: Any) -> None:
        usage = getattr(message, "usage_metadata", None) or {}
        record_usage(self.model, usage.get("input_tokens", 0), usage.get("output_tokens", 0))

    async def complete(
        self, *, system: str, prompt: str, effort: Effort = "medium", max_tokens: int = 16000
    ) -> str:
        message = await self._chat.ainvoke([("system", system), ("human", prompt)])
        self._record(message)
        return _text(message.content)

    async def structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[SchemaT],
        effort: Effort = "low",
        max_tokens: int = 16000,
    ) -> SchemaT:
        runnable = self._chat.with_structured_output(schema, include_raw=True)
        result = await runnable.ainvoke([("system", system), ("human", prompt)])
        self._record(result["raw"])
        parsed = result.get("parsed")
        if not isinstance(parsed, schema):
            raise LLMError(f"model returned no parsable {schema.__name__}")
        return parsed

    async def stream(
        self, *, system: str, prompt: str, effort: Effort = "medium", max_tokens: int = 16000
    ) -> AsyncIterator[str]:
        aggregate: Any = None
        async for chunk in self._chat.astream([("system", system), ("human", prompt)]):
            aggregate = chunk if aggregate is None else aggregate + chunk
            if text := _text(chunk.content):
                yield text
        if aggregate is not None:
            self._record(aggregate)
