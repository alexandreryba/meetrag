"""Claude adapter built on the official Anthropic SDK (Claude API or Vertex AI)."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

from anthropic import AsyncAnthropic, AsyncAnthropicVertex

from polyrag.domain.ports import Effort, LLMConfigurationError, LLMError, SchemaT
from polyrag.observability import record_usage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from anthropic.resources.beta.messages import AsyncMessages
    from anthropic.types.beta import BetaMessage

logger = logging.getLogger(__name__)

# Fallback on policy refusals, routed server-side by refusal category (Claude API only).
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


@contextmanager
def _credentials_required() -> Iterator[None]:
    """The SDK resolves credentials lazily and raises a bare TypeError when none exist."""
    try:
        yield
    except TypeError as exc:
        if "authentication" not in str(exc).lower():
            raise
        raise LLMConfigurationError(
            "no Claude credentials found: set ANTHROPIC_API_KEY (see .env.example)"
        ) from exc


class AnthropicLLM:
    """Implements the ``LLM`` port with Claude.

    * structured outputs through ``messages.parse`` (JSON-schema constrained decoding),
    * ``output_config.effort`` to trade depth for latency per call site,
    * refusals surfaced as ``LLMError`` so the graph can abstain instead of crashing.
    """

    def __init__(
        self,
        model: str,
        *,
        client: AsyncAnthropic | AsyncAnthropicVertex | None = None,
        refusal_fallback: bool = True,
    ) -> None:
        self.model = model
        client = client or AsyncAnthropic()
        # Both clients expose the same beta Messages surface.
        self._messages = cast("AsyncMessages", client.beta.messages)
        # Server-side fallbacks are a Claude API feature; Vertex AI rejects the parameter.
        self._fallback = refusal_fallback and isinstance(client, AsyncAnthropic)

    @classmethod
    def on_vertex(cls, model: str, *, project_id: str, region: str) -> AnthropicLLM:
        client = AsyncAnthropicVertex(project_id=project_id, region=region)
        return cls(model, client=client, refusal_fallback=False)

    def _request(self, system: str, prompt: str, effort: Effort, max_tokens: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": effort},
        }
        if self._fallback:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def _account(self, message: BetaMessage) -> None:
        record_usage(message.model, message.usage.input_tokens, message.usage.output_tokens)
        if message.stop_reason == "refusal":
            category = message.stop_details.category if message.stop_details else None
            raise LLMError(f"model declined the request (category={category})")
        if message.stop_reason == "max_tokens":
            raise LLMError("model output was truncated (max_tokens reached)")

    async def complete(
        self, *, system: str, prompt: str, effort: Effort = "medium", max_tokens: int = 16000
    ) -> str:
        with _credentials_required():
            message = await self._messages.create(
                **self._request(system, prompt, effort, max_tokens)
            )
        self._account(message)
        return "".join(block.text for block in message.content if block.type == "text")

    async def structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[SchemaT],
        effort: Effort = "low",
        max_tokens: int = 16000,
    ) -> SchemaT:
        with _credentials_required():
            message = await self._messages.parse(
                **self._request(system, prompt, effort, max_tokens), output_format=schema
            )
        self._account(message)
        parsed = message.parsed_output
        if parsed is None:
            raise LLMError(f"model returned no parsable {schema.__name__}")
        return parsed

    async def stream(
        self, *, system: str, prompt: str, effort: Effort = "medium", max_tokens: int = 16000
    ) -> AsyncIterator[str]:
        with _credentials_required():
            async with self._messages.stream(
                **self._request(system, prompt, effort, max_tokens)
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                final = await stream.get_final_message()
        self._account(final)
