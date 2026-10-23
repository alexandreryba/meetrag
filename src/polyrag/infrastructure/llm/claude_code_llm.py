"""Claude through the local Claude Code CLI, billed to a Claude Pro or Max subscription.

Meant for local development and evaluation on your own machine: subscription access is
personal, so a service used by other people (the Cloud Run deployment) must use the Claude API
or Vertex AI. Requires ``uv sync --extra subscription`` and a Claude Code login (the Claude
desktop app or ``claude login``), or ``CLAUDE_CODE_OAUTH_TOKEN`` created by ``claude setup-token``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from polyrag.domain.ports import (
    Effort,
    LLMConfigurationError,
    LLMError,
    LLMUnavailableError,
    SchemaT,
)
from polyrag.observability import record_usage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage

    QueryFn = Callable[..., AsyncIterator[Any]]


class ClaudeCodeLLM:
    """Implements the ``LLM`` port by running one Claude Code turn per call.

    Each call is a plain completion: no tools, and none of the user's Claude Code settings,
    memory or MCP servers. Token usage and the API-equivalent cost reported by the CLI are
    recorded, although the subscription is what is actually billed.
    """

    def __init__(
        self,
        model: str,
        *,
        max_concurrency: int = 4,
        cli_path: str | None = None,
        query_fn: QueryFn | None = None,
    ) -> None:
        try:
            from claude_agent_sdk import query
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "Install the 'subscription' extra: uv sync --extra subscription"
            ) from exc
        self.model = model
        self._query: QueryFn = query_fn or query
        self._cli_path = cli_path
        # Every call starts a CLI process: cap how many run at once.
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def _options(
        self,
        system: str,
        effort: Effort,
        *,
        output_format: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> ClaudeAgentOptions:
        from claude_agent_sdk import ClaudeAgentOptions

        return ClaudeAgentOptions(
            system_prompt=system,
            model=self.model,
            effort=effort,
            tools=[],
            setting_sources=[],
            strict_mcp_config=True,
            max_turns=3 if output_format else 1,  # structured output takes an extra turn
            output_format=output_format,
            include_partial_messages=stream,
            cli_path=self._cli_path,
        )

    async def _run(self, prompt: str, options: ClaudeAgentOptions) -> AsyncIterator[Any]:
        from claude_agent_sdk import (
            AssistantMessage,
            CLINotFoundError,
            RateLimitEvent,
            ResultMessage,
        )

        async with self._semaphore:
            try:
                async for message in self._query(prompt=prompt, options=options):
                    if isinstance(message, AssistantMessage) and message.error:
                        _raise_for(message.error)
                    if (
                        isinstance(message, RateLimitEvent)
                        and message.rate_limit_info.status == "rejected"
                    ):
                        resets_at = message.rate_limit_info.resets_at
                        when = (
                            datetime.fromtimestamp(resets_at, UTC).strftime("%H:%M UTC")
                            if resets_at
                            else "later"
                        )
                        raise LLMUnavailableError(
                            f"Claude subscription usage limit reached; it resets at {when}"
                        )
                    if isinstance(message, ResultMessage):
                        self._account(message)
                    yield message
            except CLINotFoundError as exc:
                raise LLMConfigurationError(
                    "Claude Code CLI not found: reinstall the 'subscription' extra"
                ) from exc

    def _account(self, result: ResultMessage) -> None:
        usage = result.usage or {}
        input_tokens = sum(
            int(usage.get(key) or 0)
            for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        record_usage(
            self.model,
            input_tokens,
            int(usage.get("output_tokens") or 0),
            cost_usd=result.total_cost_usd,
        )
        if result.is_error:
            if result.api_error_status == 429:
                raise LLMUnavailableError("Claude subscription usage limit reached")
            detail = "; ".join(result.errors or []) or result.subtype
            raise LLMError(f"Claude Code call failed: {detail}")

    async def complete(
        self, *, system: str, prompt: str, effort: Effort = "medium", max_tokens: int = 16000
    ) -> str:
        from claude_agent_sdk import ResultMessage

        text = ""
        async for message in self._run(prompt, self._options(system, effort)):
            if isinstance(message, ResultMessage):
                text = message.result or ""
        return text

    async def structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[SchemaT],
        effort: Effort = "low",
        max_tokens: int = 16000,
    ) -> SchemaT:
        from claude_agent_sdk import ResultMessage

        output_format = {"type": "json_schema", "schema": schema.model_json_schema()}
        data: Any = None
        options = self._options(system, effort, output_format=output_format)
        async for message in self._run(prompt, options):
            if isinstance(message, ResultMessage):
                data = message.structured_output
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise LLMError(f"model returned no parsable {schema.__name__}") from exc

    async def stream(
        self, *, system: str, prompt: str, effort: Effort = "medium", max_tokens: int = 16000
    ) -> AsyncIterator[str]:
        from claude_agent_sdk import StreamEvent

        async for message in self._run(prompt, self._options(system, effort, stream=True)):
            if isinstance(message, StreamEvent) and message.event.get("type") == (
                "content_block_delta"
            ):
                delta = message.event.get("delta", {})
                if delta.get("type") == "text_delta" and delta.get("text"):
                    yield delta["text"]


def _raise_for(error: str) -> None:
    if error == "authentication_failed":
        raise LLMConfigurationError(
            "Claude Code is not logged in: open the Claude app or run `claude login`"
        )
    if error in {"rate_limit", "billing_error"}:
        raise LLMUnavailableError(f"Claude subscription unavailable ({error})")
    raise LLMError(f"Claude Code call failed: {error}")
