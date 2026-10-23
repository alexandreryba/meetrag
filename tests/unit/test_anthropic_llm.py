"""The Claude adapter, tested against a stub of the SDK's ``beta.messages`` resource."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from polyrag.domain.ports import LLMError
from polyrag.infrastructure.llm.anthropic_llm import AnthropicLLM
from polyrag.observability import track_usage


class Answer(BaseModel):
    value: int


def _message(stop_reason: str = "end_turn", parsed: Any = None, category: str | None = None) -> Any:
    return SimpleNamespace(
        model="claude-opus-5-5",
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category=category) if category else None,
        usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
        content=[
            SimpleNamespace(type="thinking", text=""),
            SimpleNamespace(type="text", text="Hello"),
        ],
        parsed_output=parsed,
    )


class StubMessages:
    def __init__(self, message: Any) -> None:
        self.message = message
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.message

    async def parse(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.message


def _llm(message: Any, *, fallback: bool = True) -> tuple[AnthropicLLM, StubMessages]:
    from anthropic import AsyncAnthropic

    llm = AnthropicLLM("claude-opus-5-5", client=AsyncAnthropic(api_key="test"))
    llm._fallback = fallback
    stub = StubMessages(message)
    llm._messages = stub  # type: ignore[assignment]
    return llm, stub


async def test_complete_sends_effort_and_refusal_fallback_and_records_cost() -> None:
    llm, stub = _llm(_message())
    with track_usage() as usage:
        text = await llm.complete(system="s", prompt="p", effort="low")
    assert text == "Hello"
    assert stub.kwargs["output_config"] == {"effort": "low"}
    assert stub.kwargs["fallbacks"] == "default"
    assert stub.kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    assert "temperature" not in stub.kwargs
    assert usage.llm_calls == 1
    assert usage.cost_usd == pytest.approx((1000 * 4 + 200 * 20) / 1_000_000)


async def test_vertex_requests_omit_fallbacks() -> None:
    llm, stub = _llm(_message(), fallback=False)
    await llm.complete(system="s", prompt="p")
    assert "fallbacks" not in stub.kwargs and "betas" not in stub.kwargs


async def test_structured_returns_the_parsed_model() -> None:
    llm, stub = _llm(_message(parsed=Answer(value=42)))
    assert await llm.structured(system="s", prompt="p", schema=Answer) == Answer(value=42)
    assert stub.kwargs["output_format"] is Answer


@pytest.mark.parametrize(
    ("message", "match"),
    [
        (_message(stop_reason="refusal", category="cyber"), "declined"),
        (_message(stop_reason="max_tokens"), "truncated"),
        (_message(parsed=None), "no parsable"),
    ],
)
async def test_failures_become_llm_errors(message: Any, match: str) -> None:
    llm, _ = _llm(message)
    with pytest.raises(LLMError, match=match):
        await llm.structured(system="s", prompt="p", schema=Answer)


async def test_missing_credentials_raise_a_configuration_error() -> None:
    from anthropic import AsyncAnthropic

    from polyrag.domain.ports import LLMConfigurationError

    llm = AnthropicLLM("claude-opus-5-5", client=AsyncAnthropic(api_key=None, auth_token=None))
    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        await llm.complete(system="s", prompt="p")
