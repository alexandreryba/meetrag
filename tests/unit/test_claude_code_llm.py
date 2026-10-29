"""The subscription adapter, tested against a fake of ``claude_agent_sdk.query``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    RateLimitEvent,
    ResultMessage,
    StreamEvent,
    TextBlock,
)
from claude_agent_sdk.types import RateLimitInfo
from pydantic import BaseModel

from meetrag.domain.ports import LLMConfigurationError, LLMError, LLMUnavailableError
from meetrag.infrastructure.llm.claude_code_llm import ClaudeCodeLLM
from meetrag.observability import track_usage


class Answer(BaseModel):
    value: int


def _result(**overrides: Any) -> ResultMessage:
    fields: dict[str, Any] = {
        "subtype": "success",
        "duration_ms": 900,
        "duration_api_ms": 700,
        "is_error": False,
        "num_turns": 1,
        "session_id": "s",
        "total_cost_usd": 0.012,
        "usage": {
            "input_tokens": 10,
            "cache_creation_input_tokens": 1000,
            "cache_read_input_tokens": 0,
            "output_tokens": 50,
        },
        "result": "Hello",
    }
    return ResultMessage(**{**fields, **overrides})


def _delta(text: str) -> StreamEvent:
    event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}
    return StreamEvent(uuid="u", session_id="s", event=event)


class FakeQuery:
    def __init__(self, *messages: Any) -> None:
        self.messages = messages
        self.options: Any = None

    async def __call__(self, *, prompt: str, options: Any) -> AsyncIterator[Any]:
        self.options = options
        for message in self.messages:
            yield message


def _llm(*messages: Any) -> tuple[ClaudeCodeLLM, FakeQuery]:
    fake = FakeQuery(*messages)
    return ClaudeCodeLLM("claude-opus-5-5", query_fn=fake), fake


async def test_complete_runs_a_tool_free_isolated_turn_and_records_usage() -> None:
    llm, fake = _llm(_result())
    with track_usage() as usage:
        assert await llm.complete(system="sys", prompt="p", effort="low") == "Hello"
    options = fake.options
    assert options.system_prompt == "sys" and options.model == "claude-opus-5-5"
    assert options.tools == [] and options.setting_sources == [] and options.strict_mcp_config
    assert options.effort == "low" and options.max_turns == 1
    assert usage.input_tokens == 1010 and usage.output_tokens == 50
    assert usage.cost_usd == pytest.approx(0.012)  # cost reported by the CLI, not estimated


async def test_structured_uses_a_json_schema_and_validates_it() -> None:
    llm, fake = _llm(_result(structured_output={"value": 42}))
    assert await llm.structured(system="s", prompt="p", schema=Answer) == Answer(value=42)
    assert fake.options.output_format["type"] == "json_schema"
    assert fake.options.output_format["schema"]["properties"]["value"]["type"] == "integer"


async def test_invalid_structured_output_is_an_llm_error() -> None:
    llm, _ = _llm(_result(structured_output={"value": "not a number"}))
    with pytest.raises(LLMError, match="no parsable Answer"):
        await llm.structured(system="s", prompt="p", schema=Answer)


async def test_stream_yields_text_deltas() -> None:
    llm, fake = _llm(_delta("Hel"), _delta("lo"), _result())
    assert [t async for t in llm.stream(system="s", prompt="p")] == ["Hel", "lo"]
    assert fake.options.include_partial_messages is True


@pytest.mark.parametrize(
    ("message", "error", "match"),
    [
        (
            AssistantMessage(
                content=[TextBlock(text="")], model="m", error="authentication_failed"
            ),
            LLMConfigurationError,
            "not logged in",
        ),
        (
            RateLimitEvent(
                rate_limit_info=RateLimitInfo(status="rejected", resets_at=1_800_000_000, raw={}),
                uuid="u",
                session_id="s",
            ),
            LLMUnavailableError,
            "usage limit reached; it resets at",
        ),
        (_result(is_error=True, api_error_status=429), LLMUnavailableError, "usage limit"),
        (_result(is_error=True, subtype="error_during_execution"), LLMError, "call failed"),
    ],
)
async def test_failures_are_mapped_to_domain_errors(
    message: Any, error: type[Exception], match: str
) -> None:
    llm, _ = _llm(message)
    with pytest.raises(error, match=match):
        await llm.complete(system="s", prompt="p")
