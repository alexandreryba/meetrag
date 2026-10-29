"""Logging setup and per-request token / cost accounting."""

from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from typing import TYPE_CHECKING

from meetrag.config import MODEL_PRICING
from meetrag.domain.models import TokenUsage

if TYPE_CHECKING:
    from collections.abc import Iterator

_current_usage: ContextVar[TokenUsage | None] = ContextVar("meetrag_usage", default=None)


@contextmanager
def track_usage() -> Iterator[TokenUsage]:
    """Collect token usage of every LLM call made inside the block, including calls made by
    concurrent LangGraph branches (asyncio tasks inherit the context)."""
    usage = TokenUsage()
    token = _current_usage.set(usage)
    try:
        yield usage
    finally:
        # An async generator closed from another task (client disconnect) runs this in a
        # different context, where the token cannot be reset; the value is then unreachable.
        with suppress(ValueError):
            _current_usage.reset(token)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    input_price, output_price = MODEL_PRICING.get(model, (0.0, 0.0))
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def record_usage(
    model: str, input_tokens: int, output_tokens: int, *, cost_usd: float | None = None
) -> None:
    """Add one call to the current scope; ``cost_usd`` overrides the price-table estimate."""
    usage = _current_usage.get()
    if usage is not None:
        if cost_usd is None:
            cost_usd = estimate_cost(model, input_tokens, output_tokens)
        usage.add(input_tokens, output_tokens, cost_usd)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "severity": record.levelname,  # "severity" is understood by Cloud Logging
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record),
        }
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", *, json_logs: bool = False) -> None:
    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
