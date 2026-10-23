"""LLM adapters and the factory selecting one from settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polyrag.config import LLMProvider

if TYPE_CHECKING:
    from polyrag.config import Settings
    from polyrag.domain.ports import LLM


def build_llm(settings: Settings, model: str) -> LLM:
    if settings.llm_provider is LLMProvider.ANTHROPIC:
        from polyrag.infrastructure.llm.anthropic_llm import AnthropicLLM

        return AnthropicLLM(model, refusal_fallback=settings.llm_refusal_fallback)
    if settings.llm_provider is LLMProvider.ANTHROPIC_VERTEX:
        from polyrag.infrastructure.llm.anthropic_llm import AnthropicLLM

        if not settings.gcp_project:
            raise ValueError("POLYRAG_GCP_PROJECT is required for the anthropic-vertex provider")
        return AnthropicLLM.on_vertex(
            model, project_id=settings.gcp_project, region=settings.gcp_region
        )
    from polyrag.infrastructure.llm.langchain_llm import LangChainLLM

    return LangChainLLM(model)
