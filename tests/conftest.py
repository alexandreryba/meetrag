"""Shared fixtures. The whole pipeline runs offline: hashing embeddings, an in-memory vector
store, and a scripted LLM that implements the ``LLM`` port with deterministic rules."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from meetrag.agents.state import GradeResult, RoutePlan, RouteStep, SqlDraft, Verdict
from meetrag.config import Settings
from meetrag.container import Container
from meetrag.domain.models import Source
from meetrag.domain.ports import Effort
from meetrag.infrastructure.embeddings import HashingEmbedder
from meetrag.infrastructure.vectorstores import InMemoryVectorStore

ROOT = Path(__file__).resolve().parents[1]

ROUTING_RULES: list[tuple[str, Source]] = [
    (r"remote|leave|hotel|expense|phishing|gift|security", Source.POLICIES),
    (r"price|offer|wallbox|solar|meter|kwh", Source.PRODUCTS),
    (r"dispute|outage|retention|ticket priority|escalat|goodwill", Source.SUPPORT),
    (r"how many|average|churn rate|total|count", Source.ANALYTICS),
]


def scripted_router(prompt: str) -> RoutePlan:
    question = prompt.split("Question: ", 1)[-1]
    steps = [
        RouteStep(source=source, query=question)
        for pattern, source in ROUTING_RULES
        if re.search(pattern, question, re.IGNORECASE)
    ]
    return RoutePlan(reasoning="keyword rules", steps=steps)


def grade_everything(prompt: str) -> GradeResult:
    ids = re.findall(r'<passage id="(P\d+)"', prompt)
    return GradeResult(relevant_ids=ids, sufficient=True)


@dataclass
class ScriptedLLM:
    """Deterministic implementation of the ``LLM`` port for tests."""

    model: str = "scripted"
    handlers: dict[type[BaseModel], Callable[[str], BaseModel]] = field(default_factory=dict)
    answer: Callable[[str], str] = lambda prompt: "Here is the answer [1]."
    calls: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        defaults: dict[type[BaseModel], Callable[[str], BaseModel]] = {
            RoutePlan: scripted_router,
            GradeResult: grade_everything,
            SqlDraft: lambda _: SqlDraft(
                sql="SELECT COUNT(*) AS active_customers FROM customers WHERE status = 'active'",
                rationale="count active customers",
            ),
            Verdict: lambda _: Verdict(grounded=True),
        }
        self.handlers = {**defaults, **self.handlers}

    async def complete(
        self, *, system: str, prompt: str, effort: Effort = "medium", max_tokens: int = 4096
    ) -> str:
        self.calls.append(("complete", prompt))
        return self.answer(prompt)

    async def structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[Any],
        effort: Effort = "low",
        max_tokens: int = 4096,
    ) -> Any:
        self.calls.append((schema.__name__, prompt))
        return self.handlers[schema](prompt)

    async def stream(
        self, *, system: str, prompt: str, effort: Effort = "medium", max_tokens: int = 4096
    ) -> AsyncIterator[str]:
        self.calls.append(("stream", prompt))
        for word in self.answer(prompt).split(" "):
            yield word + " "


@pytest.fixture(scope="session")
def index_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("index")


@pytest.fixture(scope="session")
def settings(index_dir: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        corpus_dir=ROOT / "data" / "corpus",
        golden_set=ROOT / "eval" / "golden_set.jsonl",
        index_dir=index_dir,
        reports_dir=index_dir / "reports",
        reranker_model=None,
        api_key=None,
        auto_ingest=False,
    )


@pytest.fixture(scope="session")
def indexed(settings: Settings) -> tuple[HashingEmbedder, InMemoryVectorStore]:
    """Ingest the real corpus once per session with offline components."""
    embedder, store = HashingEmbedder(), InMemoryVectorStore()
    Container(settings, embedder=embedder, store=store).ingest()
    return embedder, store


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def container(
    settings: Settings, indexed: tuple[HashingEmbedder, InMemoryVectorStore], llm: ScriptedLLM
) -> Container:
    embedder, store = indexed
    return Container(
        settings,
        embedder=embedder,
        store=store,
        reranker=None,
        llm=llm,
        fast_llm=llm,
        judge_llm=llm,
    )
