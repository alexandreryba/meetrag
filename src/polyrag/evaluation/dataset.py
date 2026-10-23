"""Golden evaluation set."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from polyrag.domain.models import Source

if TYPE_CHECKING:
    from pathlib import Path

Category = Literal["policies", "products", "support", "analytics", "multi_source", "out_of_scope"]


class GoldenItem(BaseModel):
    id: str
    category: Category
    question: str
    expected_sources: list[Source] = Field(default_factory=list)
    relevant_docs: list[str] = Field(default_factory=list)  # doc ids, for retrieval metrics
    reference_answer: str | None = None
    reference_sql: str | None = None  # executed at eval time: the reference stays exact
    answerable: bool = True


def load_golden_set(path: Path) -> list[GoldenItem]:
    with path.open(encoding="utf-8") as handle:
        return [GoldenItem.model_validate(json.loads(line)) for line in handle if line.strip()]
