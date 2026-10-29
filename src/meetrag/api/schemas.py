"""HTTP request / response models (responses reuse the domain ``AnswerResult``)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from meetrag.domain.models import Source


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3, max_length=2000, examples=["What is the hotel cap in Paris?"]
    )


class SourceInfo(BaseModel):
    name: Source
    kind: str
    chunks: int | None = None


class Readiness(BaseModel):
    status: str
    chunks: int
