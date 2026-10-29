"""Cross-encoder reranking: scores (query, passage) pairs jointly, more precise than
bi-encoder similarity, applied only to the few fused candidates."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from meetrag.domain.models import ScoredChunk


class FastEmbedReranker:
    def __init__(self, model_name: str, cache_dir: Path | None = None) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self._model = TextCrossEncoder(
            model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None
        )

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        scores = list(self._model.rerank(query, [c.chunk.contextualized_text for c in candidates]))
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: -pair[1])
        return [
            candidate.model_copy(update={"score": float(score)})
            for candidate, score in ranked[:top_n]
        ]
