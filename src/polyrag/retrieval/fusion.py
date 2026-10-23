"""Reciprocal Rank Fusion (Cormack et al., 2009).

RRF combines rankings using ranks only, so BM25 scores (unbounded) and cosine similarities
(bounded) never need to be calibrated against each other.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from polyrag.domain.models import ScoredChunk

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polyrag.domain.models import Chunk


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], *, k: int = 60, limit: int | None = None
) -> list[ScoredChunk]:
    fused: dict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, scored in enumerate(ranking, start=1):
            chunk_id = scored.chunk.chunk_id
            fused[chunk_id] += 1.0 / (k + rank)
            chunks[chunk_id] = scored.chunk
    ordered = sorted(fused.items(), key=lambda item: -item[1])
    if limit is not None:
        ordered = ordered[:limit]
    return [ScoredChunk(chunk=chunks[chunk_id], score=score) for chunk_id, score in ordered]
