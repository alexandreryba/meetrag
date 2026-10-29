"""Hybrid retriever: BM25 + dense search, fused with RRF, optionally reranked."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from meetrag.retrieval.fusion import reciprocal_rank_fusion

if TYPE_CHECKING:
    from meetrag.domain.models import ScoredChunk, SearchFilter
    from meetrag.domain.ports import Embedder, Reranker, VectorStore
    from meetrag.retrieval.bm25 import BM25Index


class RetrievalMode(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid+rerank"


class HybridRetriever:
    def __init__(
        self,
        *,
        bm25: BM25Index,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker | None = None,
        candidates_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.bm25 = bm25
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.candidates_k = candidates_k
        self.rrf_k = rrf_k

    @property
    def default_mode(self) -> RetrievalMode:
        return RetrievalMode.HYBRID_RERANK if self.reranker else RetrievalMode.HYBRID

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        where: SearchFilter | None = None,
        mode: RetrievalMode | None = None,
    ) -> list[ScoredChunk]:
        mode = mode or self.default_mode
        if mode is RetrievalMode.BM25:
            return self.bm25.search(query, limit=top_k, where=where)
        dense = self.store.search(
            self.embedder.embed_query(query), limit=self.candidates_k, where=where
        )
        if mode is RetrievalMode.DENSE:
            return dense[:top_k]

        keyword = self.bm25.search(query, limit=self.candidates_k, where=where)
        fused = reciprocal_rank_fusion([keyword, dense], k=self.rrf_k)
        if mode is RetrievalMode.HYBRID:
            return fused[:top_k]
        if self.reranker is None:
            raise ValueError("reranking requested but no reranker is configured")
        # Rerank a short list only: cross-encoders are accurate but cost one pass per pair.
        return self.reranker.rerank(query, fused[: max(top_k * 3, 10)], top_k)
