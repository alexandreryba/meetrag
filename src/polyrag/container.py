"""Composition root: builds adapters from settings and wires them into the graph.

Every dependency is created lazily (the first ``ask`` loads the models, ``ingest`` never
needs an API key) and can be overridden, which is how tests inject in-memory fakes.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

from polyrag.agents.graph import build_graph
from polyrag.infrastructure.llm import build_llm
from polyrag.ingestion.pipeline import IngestionReport, ingest, load_chunks
from polyrag.retrieval.bm25 import BM25Index
from polyrag.retrieval.hybrid import HybridRetriever
from polyrag.service import Assistant

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from polyrag.config import Settings
    from polyrag.domain.ports import LLM, Embedder, Reranker, SqlDatabase, VectorStore

_DERIVED = ("bm25", "retriever", "graph", "assistant")


class Container:
    def __init__(self, settings: Settings, **overrides: Any) -> None:
        self.settings = settings
        for name, value in overrides.items():
            if not hasattr(type(self), name):
                raise TypeError(f"unknown dependency override: {name}")
            # cached_property reads the instance __dict__ first: this pins the override.
            self.__dict__[name] = value

    @cached_property
    def embedder(self) -> Embedder:
        from polyrag.infrastructure.embeddings import FastEmbedEmbedder

        return FastEmbedEmbedder(self.settings.embedding_model)

    @cached_property
    def store(self) -> VectorStore:
        from polyrag.infrastructure.vectorstores import QdrantVectorStore

        return QdrantVectorStore.from_settings(self.settings)

    @cached_property
    def reranker(self) -> Reranker | None:
        if not self.settings.reranker_model:
            return None
        from polyrag.infrastructure.rerankers import FastEmbedReranker

        return FastEmbedReranker(self.settings.reranker_model)

    @cached_property
    def database(self) -> SqlDatabase:
        from polyrag.infrastructure.sql import SQLiteDatabase

        return SQLiteDatabase(self.settings.database_path)

    @cached_property
    def llm(self) -> LLM:
        return build_llm(self.settings, self.settings.llm_model)

    @cached_property
    def fast_llm(self) -> LLM:
        return build_llm(self.settings, self.settings.llm_fast_model)

    @cached_property
    def judge_llm(self) -> LLM:
        return build_llm(self.settings, self.settings.judge_model)

    @cached_property
    def bm25(self) -> BM25Index:
        return BM25Index(load_chunks(self.settings.chunks_path))

    @cached_property
    def retriever(self) -> HybridRetriever:
        return HybridRetriever(
            bm25=self.bm25,
            embedder=self.embedder,
            store=self.store,
            reranker=self.reranker,
            candidates_k=self.settings.candidates_k,
            rrf_k=self.settings.rrf_k,
        )

    @cached_property
    def graph(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        return build_graph(
            retriever=self.retriever,
            database=self.database,
            llm=self.llm,
            fast_llm=self.fast_llm,
            settings=self.settings,
        )

    @cached_property
    def assistant(self) -> Assistant:
        return Assistant(self.graph)

    def ingest(self) -> IngestionReport:
        report = ingest(self.settings, embedder=self.embedder, store=self.store)
        for name in _DERIVED:  # rebuild the keyword index and graph on next access
            self.__dict__.pop(name, None)
        return report

    def close(self) -> None:
        store = self.__dict__.get("store")
        if store is not None and hasattr(store, "close"):
            store.close()

    @property
    def is_ready(self) -> bool:
        return (
            self.settings.chunks_path.exists()
            and self.settings.database_path.exists()
            and self.store.count() > 0
        )
