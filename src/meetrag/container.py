"""Composition root: builds adapters from settings and wires them into the graph.

Every dependency is created lazily (the first ``ask`` loads the models, ``ingest`` never
needs an API key) and can be overridden, which is how tests inject in-memory fakes.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

from meetrag.agents.graph import build_graph
from meetrag.domain.models import Source
from meetrag.infrastructure.llm import build_llm
from meetrag.ingestion.meeting_records import MeetingExtractor, write_meeting_records
from meetrag.ingestion.pipeline import IngestionReport, ingest, load_chunks
from meetrag.ingestion.transcripts import load_transcripts
from meetrag.retrieval.bm25 import BM25Index
from meetrag.retrieval.hybrid import HybridRetriever
from meetrag.retrieval.meetings import MeetingCatalog
from meetrag.service import Assistant

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from meetrag.config import Settings
    from meetrag.domain.ports import LLM, Embedder, Reranker, SqlDatabase, VectorStore

_DERIVED = ("bm25", "catalog", "retriever", "graph", "assistant")


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
        from meetrag.infrastructure.embeddings import FastEmbedEmbedder

        return FastEmbedEmbedder(self.settings.embedding_model)

    @cached_property
    def store(self) -> VectorStore:
        from meetrag.infrastructure.vectorstores import QdrantVectorStore

        return QdrantVectorStore.from_settings(self.settings)

    @cached_property
    def reranker(self) -> Reranker | None:
        if not self.settings.reranker_model:
            return None
        from meetrag.infrastructure.rerankers import FastEmbedReranker

        return FastEmbedReranker(self.settings.reranker_model)

    @cached_property
    def database(self) -> SqlDatabase:
        from meetrag.infrastructure.sql import SQLiteDatabase

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
    def catalog(self) -> MeetingCatalog:
        return MeetingCatalog.from_chunks(self.bm25.chunks)

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
            catalog=self.catalog,
        )

    @cached_property
    def assistant(self) -> Assistant:
        return Assistant(self.graph)

    def ingest(self) -> IngestionReport:
        report = ingest(self.settings, embedder=self.embedder, store=self.store)
        for name in _DERIVED:  # rebuild the keyword index and graph on next access
            self.__dict__.pop(name, None)
        return report

    async def extract_meeting_records(self, *, force: bool = False) -> tuple[dict[str, int], int]:
        """Extract decisions, action items and summaries (LLM, cached) into the database.
        Returns the row counts and the number of meetings that needed a model call."""
        settings = self.settings
        transcripts = load_transcripts(
            settings.corpus_dir / Source.MEETINGS.value, redact=settings.redact_pii
        )
        extractor = MeetingExtractor(self.llm, settings.insights_dir)
        insights, fresh = await extractor.extract_all(transcripts, force=force)
        counts = write_meeting_records(settings.database_path, transcripts, insights)
        for name in ("graph", "assistant"):  # the SQL agent caches the schema
            self.__dict__.pop(name, None)
        return counts, fresh

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
