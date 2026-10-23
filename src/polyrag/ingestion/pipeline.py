"""Ingestion pipeline: load -> chunk -> embed -> index, plus the analytics database."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from typing import TYPE_CHECKING

from pydantic import BaseModel

from polyrag.demo.voltaria_db import build_database
from polyrag.domain.models import Chunk
from polyrag.ingestion.chunking import MarkdownChunker
from polyrag.ingestion.loaders import load_corpus

if TYPE_CHECKING:
    from pathlib import Path

    from polyrag.config import Settings
    from polyrag.domain.ports import Embedder, VectorStore

logger = logging.getLogger(__name__)


class IngestionReport(BaseModel):
    documents: int
    chunks: int
    chunks_per_source: dict[str, int]
    database_rows: dict[str, int]
    seconds: float


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [Chunk.model_validate(json.loads(line)) for line in handle if line.strip()]


def ingest(settings: Settings, *, embedder: Embedder, store: VectorStore) -> IngestionReport:
    started = time.perf_counter()
    documents = list(load_corpus(settings.corpus_dir))
    if not documents:
        raise FileNotFoundError(f"no documents found in {settings.corpus_dir}")

    chunks = MarkdownChunker(settings.chunk_size, settings.chunk_overlap).split_all(documents)
    logger.info("chunked %d documents into %d chunks", len(documents), len(chunks))

    vectors = embedder.embed_documents([chunk.contextualized_text for chunk in chunks])
    store.reset(embedder.dimension)
    store.upsert(chunks, vectors)
    save_chunks(chunks, settings.chunks_path)

    rows = build_database(settings.database_path)
    logger.info("built analytics database %s: %s", settings.database_path, rows)

    return IngestionReport(
        documents=len(documents),
        chunks=len(chunks),
        chunks_per_source=dict(Counter(chunk.source.value for chunk in chunks)),
        database_rows=rows,
        seconds=round(time.perf_counter() - started, 2),
    )
