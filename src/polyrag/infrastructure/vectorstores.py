"""Vector store adapters: Qdrant (embedded or server) and a NumPy in-memory store."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import numpy as np

from polyrag.domain.models import Chunk, ScoredChunk

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qdrant_client import QdrantClient

    from polyrag.config import Settings
    from polyrag.domain.models import Source

_ID_NAMESPACE = uuid.UUID("6f1c2b8e-3d4a-4f5b-9c7d-0e1f2a3b4c5d")


class QdrantVectorStore:
    """Qdrant-backed store. Runs embedded (local files) by default, or against a Qdrant
    server / Qdrant Cloud when ``POLYRAG_QDRANT_URL`` is set."""

    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    @classmethod
    def from_settings(cls, settings: Settings) -> QdrantVectorStore:
        from qdrant_client import QdrantClient

        if settings.qdrant_url:
            api_key = (
                settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
            )
            client = QdrantClient(url=settings.qdrant_url, api_key=api_key)
        else:
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            client = QdrantClient(path=str(settings.qdrant_path))
        return cls(client, settings.collection)

    def reset(self, dimension: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self._client.create_collection(
            self._collection,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=str(uuid.uuid5(_ID_NAMESPACE, chunk.chunk_id)),
                vector=list(vector),
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), 256):
            self._client.upsert(self._collection, points=points[start : start + 256], wait=True)

    def search(
        self, vector: Sequence[float], *, limit: int, source: Source | None = None
    ) -> list[ScoredChunk]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = (
            Filter(must=[FieldCondition(key="source", match=MatchValue(value=source.value))])
            if source
            else None
        )
        response = self._client.query_points(
            self._collection,
            query=list(vector),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return [
            ScoredChunk(chunk=Chunk.model_validate(point.payload), score=float(point.score))
            for point in response.points
        ]

    def count(self) -> int:
        if not self._client.collection_exists(self._collection):
            return 0
        return int(self._client.count(self._collection).count)

    def close(self) -> None:
        self._client.close()


class InMemoryVectorStore:
    """Exact cosine search over a NumPy matrix. Fast enough for a few thousand chunks."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)

    def reset(self, dimension: int) -> None:
        self._chunks = []
        self._matrix = np.zeros((0, dimension), dtype=np.float32)

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = (matrix / np.where(norms == 0, 1, norms)).astype(np.float32)
        self._chunks.extend(chunks)
        self._matrix = np.vstack([self._matrix, matrix]) if self._matrix.size else matrix

    def search(
        self, vector: Sequence[float], *, limit: int, source: Source | None = None
    ) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        query = np.asarray(vector, dtype=np.float32)
        query = query / (np.linalg.norm(query) or 1.0)
        scores = self._matrix @ query
        order = np.argsort(-scores)
        results: list[ScoredChunk] = []
        for index in order:
            chunk = self._chunks[int(index)]
            if source is None or chunk.source == source:
                results.append(ScoredChunk(chunk=chunk, score=float(scores[index])))
                if len(results) == limit:
                    break
        return results

    def count(self) -> int:
        return len(self._chunks)
