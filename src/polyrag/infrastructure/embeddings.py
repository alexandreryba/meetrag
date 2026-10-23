"""Embedding adapters: ONNX models through FastEmbed (CPU-friendly) and a hashing fallback."""

from __future__ import annotations

import hashlib
import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class FastEmbedEmbedder:
    """Local ONNX embeddings (default: BAAI/bge-small-en-v1.5, 384 dims, ~67 MB).

    No GPU and no API key needed, so ingestion and retrieval evaluation run anywhere,
    including CI and a 2-vCPU Cloud Run instance.
    """

    def __init__(self, model_name: str, cache_dir: Path | None = None) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(
            model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None
        )
        self._dimension = int(self._model.embedding_size)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        # query_embed applies the model-specific query instruction (e.g. BGE's prefix).
        return [float(x) for x in next(iter(self._model.query_embed(text)))]


_WORD = re.compile(r"\w+")


class HashingEmbedder:
    """Deterministic bag-of-words embeddings via the hashing trick.

    Not semantically strong, but dependency-free and stable across runs: used in unit
    tests and as an offline fallback.
    """

    def __init__(self, dimension: int = 256) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in _WORD.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
