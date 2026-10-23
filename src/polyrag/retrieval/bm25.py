"""Okapi BM25 keyword index.

Dense embeddings miss exact tokens such as offer codes ("FIXE3"), amounts ("€75") or
internal tool names ("Expensio"); BM25 catches them, which is why it is fused with dense
retrieval rather than replaced by it.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from polyrag.domain.models import ScoredChunk

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polyrag.domain.models import Chunk, Source

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "the",
        "their",
        "them",
        "there",
        "this",
        "to",
        "us",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    ]
)


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.lower())
    ascii_text = normalized.encode("ascii", "ignore").decode()
    tokens = []
    for token in _TOKEN.findall(ascii_text):
        if token in _STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]  # light plural stemming: "offers" -> "offer"
        tokens.append(token)
    return tokens


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._doc_len: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, chunk in enumerate(self.chunks):
            counts = Counter(tokenize(chunk.contextualized_text))
            self._doc_len.append(sum(counts.values()))
            for term, frequency in counts.items():
                self._postings[term].append((index, frequency))
        self._avg_len = sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0.0
        n = len(self.chunks)
        self._idf = {
            term: math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self._postings.items()
        }

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, *, limit: int, source: Source | None = None) -> list[ScoredChunk]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for index, frequency in self._postings[term]:
                length_norm = 1 - self.b + self.b * self._doc_len[index] / self._avg_len
                scores[index] += (
                    idf * frequency * (self.k1 + 1) / (frequency + self.k1 * length_norm)
                )
        ranked = sorted(scores.items(), key=lambda item: -item[1])
        results: list[ScoredChunk] = []
        for index, score in ranked:
            chunk = self.chunks[index]
            if source is None or chunk.source == source:
                results.append(ScoredChunk(chunk=chunk, score=score))
                if len(results) == limit:
                    break
        return results
