"""Pure metric functions (no I/O), easy to unit-test."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence


def hit_at_k(ranked_docs: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1.0 if any of the first k retrieved items belongs to a relevant document."""
    return float(any(doc in relevant for doc in ranked_docs[:k]))


def recall_at_k(ranked_docs: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Share of relevant documents covered by the first k retrieved items."""
    if not relevant:
        return 0.0
    return len(set(ranked_docs[:k]) & set(relevant)) / len(set(relevant))


def reciprocal_rank(ranked_docs: Sequence[str], relevant: Collection[str], k: int = 10) -> float:
    for rank, doc in enumerate(ranked_docs[:k], start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_docs: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Binary-relevance nDCG; a document only counts the first time it appears."""
    seen: set[str] = set()
    dcg = 0.0
    for rank, doc in enumerate(ranked_docs[:k], start=1):
        if doc in relevant and doc not in seen:
            dcg += 1.0 / math.log2(rank + 1)
        seen.add(doc)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def set_precision_recall(
    predicted: Collection[str], expected: Collection[str]
) -> tuple[float, float]:
    predicted_set, expected_set = set(predicted), set(expected)
    if not predicted_set and not expected_set:
        return 1.0, 1.0
    overlap = len(predicted_set & expected_set)
    precision = overlap / len(predicted_set) if predicted_set else 0.0
    recall = overlap / len(expected_set) if expected_set else 1.0
    return precision, recall


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
