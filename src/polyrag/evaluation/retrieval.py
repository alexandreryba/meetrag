"""Offline retrieval evaluation: compares BM25, dense, hybrid and hybrid+rerank.

No LLM is involved, so this runs for free in CI and acts as a regression gate.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pydantic import BaseModel

from polyrag.evaluation.metrics import hit_at_k, mean, ndcg_at_k, recall_at_k, reciprocal_rank
from polyrag.retrieval.hybrid import RetrievalMode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polyrag.evaluation.dataset import GoldenItem
    from polyrag.retrieval.hybrid import HybridRetriever


class RetrievalScores(BaseModel):
    mode: RetrievalMode
    questions: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    recall_at_5: float
    mrr_at_10: float
    ndcg_at_5: float
    mean_latency_ms: float


def evaluate_retrieval(
    retriever: HybridRetriever,
    items: Sequence[GoldenItem],
    modes: Sequence[RetrievalMode] | None = None,
) -> list[RetrievalScores]:
    """Score every mode on the questions that have relevant documents. Each retrieved chunk
    is mapped to its document, so the metrics reflect what an agent actually receives."""
    labelled = [item for item in items if item.relevant_docs]
    if modes is None:
        modes = [m for m in RetrievalMode if m is not RetrievalMode.HYBRID_RERANK]
        if retriever.reranker is not None:
            modes.append(RetrievalMode.HYBRID_RERANK)

    results: list[RetrievalScores] = []
    for mode in modes:
        rows: dict[str, list[float]] = {
            k: [] for k in ("h1", "h3", "h5", "r5", "mrr", "ndcg", "ms")
        }
        for item in labelled:
            started = time.perf_counter()
            hits = retriever.retrieve(item.question, top_k=10, mode=mode)
            rows["ms"].append((time.perf_counter() - started) * 1000)
            docs = [hit.chunk.doc_id for hit in hits]
            relevant = set(item.relevant_docs)
            rows["h1"].append(hit_at_k(docs, relevant, 1))
            rows["h3"].append(hit_at_k(docs, relevant, 3))
            rows["h5"].append(hit_at_k(docs, relevant, 5))
            rows["r5"].append(recall_at_k(docs, relevant, 5))
            rows["mrr"].append(reciprocal_rank(docs, relevant, 10))
            rows["ndcg"].append(ndcg_at_k(docs, relevant, 5))
        results.append(
            RetrievalScores(
                mode=mode,
                questions=len(labelled),
                hit_at_1=mean(rows["h1"]),
                hit_at_3=mean(rows["h3"]),
                hit_at_5=mean(rows["h5"]),
                recall_at_5=mean(rows["r5"]),
                mrr_at_10=mean(rows["mrr"]),
                ndcg_at_5=mean(rows["ndcg"]),
                mean_latency_ms=mean(rows["ms"]),
            )
        )
    return results


def retrieval_markdown(scores: Sequence[RetrievalScores], *, chunks: int, embedding: str) -> str:
    lines = [
        "# Retrieval evaluation",
        "",
        f"{scores[0].questions if scores else 0} labelled questions, {chunks} chunks, "
        f"embeddings `{embedding}`. Global search (no source filter), chunk-level ranking "
        "mapped to documents.",
        "",
        "| Mode | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR@10 | nDCG@5 | Latency (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {s.mode.value} | {s.hit_at_1:.3f} | {s.hit_at_3:.3f} | {s.hit_at_5:.3f} | "
        f"{s.recall_at_5:.3f} | {s.mrr_at_10:.3f} | {s.ndcg_at_5:.3f} | {s.mean_latency_ms:.1f} |"
        for s in scores
    ]
    return "\n".join(lines) + "\n"
