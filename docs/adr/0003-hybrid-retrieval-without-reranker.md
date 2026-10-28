# ADR 0003: Hybrid retrieval by default, cross-encoder reranking off

- Status: accepted
- Date: 2026-10-23

## Context

Enterprise documents contain exact tokens that dense embeddings handle poorly (offer codes
such as `FIXE3`, amounts such as `€75`, tool names such as `Expensio`), and paraphrased
questions that keyword search handles poorly. Cross-encoders usually improve precision, but
they are expensive on CPU.

## Decision

Retrieval combines BM25 and dense search (`BAAI/bge-small-en-v1.5`, 384 dimensions) with
Reciprocal Rank Fusion (k = 60). RRF only uses ranks, so the two score scales never need
calibrating. Chunks are split along Markdown headings and indexed with their heading path
prepended (contextual chunk headers).

The decision is based on `polyrag eval retrieval` (33 labelled questions, 98 chunks):

| Mode | Hit@1 | Recall@5 | MRR@10 | Latency |
|---|---|---|---|---|
| BM25 | 0.879 | 0.970 | 0.923 | 0.1 ms |
| Dense | 0.970 | 0.985 | 0.985 | 14 ms |
| Hybrid (RRF) | 0.939 | **1.000** | 0.961 | 13 ms |
| Hybrid + rerank (MiniLM-L6) | 0.909 | 0.985 | 0.949 | 1,057 ms |

- Hybrid is the only mode with perfect Recall@5. Agents take the top 5 passages and an LLM
  grader removes distractors, so recall is the metric that matters at this stage.
- The cross-encoder costs about 1 s per query and improves no metric on this corpus, so it is
  disabled by default (`POLYRAG_RERANKER_MODEL` re-enables it).

## Update, 2026-10-28: with meeting transcripts

With the 10 meeting transcripts indexed (48 labelled questions, 123 chunks), dense and hybrid
search are on par (MRR@10 0.923 and 0.918, Recall@5 0.934 and 0.931); hybrid keeps the best
Hit@1 (0.896). Hybrid remains the default: BM25 is what reliably matches exact identifiers
(offer codes, amounts, names), for 3 ms more per query. The reranker still improves no metric.

## Consequences

- The benchmark runs in CI with real embeddings and fails the build if hybrid MRR@10 drops
  below 0.90.
- The decision should be revisited on a larger or noisier corpus, where rerankers usually pay
  off. The comparison is one command away.
