# ADR 0005: Two-level evaluation, with retrieval as a CI quality gate

- Status: accepted
- Date: 2026-10-23

## Context

Changes to chunking, prompts or models can silently degrade answers. LLM-based evaluation is
informative but slow and paid; it cannot run on every commit.

## Decision

A golden set of 46 questions (`eval/golden_set.jsonl`) covers six categories: policies,
products, support, analytics, multi-source and out-of-scope. Each item records the expected
sources, the relevant documents, and a reference answer or a reference SQL query.

1. **Retrieval evaluation** (offline, free): Hit@k, Recall@5, MRR@10 and nDCG@5 for BM25,
   dense, hybrid and hybrid + rerank. It runs in CI with real embeddings and fails the build
   below a threshold.
2. **End-to-end evaluation** (LLM-as-judge, manual workflow):
   - *accuracy*: the judge grades the answer against the reference (correct / partial /
     incorrect); SQL references are executed at evaluation time;
   - *faithfulness*: the judge splits the answer into atomic claims and checks each against the
     evidence that was given to the synthesizer;
   - *routing* precision and recall, *abstention* accuracy on out-of-scope questions and
     false-abstention rate on the others;
   - *latency* p50 / p95, *cost* and *LLM calls* per question, measured from token usage.

The judge's own token usage is excluded from the cost figures. Latency is measured with
`concurrency=1` by default so that the figures are not inflated by parallel load.

## Consequences

- Retrieval regressions are caught on every pull request at no cost.
- End-to-end numbers are reproducible from a single command and published as a report, which
  makes claims about the system verifiable.
- LLM judges have biases; using a different judge model from the system under test reduces
  self-preference, and is supported through `MEETRAG_JUDGE_MODEL`.
