# ADR 0002: Ports and adapters around every external dependency

- Status: accepted
- Date: 2026-10-23

## Context

The LLM provider, the vector database and the embedding model are the parts of a RAG system
that change most often (price, quality, hosting constraints). Tests must not depend on network
access or API keys.

## Decision

The domain layer defines `Protocol` ports: `LLM` (`complete`, `structured`, `stream`),
`Embedder`, `VectorStore`, `Reranker` and `SqlDatabase`. Agents only depend on these ports.
Adapters live in `polyrag.infrastructure`:

| Port | Adapters |
|---|---|
| `LLM` | Claude through the Anthropic SDK (Claude API or Vertex AI); any LangChain chat model |
| `Embedder` | FastEmbed (ONNX, CPU); hashing embedder for tests |
| `VectorStore` | Qdrant (embedded or server); NumPy in-memory store |
| `Reranker` | FastEmbed cross-encoder |
| `SqlDatabase` | SQLite with guardrails |

`polyrag.container.Container` is the composition root. It builds adapters lazily from
settings and accepts overrides, which is how tests inject fakes.

## Consequences

- The full graph, the API and the CLI are tested offline with a scripted LLM in about 10 s.
- Moving to pgvector, BigQuery or another model means writing one adapter, without touching
  the agents.
- The `LLM` port is deliberately small. Provider-specific features (effort, refusal fallback,
  structured outputs) stay inside the adapters.
