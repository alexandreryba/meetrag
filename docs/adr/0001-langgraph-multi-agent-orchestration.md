# ADR 0001: Orchestrate specialist agents with LangGraph

- Status: accepted
- Date: 2026-10-23

## Context

Questions mix several kinds of knowledge: written policies and procedures, product
documentation, and figures that only exist in a database. A single "retrieve top-k from one
index" chain either misses the database entirely or floods the prompt with unrelated passages.

## Decision

Model the assistant as a LangGraph `StateGraph`:

1. a **router** produces a structured plan: which sources, and one self-contained sub-query per
   source (question decomposition);
2. the plan fans out with `Send`, so **document agents** and the **SQL agent** run concurrently;
   a reducer (`operator.add`) merges their evidence;
3. a **synthesizer** writes the answer from numbered evidence and streams tokens through
   LangGraph's custom stream mode;
4. a **verifier** checks every claim against the evidence and loops back to the synthesizer once
   when it finds unsupported claims.

Agents receive their dependencies through constructors, and the graph is assembled in
`meetrag.agents.graph.build_graph`, so each agent can be unit-tested in isolation.

## Consequences

- Latency is bounded by the slowest specialist rather than the sum of all of them.
- Every step emits a `TraceEvent`, which the API streams and the UI displays: the reasoning path
  is inspectable, which matters for trust and debugging.
- Each question costs several LLM calls (router, one grader per source, synthesis, verification).
  The cheap calls run with `effort="low"` on a configurable fast model.
- A free-form ReAct agent was rejected: the control flow here is known in advance, and an
  explicit graph is cheaper, more predictable and easier to evaluate.
