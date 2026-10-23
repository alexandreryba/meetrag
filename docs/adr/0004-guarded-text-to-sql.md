# ADR 0004: Text-to-SQL with layered guardrails and self-correction

- Status: accepted
- Date: 2026-10-23

## Context

Figures such as churn rates or ticket volumes cannot be retrieved from documents: they must
be computed. Executing LLM-generated SQL is a security risk (data modification, expensive
queries) and a reliability risk (wrong column names).

## Decision

The SQL agent receives the schema (DDL with column comments documenting allowed values, row
counts and sample rows) and returns a structured `SqlDraft`. Execution goes through four
layers:

1. **Static validation with `sqlglot`**: exactly one statement, which must be a query
   (`SELECT`, CTE or set operation); any DDL/DML, `PRAGMA` or `ATTACH` node is rejected.
2. **Row limit**: a `LIMIT` is added to the outer query when missing.
3. **Read-only connection**: SQLite opened with `mode=ro` and an authorizer that denies
   everything except reads and functions.
4. **Timeout**: a progress handler aborts queries running longer than 5 s.

Rejections and execution errors are fed back to the model, which corrects its query (up to 3
attempts).

## Consequences

- Defense in depth: bypassing the parser still hits a read-only connection.
- Golden-set analytics questions store a **reference SQL query** instead of a hard-coded
  answer: the evaluation executes it, so references stay exact when the data generator changes.
- The `SqlDatabase` port keeps the agent independent of SQLite; a BigQuery adapter would reuse
  the same validation with the `bigquery` dialect.
