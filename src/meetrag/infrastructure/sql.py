"""Read-only SQL access with layered guardrails for LLM-generated queries.

1. ``sqlglot`` parses the query: exactly one read-only ``SELECT`` / set operation, no DDL/DML.
2. A ``LIMIT`` is enforced on the outermost query.
3. The SQLite connection is opened read-only and an authorizer denies anything but reads.
4. A progress handler aborts queries that run too long.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from typing import TYPE_CHECKING

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from meetrag.domain.ports import QueryError, QueryResult

if TYPE_CHECKING:
    from pathlib import Path

_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Pragma,
    exp.Attach,
    exp.Detach,
    exp.Transaction,
)

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


def validate_select(sql: str, *, dialect: str = "sqlite", max_rows: int = 200) -> str:
    """Return a normalized, row-limited version of ``sql`` or raise ``QueryError``."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise QueryError("empty query")
    try:
        statements = [s for s in sqlglot.parse(cleaned, read=dialect) if s is not None]
    except ParseError as exc:
        raise QueryError(f"could not parse SQL: {exc}") from exc
    if len(statements) != 1:
        raise QueryError("exactly one SQL statement is allowed")
    statement = statements[0]
    if not isinstance(statement, exp.Query):
        raise QueryError(f"only SELECT queries are allowed, got {statement.key.upper()}")
    forbidden = statement.find(*_FORBIDDEN_NODES)
    if forbidden is not None:
        raise QueryError(f"forbidden operation in query: {forbidden.key.upper()}")
    if statement.args.get("limit") is None:
        statement = statement.limit(max_rows)
    return statement.sql(dialect=dialect)


class SQLiteDatabase:
    """Implements the ``SqlDatabase`` port on a SQLite file."""

    dialect = "sqlite"

    def __init__(self, path: Path, *, timeout_s: float = 5.0, sample_rows: int = 3) -> None:
        self.path = path
        self._timeout_s = timeout_s
        self._sample_rows = sample_rows

    def _connect(self) -> sqlite3.Connection:
        if not self.path.exists():
            raise QueryError(f"database not found at {self.path}; run `meetrag ingest` first")
        connection = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)
        connection.set_authorizer(
            lambda action, *_: (
                sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY
            )
        )
        return connection

    def describe(self) -> str:
        with closing(self._connect()) as connection:
            tables = connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'table' ORDER BY name"
            ).fetchall()
            sections: list[str] = []
            for name, ddl in tables:
                (count,) = connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()  # noqa: S608
                cursor = connection.execute(f'SELECT * FROM "{name}" LIMIT ?', (self._sample_rows,))  # noqa: S608
                sample = QueryResult(
                    columns=[d[0] for d in cursor.description],
                    rows=[list(r) for r in cursor.fetchall()],
                )
                sections.append(f"{ddl};\n-- {count} rows. Sample:\n{sample.to_markdown()}")
        return "\n\n".join(sections)

    def execute(self, sql: str, *, max_rows: int = 200) -> QueryResult:
        safe_sql = validate_select(sql, dialect=self.dialect, max_rows=max_rows)
        deadline = time.monotonic() + self._timeout_s
        with closing(self._connect()) as connection:
            # Returning non-zero from the progress handler interrupts the query.
            connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
            try:
                cursor = connection.execute(safe_sql)
                rows = cursor.fetchmany(max_rows + 1)
            except sqlite3.DatabaseError as exc:
                raise QueryError(f"query failed: {exc}") from exc
        return QueryResult(
            columns=[d[0] for d in cursor.description or []],
            rows=[list(r) for r in rows[:max_rows]],
            truncated=len(rows) > max_rows,
        )
