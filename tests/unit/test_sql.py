from pathlib import Path

import pytest

from meetrag.demo.voltaria_db import OFFERS, build_database
from meetrag.domain.ports import QueryError
from meetrag.infrastructure.sql import SQLiteDatabase, validate_select


@pytest.fixture(scope="module")
def database(tmp_path_factory: pytest.TempPathFactory) -> SQLiteDatabase:
    path = tmp_path_factory.mktemp("db") / "voltaria.db"
    build_database(path, n_customers=200)
    return SQLiteDatabase(path)


def test_select_gets_a_limit() -> None:
    assert validate_select("SELECT * FROM customers", max_rows=50).endswith("LIMIT 50")
    assert validate_select("SELECT * FROM customers LIMIT 5;").endswith("LIMIT 5")


def test_ctes_and_unions_are_allowed() -> None:
    validate_select("WITH t AS (SELECT 1 AS x) SELECT x FROM t")
    validate_select("SELECT 1 UNION SELECT 2")


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE customers",
        "DELETE FROM customers",
        "INSERT INTO offers VALUES (1)",
        "UPDATE customers SET status = 'active'",
        "SELECT 1; DROP TABLE customers",
        "PRAGMA table_info(customers)",
        "ATTACH DATABASE 'x.db' AS x",
        "",
        "SELEC nonsense FROM",
    ],
)
def test_unsafe_or_invalid_queries_are_rejected(sql: str) -> None:
    with pytest.raises(QueryError):
        validate_select(sql)


def test_execute_returns_rows_and_truncates(database: SQLiteDatabase) -> None:
    result = database.execute("SELECT customer_id FROM customers ORDER BY customer_id", max_rows=10)
    assert result.columns == ["customer_id"]
    assert len(result.rows) == 10
    assert result.truncated is False  # LIMIT 10 was injected, so no extra row exists


def test_execution_errors_become_query_errors(database: SQLiteDatabase) -> None:
    with pytest.raises(QueryError, match="no such column"):
        database.execute("SELECT missing_column FROM customers")


def test_describe_contains_schema_comments_and_samples(database: SQLiteDatabase) -> None:
    schema = database.describe()
    assert "CREATE TABLE support_tickets" in schema
    assert "'P1' (critical)" in schema
    assert "rows. Sample:" in schema


def test_database_is_deterministic(tmp_path: Path) -> None:
    first = build_database(tmp_path / "a.db", n_customers=150, seed=7)
    second = build_database(tmp_path / "b.db", n_customers=150, seed=7)
    assert first == second


def test_offer_prices_match_the_product_documentation() -> None:
    overview = (
        Path(__file__).resolve().parents[2] / "data/corpus/products/electricity-offers-overview.md"
    ).read_text(encoding="utf-8")
    for offer in OFFERS:
        assert f"€{offer.price:.4f}" in overview, offer.offer_id
        assert f"€{offer.monthly_fee:.2f}" in overview, offer.offer_id
