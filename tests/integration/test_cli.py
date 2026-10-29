from __future__ import annotations

import json
from contextlib import contextmanager

import pytest
from typer.testing import CliRunner

from meetrag import cli
from meetrag.container import Container

runner = CliRunner()


@pytest.fixture(autouse=True)
def _offline_container(monkeypatch: pytest.MonkeyPatch, container: Container) -> None:
    @contextmanager
    def fake_open_container():  # type: ignore[no-untyped-def]
        yield container

    monkeypatch.setattr(cli, "open_container", fake_open_container)


def test_ask_prints_json() -> None:
    result = runner.invoke(cli.app, ["ask", "What is the hotel cap in Paris?", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["citations"]


def test_eval_retrieval_writes_reports_and_enforces_the_quality_gate(container: Container) -> None:
    result = runner.invoke(cli.app, ["eval", "retrieval", "--reranker", ""])
    assert result.exit_code == 0, result.output
    assert (container.settings.reports_dir / "retrieval.md").exists()

    failing = runner.invoke(cli.app, ["eval", "retrieval", "--reranker", "", "--min-mrr", "1.01"])
    assert failing.exit_code == 1
