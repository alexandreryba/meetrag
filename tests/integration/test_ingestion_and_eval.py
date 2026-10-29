from __future__ import annotations

from meetrag.config import Settings
from meetrag.container import Container
from meetrag.domain.models import Source
from meetrag.evaluation.dataset import load_golden_set
from meetrag.evaluation.end_to_end import (
    ClaimCheck,
    CorrectnessVerdict,
    FaithfulnessVerdict,
    Judge,
    end_to_end_markdown,
    evaluate_end_to_end,
)
from meetrag.evaluation.retrieval import evaluate_retrieval
from meetrag.infrastructure.sql import SQLiteDatabase
from meetrag.ingestion.loaders import WIKI_SOURCES, load_corpus
from meetrag.ingestion.pipeline import load_chunks
from meetrag.ingestion.transcripts import load_transcripts
from tests.conftest import Indexed, ScriptedLLM


def test_corpus_has_documents_for_every_source(settings: Settings) -> None:
    documents = list(load_corpus(settings.corpus_dir))
    assert len(documents) == 16
    assert {d.source for d in documents} == set(WIKI_SOURCES)
    assert any(d.title == "Remote Work Policy" for d in documents)


def test_ingestion_persists_chunks_and_database(settings: Settings, indexed: Indexed) -> None:
    chunks = load_chunks(settings.chunks_path)
    assert len(chunks) > 50
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert settings.database_path.exists()


def test_ingestion_indexes_transcripts_and_meeting_tables(
    settings: Settings, indexed: Indexed
) -> None:
    chunks = [c for c in load_chunks(settings.chunks_path) if c.source is Source.MEETINGS]
    assert len({c.doc_id for c in chunks}) == 10
    assert all(c.metadata["speakers"] and c.metadata["date"] for c in chunks)
    database = SQLiteDatabase(settings.database_path)
    assert database.execute("SELECT COUNT(*) FROM meetings").rows == [[10]]
    steering = database.execute(
        "SELECT COUNT(*) FROM meetings WHERE meeting_type = 'steering committee'"
    )
    assert steering.rows == [[3]]


def test_golden_set_references_existing_documents(settings: Settings) -> None:
    doc_ids = {d.doc_id for d in load_corpus(settings.corpus_dir)}
    doc_ids |= {t.doc_id for t in load_transcripts(settings.corpus_dir / "meetings")}
    items = load_golden_set(settings.golden_set)
    assert len(items) >= 60
    assert sum(item.category == "meetings" for item in items) >= 10
    for item in items:
        assert set(item.relevant_docs) <= doc_ids, item.id
        assert item.answerable == bool(item.expected_sources), item.id


def test_reference_sql_queries_run(container: Container) -> None:
    for item in load_golden_set(container.settings.golden_set):
        if item.reference_sql:
            assert container.database.execute(item.reference_sql).rows, item.id


def test_retrieval_evaluation_scores_every_mode(container: Container) -> None:
    scores = evaluate_retrieval(container.retriever, load_golden_set(container.settings.golden_set))
    assert [s.mode.value for s in scores] == ["bm25", "dense", "hybrid"]
    for s in scores:
        assert 0.0 <= s.mrr_at_10 <= 1.0
    bm25 = scores[0]
    assert bm25.hit_at_5 > 0.8  # keyword search alone already finds most documents


async def test_end_to_end_evaluation_produces_a_report(container: Container) -> None:
    judge_llm = ScriptedLLM(
        handlers={
            CorrectnessVerdict: lambda _: CorrectnessVerdict(rationale="ok", grade="partial"),
            FaithfulnessVerdict: lambda _: FaithfulnessVerdict(
                claims=[
                    ClaimCheck(claim="a", supported=True),
                    ClaimCheck(claim="b", supported=False),
                ]
            ),
        }
    )
    ids = {"pol-06", "ana-01", "oos-02"}
    items = [i for i in load_golden_set(container.settings.golden_set) if i.id in ids]
    report = await evaluate_end_to_end(
        container.assistant,
        Judge(judge_llm, container.database),
        items,
        models={"synthesis": "scripted"},
    )
    assert report.questions == 3
    assert report.summary["answer_accuracy"] == 0.5
    assert report.summary["faithfulness"] == 0.5
    assert report.summary["abstention_accuracy"] == 1.0
    # the reference SQL is executed so the judge compares against exact figures
    correctness_prompts = [p for kind, p in judge_llm.calls if kind == "CorrectnessVerdict"]
    assert any("Result of the reference SQL query" in p for p in correctness_prompts)
    assert "End-to-end evaluation" in end_to_end_markdown(report)
