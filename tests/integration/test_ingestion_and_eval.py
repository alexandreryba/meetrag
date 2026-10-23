from __future__ import annotations

from polyrag.config import Settings
from polyrag.container import Container
from polyrag.domain.models import DOCUMENT_SOURCES
from polyrag.evaluation.dataset import load_golden_set
from polyrag.evaluation.end_to_end import (
    ClaimCheck,
    CorrectnessVerdict,
    FaithfulnessVerdict,
    Judge,
    end_to_end_markdown,
    evaluate_end_to_end,
)
from polyrag.evaluation.retrieval import evaluate_retrieval
from polyrag.ingestion.loaders import load_corpus
from polyrag.ingestion.pipeline import load_chunks
from tests.conftest import ScriptedLLM


def test_corpus_has_documents_for_every_source(settings: Settings) -> None:
    documents = list(load_corpus(settings.corpus_dir))
    assert len(documents) == 16
    assert {d.source for d in documents} == set(DOCUMENT_SOURCES)
    assert any(d.title == "Remote Work Policy" for d in documents)


def test_ingestion_persists_chunks_and_database(settings: Settings, indexed) -> None:
    chunks = load_chunks(settings.chunks_path)
    assert len(chunks) > 50
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert settings.database_path.exists()


def test_golden_set_references_existing_documents(settings: Settings) -> None:
    doc_ids = {d.doc_id for d in load_corpus(settings.corpus_dir)}
    items = load_golden_set(settings.golden_set)
    assert len(items) >= 40
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
