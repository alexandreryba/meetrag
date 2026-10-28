"""Command-line interface: ``polyrag ingest | ask | eval retrieval | eval e2e | serve``."""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from polyrag.config import LLMProvider, get_settings
from polyrag.container import Container
from polyrag.domain.models import Source
from polyrag.domain.ports import LLMUnavailableError
from polyrag.observability import configure_logging

if TYPE_CHECKING:
    from collections.abc import Iterator

    from polyrag.evaluation.end_to_end import ItemResult

app = typer.Typer(help="PolyRAG: multi-agent, multi-source RAG assistant.", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluate retrieval (offline) or the full pipeline (LLM judge).")
app.add_typer(eval_app, name="eval")
console = Console()

DEFAULT_RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2"


@contextmanager
def open_container() -> Iterator[Container]:
    settings = get_settings()
    configure_logging("WARNING", json_logs=settings.log_json)
    container = Container(settings)
    try:
        yield container
    finally:
        container.close()


@app.command()
def ingest() -> None:
    """Index the document corpus and build the analytics database."""
    status = console.status("Indexing documents and building the analytics database...")
    with open_container() as container, status:
        report = container.ingest()
    table = Table(title="Ingestion", show_header=False)
    table.add_row("Documents", str(report.documents))
    table.add_row("Meeting transcripts", str(report.meetings))
    table.add_row("  with extracted records", str(report.meetings_with_records))
    table.add_row("Chunks", str(report.chunks))
    for source, count in report.chunks_per_source.items():
        table.add_row(f"  {source}", str(count))
    for name, rows in report.database_rows.items():
        table.add_row(f"Table {name}", f"{rows:,} rows")
    table.add_row("Duration", f"{report.seconds} s")
    console.print(table)


@app.command()
def extract(
    force: Annotated[bool, typer.Option(help="Re-extract even when the cache is valid")] = False,
) -> None:
    """Extract decisions, action items, risks and summaries from the meeting transcripts."""
    with open_container() as container:
        try:
            with console.status("Extracting meeting records..."):
                counts, fresh = asyncio.run(container.extract_meeting_records(force=force))
        except LLMUnavailableError as exc:
            console.print(f"[red]LLM provider unavailable:[/red] {exc}")
            raise typer.Exit(code=2) from exc
    table = Table(title="Meeting records", show_header=False)
    table.add_row("Meetings", str(counts["meetings"]))
    table.add_row("  extracted now / from cache", f"{fresh} / {counts['meetings'] - fresh}")
    for name in ("decisions", "action_items", "risks"):
        table.add_row(name.replace("_", " ").capitalize(), str(counts[name]))
    console.print(table)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to ask the assistant")],
    as_json: Annotated[bool, typer.Option("--json", help="Print the raw JSON result")] = False,
) -> None:
    """Ask a question and print the cited answer with the agent trace."""
    with open_container() as container:
        try:
            result = asyncio.run(container.assistant.ask(question))
        except LLMUnavailableError as exc:
            console.print(f"[red]LLM provider unavailable:[/red] {exc}")
            raise typer.Exit(code=2) from exc
    if as_json:
        console.print_json(result.model_dump_json())
        return
    for event in result.trace:
        console.print(f"[dim]{event.agent:>16}[/dim]  {event.message}")
    console.print(Panel(Markdown(result.answer), title="Answer", border_style="green"))
    for citation in result.citations:
        console.print(f"[cyan][{citation.index}][/cyan] {citation.source.value}: {citation.title}")
    console.print(
        f"[dim]grounded={result.grounded} | {result.latency_ms / 1000:.1f} s | "
        f"{result.usage.llm_calls} LLM calls | ${result.usage.cost_usd:.4f}[/dim]"
    )


@eval_app.command("retrieval")
def eval_retrieval(
    min_mrr: Annotated[
        float, typer.Option(help="Fail (exit 1) if hybrid MRR@10 falls below this value")
    ] = 0.0,
    reranker: Annotated[
        str | None,
        typer.Option(help="Cross-encoder to include in the comparison ('' to skip)"),
    ] = DEFAULT_RERANKER,
) -> None:
    """Compare BM25 / dense / hybrid / hybrid+rerank on the golden set (no LLM needed)."""
    from polyrag.evaluation.dataset import load_golden_set
    from polyrag.evaluation.retrieval import evaluate_retrieval, retrieval_markdown
    from polyrag.retrieval.hybrid import RetrievalMode

    with open_container() as container:
        settings = container.settings
        retriever = container.retriever
        if reranker and retriever.reranker is None:
            from polyrag.infrastructure.rerankers import FastEmbedReranker

            retriever.reranker = FastEmbedReranker(reranker)
        items = load_golden_set(settings.golden_set)
        with console.status("Running retrieval evaluation..."):
            scores = evaluate_retrieval(retriever, items)
        chunks = len(container.bm25)
    markdown = retrieval_markdown(scores, chunks=chunks, embedding=settings.embedding_model)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / "retrieval.md").write_text(markdown, encoding="utf-8")
    (settings.reports_dir / "retrieval.json").write_text(
        json.dumps([s.model_dump(mode="json") for s in scores], indent=2), encoding="utf-8"
    )
    console.print(Markdown(markdown))
    hybrid = next(s for s in scores if s.mode is RetrievalMode.HYBRID)
    if hybrid.mrr_at_10 < min_mrr:
        console.print(f"[red]Quality gate failed: MRR@10 {hybrid.mrr_at_10:.3f} < {min_mrr}[/red]")
        raise typer.Exit(code=1)


@eval_app.command("extraction")
def eval_extraction(
    threshold: Annotated[float, typer.Option(help="Minimum task similarity to match")] = 0.7,
) -> None:
    """Score the cached meeting records against the hand-annotated gold (no LLM call)."""
    from polyrag.evaluation.extraction import (
        embedding_similarity,
        evaluate_extraction,
        extraction_markdown,
        load_gold,
    )
    from polyrag.ingestion.meeting_records import load_cached_insights
    from polyrag.ingestion.transcripts import load_transcripts

    with open_container() as container:
        settings = container.settings
        transcripts = load_transcripts(
            settings.corpus_dir / Source.MEETINGS.value, redact=settings.redact_pii
        )
        extracted = load_cached_insights(settings.insights_dir, transcripts)
        if not extracted:
            console.print("[red]No extracted records: run `polyrag extract` first.[/red]")
            raise typer.Exit(code=1)
        scores = evaluate_extraction(
            load_gold(settings.extraction_gold),
            extracted,
            embedding_similarity(container.embedder.embed_documents),
            threshold=threshold,
        )
    markdown = extraction_markdown(scores, model=settings.llm_model)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / "extraction.md").write_text(markdown, encoding="utf-8")
    (settings.reports_dir / "extraction.json").write_text(
        scores.model_dump_json(indent=2), encoding="utf-8"
    )
    console.print(Markdown(markdown))


@eval_app.command("e2e")
def eval_end_to_end(
    limit: Annotated[int | None, typer.Option(help="Only run the first N questions")] = None,
    category: Annotated[str | None, typer.Option(help="Only run one category")] = None,
    concurrency: Annotated[int, typer.Option(min=1, help="Parallel questions")] = 1,
    resume: Annotated[
        bool, typer.Option(help="Skip questions already graded by an interrupted run")
    ] = False,
) -> None:
    """Run the full pipeline on the golden set and grade answers with an LLM judge."""
    from polyrag.evaluation.dataset import load_golden_set
    from polyrag.evaluation.end_to_end import (
        ItemResult,
        Judge,
        end_to_end_markdown,
        evaluate_end_to_end,
    )

    settings = get_settings()
    checkpoint = settings.reports_dir / "e2e.partial.jsonl"
    completed: list[ItemResult] = []
    if resume and checkpoint.exists():
        completed = [
            ItemResult.model_validate_json(line)
            for line in checkpoint.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        console.print(f"Resuming: {len(completed)} question(s) already graded.")
    elif checkpoint.exists():
        checkpoint.unlink()

    def progress(record: ItemResult) -> None:
        status = "[red]error[/red]" if record.error else f"score={record.correctness}"
        console.print(f"  {record.id:<8} {status}  {record.latency_ms / 1000:.1f} s")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")

    with open_container() as container:
        settings = container.settings
        items = load_golden_set(settings.golden_set)
        if category:
            items = [item for item in items if item.category == category]
        items = items[:limit] if limit else items
        console.print(f"Evaluating {len(items)} questions...")
        notes = []
        if settings.llm_provider is LLMProvider.CLAUDE_CODE:
            notes.append(
                "Run through the Claude Code CLI on a Claude subscription: latency includes "
                "the CLI start-up of every call, and costs are the API-equivalent prices "
                "reported by the CLI (the subscription is what was billed)."
            )
        try:
            report = asyncio.run(
                evaluate_end_to_end(
                    container.assistant,
                    Judge(container.judge_llm, container.database),
                    items,
                    models={
                        "provider": settings.llm_provider.value,
                        "synthesis": settings.llm_model,
                        "agents": settings.llm_fast_model,
                        "judge": settings.judge_model,
                    },
                    notes=notes,
                    concurrency=concurrency,
                    on_progress=progress,
                    completed=completed,
                )
            )
        except LLMUnavailableError as exc:
            console.print(f"[red]Evaluation stopped, LLM provider unavailable:[/red] {exc}")
            console.print("Graded questions are kept: re-run with --resume to continue.")
            raise typer.Exit(code=2) from exc
    checkpoint.unlink(missing_ok=True)
    markdown = end_to_end_markdown(report)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / "e2e.md").write_text(markdown, encoding="utf-8")
    (settings.reports_dir / "e2e.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    console.print(Markdown(markdown))


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    reload: Annotated[bool, typer.Option()] = False,
) -> None:
    """Start the API and the demo UI."""
    import uvicorn

    uvicorn.run("polyrag.api.app:create_app", factory=True, host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
