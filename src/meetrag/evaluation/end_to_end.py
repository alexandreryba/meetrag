"""End-to-end evaluation with an LLM judge.

Metrics:
* answer accuracy: judge compares the answer with the reference (correct / partial / wrong);
  for analytics questions the reference SQL is executed, so the reference is exact;
* faithfulness: share of the answer's claims supported by the retrieved evidence;
* routing precision / recall against the expected sources;
* abstention accuracy on out-of-scope questions, and false-abstention rate on the others;
* latency (p50 / p95), cost and number of LLM calls per question.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from meetrag.agents.synthesis import format_evidence
from meetrag.domain.models import Source
from meetrag.domain.ports import LLMError, LLMUnavailableError, QueryError
from meetrag.evaluation.metrics import mean, percentile, set_precision_recall

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from meetrag.domain.models import AnswerResult
    from meetrag.domain.ports import LLM, SqlDatabase
    from meetrag.evaluation.dataset import GoldenItem
    from meetrag.service import Assistant

logger = logging.getLogger(__name__)

CORRECTNESS_SYSTEM = """\
You grade the answers of a company assistant against a reference answer.
- "correct": the answer states every key fact of the reference and contradicts none of them. \
Extra details are fine if they do not contradict the reference. Minor rounding differences \
on computed figures are fine.
- "partial": some key facts are missing or one of them is wrong.
- "incorrect": the answer is wrong, misses the point, or declines to answer."""

FAITHFULNESS_SYSTEM = """\
You check a generated answer against the evidence it was given. Split the answer into atomic \
factual claims (ignore citations markers and polite phrasing) and mark each claim as \
supported only if the evidence states it or it follows from it by simple arithmetic."""


class CorrectnessVerdict(BaseModel):
    rationale: str = Field(description="One or two sentences justifying the grade")
    grade: Literal["correct", "partial", "incorrect"]


class ClaimCheck(BaseModel):
    claim: str
    supported: bool


class FaithfulnessVerdict(BaseModel):
    claims: list[ClaimCheck]


_GRADE_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


class ItemResult(BaseModel):
    id: str
    category: str
    question: str
    answer: str = ""
    expected_sources: list[Source]
    routed_sources: list[Source] = Field(default_factory=list)
    routing_precision: float | None = None
    routing_recall: float | None = None
    answerable: bool
    abstained: bool = False
    correctness: float | None = None
    correctness_rationale: str | None = None
    faithfulness: float | None = None
    unsupported_claims: list[str] = Field(default_factory=list)
    grounded: bool = False
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    llm_calls: int = 0
    error: str | None = None


class EndToEndReport(BaseModel):
    created_at: str
    models: dict[str, str]
    notes: list[str] = Field(default_factory=list)
    questions: int
    summary: dict[str, float]
    by_category: dict[str, dict[str, float]]
    items: list[ItemResult]


class Judge:
    def __init__(self, llm: LLM, database: SqlDatabase) -> None:
        self.llm = llm
        self.database = database

    def reference(self, item: GoldenItem) -> str:
        parts: list[str] = []
        if item.reference_sql:
            try:
                table = self.database.execute(item.reference_sql).to_markdown()
            except QueryError as exc:
                table = f"(reference query failed: {exc})"
            parts.append(f"Result of the reference SQL query:\n{table}")
        if item.reference_answer:
            parts.append(item.reference_answer)
        return "\n\n".join(parts)

    async def correctness(self, item: GoldenItem, answer: str) -> CorrectnessVerdict:
        return await self.llm.structured(
            system=CORRECTNESS_SYSTEM,
            prompt=(
                f"Question: {item.question}\n\nReference:\n{self.reference(item)}"
                f"\n\nAnswer to grade:\n{answer}"
            ),
            schema=CorrectnessVerdict,
            effort="low",
        )

    async def faithfulness(self, result: AnswerResult) -> tuple[float, list[str]]:
        verdict = await self.llm.structured(
            system=FAITHFULNESS_SYSTEM,
            prompt=(f"Evidence:\n{format_evidence(result.evidence)}\n\nAnswer:\n{result.answer}"),
            schema=FaithfulnessVerdict,
            effort="low",
        )
        if not verdict.claims:
            return 1.0, []
        supported = sum(claim.supported for claim in verdict.claims)
        unsupported = [claim.claim for claim in verdict.claims if not claim.supported]
        return supported / len(verdict.claims), unsupported


async def _evaluate_item(assistant: Assistant, judge: Judge, item: GoldenItem) -> ItemResult:
    record = ItemResult(
        id=item.id,
        category=item.category,
        question=item.question,
        expected_sources=item.expected_sources,
        answerable=item.answerable,
    )
    try:
        result = await assistant.ask(item.question)
    except LLMUnavailableError:
        raise  # no point grading the remaining questions
    except Exception as exc:  # an eval run must report failures, not stop on them
        logger.exception("question %s failed", item.id)
        record.error = f"{type(exc).__name__}: {exc}"
        return record

    record.answer = result.answer
    record.routed_sources = result.routed_sources
    record.abstained = result.abstained
    record.grounded = result.grounded
    record.latency_ms = result.latency_ms
    record.cost_usd = result.usage.cost_usd
    record.llm_calls = result.usage.llm_calls
    if not item.answerable:
        return record

    record.routing_precision, record.routing_recall = set_precision_recall(
        result.routed_sources, item.expected_sources
    )
    if result.abstained:
        record.correctness = 0.0
        record.correctness_rationale = "abstained on an answerable question"
        return record
    try:
        verdict = await judge.correctness(item, result.answer)
        record.correctness = _GRADE_SCORE[verdict.grade]
        record.correctness_rationale = verdict.rationale
        record.faithfulness, record.unsupported_claims = await judge.faithfulness(result)
    except LLMError as exc:
        record.error = f"judge failed: {exc}"
    return record


def _summary(items: Sequence[ItemResult]) -> dict[str, float]:
    ok = [i for i in items if i.error is None]
    answerable = [i for i in ok if i.answerable]
    out_of_scope = [i for i in ok if not i.answerable]
    graded = [i.correctness for i in answerable if i.correctness is not None]
    faithfulness = [i.faithfulness for i in answerable if i.faithfulness is not None]
    latencies = [i.latency_ms / 1000 for i in ok]
    return {
        "questions": float(len(items)),
        "errors": float(len(items) - len(ok)),
        "answer_accuracy": mean(graded),
        "strict_accuracy": mean([float(c == 1.0) for c in graded]),
        "faithfulness": mean(faithfulness),
        "routing_precision": mean([i.routing_precision or 0.0 for i in answerable]),
        "routing_recall": mean([i.routing_recall or 0.0 for i in answerable]),
        "abstention_accuracy": mean([float(i.abstained) for i in out_of_scope]),
        "false_abstention_rate": mean([float(i.abstained) for i in answerable]),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "cost_per_question_usd": mean([i.cost_usd for i in ok]),
        "llm_calls_per_question": mean([float(i.llm_calls) for i in ok]),
    }


async def evaluate_end_to_end(
    assistant: Assistant,
    judge: Judge,
    items: Sequence[GoldenItem],
    *,
    models: dict[str, str],
    notes: Sequence[str] = (),
    concurrency: int = 1,
    on_progress: Callable[[ItemResult], None] | None = None,
    completed: Sequence[ItemResult] = (),
) -> EndToEndReport:
    """Run every question not already in ``completed`` (results of an interrupted run).
    Keep ``concurrency=1`` when the latency figures matter."""
    semaphore = asyncio.Semaphore(concurrency)
    done = {record.id: record for record in completed}

    async def run(item: GoldenItem) -> ItemResult:
        async with semaphore:
            record = await _evaluate_item(assistant, judge, item)
        if on_progress:
            on_progress(record)
        return record

    fresh = await asyncio.gather(*(run(item) for item in items if item.id not in done))
    by_id = {**done, **{record.id: record for record in fresh}}
    records = [by_id[item.id] for item in items if item.id in by_id]
    grouped: dict[str, list[ItemResult]] = defaultdict(list)
    for record in records:
        grouped[record.category].append(record)
    return EndToEndReport(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        models=models,
        notes=list(notes),
        questions=len(records),
        summary=_summary(records),
        by_category={category: _summary(group) for category, group in sorted(grouped.items())},
        items=records,
    )


def end_to_end_markdown(report: EndToEndReport) -> str:
    s = report.summary
    lines = [
        "# End-to-end evaluation",
        "",
        f"{report.questions} questions, run on {report.created_at}. Models: "
        + ", ".join(f"{role} `{model}`" for role, model in report.models.items())
        + ".",
        "",
        *(f"> {note}\n" for note in report.notes),
        "| Metric | Value |",
        "|---|---|",
        f"| Answer accuracy (correct = 1, partial = 0.5) | {s['answer_accuracy']:.1%} |",
        f"| Strict accuracy (fully correct) | {s['strict_accuracy']:.1%} |",
        f"| Faithfulness (claims supported by evidence) | {s['faithfulness']:.1%} |",
        f"| Routing recall / precision | {s['routing_recall']:.1%} / "
        f"{s['routing_precision']:.1%} |",
        f"| Abstention on out-of-scope questions | {s['abstention_accuracy']:.1%} |",
        f"| False abstention on answerable questions | {s['false_abstention_rate']:.1%} |",
        f"| Latency p50 / p95 | {s['latency_p50_s']:.1f} s / {s['latency_p95_s']:.1f} s |",
        f"| Cost per question | ${s['cost_per_question_usd']:.4f} |",
        f"| LLM calls per question | {s['llm_calls_per_question']:.1f} |",
        f"| Errors | {int(s['errors'])} |",
        "",
        "## By category",
        "",
        "| Category | N | Accuracy | Faithfulness | Routing recall | Abstention | p50 (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for category, c in report.by_category.items():
        if category == "out_of_scope":
            cells = ["-", "-", "-", f"{c['abstention_accuracy']:.1%}"]
        else:
            cells = [
                f"{c['answer_accuracy']:.1%}",
                f"{c['faithfulness']:.1%}",
                f"{c['routing_recall']:.1%}",
                "-",
            ]
        row = [category, str(int(c["questions"])), *cells, f"{c['latency_p50_s']:.1f}"]
        lines.append("| " + " | ".join(row) + " |")
    failures = [
        i for i in report.items if i.error or (i.correctness is not None and i.correctness < 1)
    ]
    if failures:
        lines += ["", "## Questions not fully correct", ""]
        for item in failures:
            reason = item.error or item.correctness_rationale or ""
            lines.append(f"- **{item.id}** ({item.correctness}): {reason}")
    return "\n".join(lines) + "\n"
