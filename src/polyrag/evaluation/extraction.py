"""Evaluation of meeting-record extraction against hand-annotated gold records.

Action items are matched one-to-one when the owner is the same person and the task
descriptions are semantically close (cosine similarity of their embeddings above a threshold),
which tolerates paraphrases such as "send the charter" / "share the project charter".
Precision and recall are reported for action items, recall for decisions (what counts as a
decision is fuzzier, so extra decisions are not penalised), and due-date accuracy on matched
action items.
"""

from __future__ import annotations

import json
import math
import unicodedata
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from polyrag.ingestion.meeting_records import ActionItem

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from polyrag.ingestion.meeting_records import MeetingInsights

    Similarity = Callable[[Sequence[str], Sequence[str]], list[list[float]]]


class GoldMeeting(BaseModel):
    meeting_id: str
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)


class ExtractionScores(BaseModel):
    meetings: int
    threshold: float
    gold_action_items: int
    extracted_action_items: int
    matched_action_items: int
    action_precision: float
    action_recall: float
    action_f1: float
    due_date_accuracy: float
    gold_decisions: int
    extracted_decisions: int
    decision_recall: float
    missed: list[str]
    unexpected: list[str]


def load_gold(path: Path) -> list[GoldMeeting]:
    with path.open(encoding="utf-8") as handle:
        return [GoldMeeting.model_validate(json.loads(line)) for line in handle if line.strip()]


def embedding_similarity(embed: Callable[[Sequence[str]], list[list[float]]]) -> Similarity:
    """Build a cosine-similarity function from any document embedder."""

    def similarity(left: Sequence[str], right: Sequence[str]) -> list[list[float]]:
        if not left or not right:
            return [[] for _ in left]
        vectors = embed([*left, *right])
        unit = [_normalise(v) for v in vectors]
        a, b = unit[: len(left)], unit[len(left) :]
        return [[sum(x * y for x, y in zip(u, v, strict=True)) for v in b] for u in a]

    return similarity


def _normalise(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def _person(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().strip()


def _match(
    gold: Sequence[str], found: Sequence[str], scores: list[list[float]], threshold: float,
    allowed: Callable[[int, int], bool],
) -> list[tuple[int, int]]:  # fmt: skip
    """Greedy one-to-one matching on the highest similarities above the threshold."""
    candidates = sorted(
        (
            (scores[i][j], i, j)
            for i in range(len(gold))
            for j in range(len(found))
            if scores[i][j] >= threshold and allowed(i, j)
        ),
        reverse=True,
    )
    used_gold: set[int] = set()
    used_found: set[int] = set()
    pairs = []
    for _, i, j in candidates:
        if i not in used_gold and j not in used_found:
            used_gold.add(i)
            used_found.add(j)
            pairs.append((i, j))
    return pairs


def evaluate_extraction(
    gold: Sequence[GoldMeeting],
    extracted: dict[str, MeetingInsights],
    similarity: Similarity,
    *,
    threshold: float = 0.7,
) -> ExtractionScores:
    gold_actions = extracted_actions = matched = due_ok = 0
    gold_decisions = extracted_decisions = decisions_found = 0
    missed: list[str] = []
    unexpected: list[str] = []
    for meeting in gold:
        found = extracted.get(meeting.meeting_id)
        actions = found.action_items if found else []
        decisions = [d.decision for d in found.decisions] if found else []
        gold_actions += len(meeting.action_items)
        extracted_actions += len(actions)
        gold_decisions += len(meeting.decisions)
        extracted_decisions += len(decisions)

        expected = meeting.action_items

        def same_owner(
            i: int, j: int, e: list[ActionItem] = expected, a: list[ActionItem] = actions
        ) -> bool:
            return _person(e[i].owner) == _person(a[j].owner)

        scores = similarity([a.task for a in expected], [a.task for a in actions])
        pairs = _match(
            [a.task for a in expected], [a.task for a in actions], scores, threshold, same_owner
        )
        matched += len(pairs)
        due_ok += sum(expected[i].due_date == actions[j].due_date for i, j in pairs)
        matched_gold = {i for i, _ in pairs}
        matched_found = {j for _, j in pairs}
        missed += [
            f"{meeting.meeting_id}: {a.owner}: {a.task}"
            for i, a in enumerate(expected)
            if i not in matched_gold
        ]
        unexpected += [
            f"{meeting.meeting_id}: {a.owner}: {a.task}"
            for j, a in enumerate(actions)
            if j not in matched_found
        ]

        decision_scores = similarity(meeting.decisions, decisions)
        decisions_found += len(
            _match(meeting.decisions, decisions, decision_scores, threshold, lambda i, j: True)
        )

    precision = matched / extracted_actions if extracted_actions else 0.0
    recall = matched / gold_actions if gold_actions else 0.0
    return ExtractionScores(
        meetings=len(gold),
        threshold=threshold,
        gold_action_items=gold_actions,
        extracted_action_items=extracted_actions,
        matched_action_items=matched,
        action_precision=precision,
        action_recall=recall,
        action_f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        due_date_accuracy=due_ok / matched if matched else 0.0,
        gold_decisions=gold_decisions,
        extracted_decisions=extracted_decisions,
        decision_recall=decisions_found / gold_decisions if gold_decisions else 0.0,
        missed=missed,
        unexpected=unexpected,
    )


def extraction_markdown(scores: ExtractionScores, *, model: str) -> str:
    lines = [
        "# Meeting-record extraction",
        "",
        f"{scores.meetings} meetings, hand-annotated gold records, extraction model `{model}`. "
        f"Action items match when the owner is the same and the tasks have a cosine "
        f"similarity of at least {scores.threshold}.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Action items: precision | {scores.action_precision:.1%} "
        f"({scores.matched_action_items}/{scores.extracted_action_items}) |",
        f"| Action items: recall | {scores.action_recall:.1%} "
        f"({scores.matched_action_items}/{scores.gold_action_items}) |",
        f"| Action items: F1 | {scores.action_f1:.1%} |",
        f"| Due dates correct (matched items) | {scores.due_date_accuracy:.1%} |",
        f"| Decisions: recall | {scores.decision_recall:.1%} "
        f"({scores.extracted_decisions} extracted for {scores.gold_decisions} expected) |",
    ]
    if scores.missed:
        lines += ["", "Missed action items:", "", *(f"- {m}" for m in scores.missed)]
    if scores.unexpected:
        lines += ["", "Extra action items:", "", *(f"- {u}" for u in scores.unexpected)]
    return "\n".join(lines) + "\n"
