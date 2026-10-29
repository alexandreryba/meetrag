from __future__ import annotations

from datetime import date
from pathlib import Path

from meetrag.evaluation.extraction import GoldMeeting, evaluate_extraction
from meetrag.infrastructure.sql import SQLiteDatabase
from meetrag.ingestion.meeting_records import (
    ActionItem,
    Decision,
    MeetingExtractor,
    MeetingInsights,
    load_cached_insights,
    write_meeting_records,
)
from meetrag.ingestion.transcripts import Participant, Transcript, Turn
from tests.conftest import ScriptedLLM

INSIGHTS = MeetingInsights(
    summary="Budget extended.",
    decisions=[Decision(decision="Budget raised to EUR 480,000", decided_by="Sophie Bernard")],
    action_items=[ActionItem(owner="Hugo Lambert", task="Send the minutes", due_date="2026-05-14")],
    risks=["Summer availability"],
)


def _transcript(text: str = "The budget goes to 480 000 euros.") -> Transcript:
    return Transcript(
        meeting_id="2026-05-12-sc1",
        title="Steering committee #1",
        date=date(2026, 5, 12),
        meeting_type="steering committee",
        language="en",
        participants=[
            Participant(name="Sophie Bernard", role="CFO"),
            Participant(name="Hugo Lambert"),
        ],
        turns=[
            Turn(speaker="Sophie Bernard", start_s=5, text=text),
            Turn(speaker="Hugo Lambert", start_s=125, text="Noted."),
        ],
    )


async def test_extraction_is_cached_and_invalidated_when_the_transcript_changes(
    tmp_path: Path,
) -> None:
    llm = ScriptedLLM(handlers={MeetingInsights: lambda _: INSIGHTS})
    extractor = MeetingExtractor(llm, tmp_path)
    insights, fresh = await extractor.extract_all([_transcript()])
    assert fresh == 1 and insights["2026-05-12-sc1"] == INSIGHTS
    _, fresh_again = await extractor.extract_all([_transcript()])
    assert fresh_again == 0  # served from the cache, no model call
    assert sum(kind == "MeetingInsights" for kind, _ in llm.calls) == 1
    assert load_cached_insights(tmp_path, [_transcript("A different discussion.")]) == {}


def test_meeting_tables_are_queryable(tmp_path: Path) -> None:
    db_path = tmp_path / "records.db"
    counts = write_meeting_records(db_path, [_transcript()], {"2026-05-12-sc1": INSIGHTS})
    assert counts == {
        "meetings": 1,
        "meeting_participants": 2,
        "decisions": 1,
        "action_items": 1,
        "risks": 1,
    }
    database = SQLiteDatabase(db_path)
    row = database.execute("SELECT duration_min, summary FROM meetings").rows[0]
    assert row == [3, "Budget extended."]
    owner = database.execute("SELECT owner, due_date FROM action_items").rows
    assert owner == [["Hugo Lambert", "2026-05-14"]]


def test_tables_exist_before_extraction(tmp_path: Path) -> None:
    db_path = tmp_path / "records.db"
    counts = write_meeting_records(db_path, [_transcript()])
    assert counts["meetings"] == 1 and counts["action_items"] == 0


def _similarity(left, right):  # type: ignore[no-untyped-def]
    """Word-overlap similarity, enough to test the matching logic."""

    def jaccard(a: str, b: str) -> float:
        x, y = set(a.lower().split()), set(b.lower().split())
        return len(x & y) / len(x | y)

    return [[jaccard(a, b) for b in right] for a in left]


def test_extraction_scores_match_owner_and_paraphrased_task() -> None:
    gold = [
        GoldMeeting(
            meeting_id="2026-05-12-sc1",
            decisions=["Budget raised to EUR 480,000", "Dashboard postponed"],
            action_items=[
                ActionItem(owner="Hugo Lambert", task="Send the minutes", due_date="2026-05-14"),
                ActionItem(owner="Sophie Bernard", task="Present the budget to the board"),
            ],
        )
    ]
    found = INSIGHTS.model_copy(
        update={
            "action_items": [
                ActionItem(
                    owner="hugo lambert", task="send the meeting minutes", due_date="2026-05-14"
                ),
                ActionItem(
                    owner="Karim Haddad", task="Send the minutes"
                ),  # right task, wrong owner
            ]
        }
    )
    scores = evaluate_extraction(gold, {"2026-05-12-sc1": found}, _similarity, threshold=0.5)
    assert scores.matched_action_items == 1
    assert scores.action_precision == 0.5 and scores.action_recall == 0.5
    assert scores.due_date_accuracy == 1.0
    assert scores.decision_recall == 0.5
    assert len(scores.missed) == 1 and len(scores.unexpected) == 1
