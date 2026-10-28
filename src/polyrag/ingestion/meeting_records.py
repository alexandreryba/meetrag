"""Structured meeting records: metadata parsed from the transcripts (deterministic) and
decisions, action items, risks and summaries extracted by an LLM (cached).

Extractions are cached per meeting in JSON files keyed by a hash of the transcript and of the
extraction prompt version: re-indexing never pays twice for an unchanged meeting, and the cache
can be committed so that a fresh checkout, the CI and the Docker image get the records
without calling a model.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from polyrag.domain.ports import LLM
    from polyrag.ingestion.transcripts import Transcript

logger = logging.getLogger(__name__)

EXTRACTION_VERSION = "1"  # bump when the prompt or the schema changes

EXTRACTION_SYSTEM = """\
You turn a meeting transcript into a structured record for the engagement's database.

- summary: 3 to 5 sentences on what was discussed and concluded.
- decisions: only what the participants explicitly agreed or approved in this meeting, with \
the person who made or approved the decision when stated.
- action_items: only commitments where a named participant agrees to do something; owner \
is the participant's full name as listed; due_date in YYYY-MM-DD when a date or delay is \
stated (resolve relative dates such as "next Friday" from the meeting date), otherwise null.
- risks: risks, blockers or concerns raised.
Write every field in English, even when the meeting is held in French. Do not infer \
anything that was not said."""

MEETING_SCHEMA = """
DROP TABLE IF EXISTS risks;
DROP TABLE IF EXISTS action_items;
DROP TABLE IF EXISTS decisions;
DROP TABLE IF EXISTS meeting_participants;
DROP TABLE IF EXISTS meetings;
CREATE TABLE meetings (
    meeting_id TEXT PRIMARY KEY,         -- file-based id, e.g. '2026-05-12-steering-committee-1'
    title TEXT NOT NULL,
    date DATE NOT NULL,
    meeting_type TEXT NOT NULL,  -- kick-off, workshop, weekly sync, review, steering committee
    language TEXT NOT NULL,              -- 'fr' | 'en'
    duration_min INTEGER,                -- from the transcript timestamps, NULL if unknown
    summary TEXT                         -- extracted summary, NULL until `polyrag extract` ran
);
CREATE TABLE meeting_participants (
    meeting_id TEXT NOT NULL REFERENCES meetings (meeting_id),
    name TEXT NOT NULL,                  -- full name
    role TEXT,                           -- job title, e.g. 'CFO'
    organisation TEXT,                   -- 'Voltaria' or 'Consulting team'
    spoke INTEGER NOT NULL               -- 1 if the person spoke during the meeting
);
CREATE TABLE decisions (
    meeting_id TEXT NOT NULL REFERENCES meetings (meeting_id),
    decision TEXT NOT NULL,              -- what was agreed, in English
    decided_by TEXT                      -- person who made or approved it, NULL if not stated
);
CREATE TABLE action_items (
    meeting_id TEXT NOT NULL REFERENCES meetings (meeting_id),
    owner TEXT NOT NULL,                 -- full name of the person who committed
    task TEXT NOT NULL,                  -- in English
    due_date DATE                        -- NULL when no deadline was given
);
CREATE TABLE risks (
    meeting_id TEXT NOT NULL REFERENCES meetings (meeting_id),
    risk TEXT NOT NULL
);
"""


class Decision(BaseModel):
    decision: str
    decided_by: str | None = None


class ActionItem(BaseModel):
    owner: str = Field(description="Full name of the participant who committed")
    task: str
    due_date: str | None = Field(default=None, description="YYYY-MM-DD, or null")


class MeetingInsights(BaseModel):
    summary: str
    decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class InsightsRecord(BaseModel):
    """What is stored in the cache, with its provenance."""

    meeting_id: str
    transcript_sha256: str
    extraction_version: str
    model: str
    extracted_at: str
    insights: MeetingInsights


def transcript_fingerprint(transcript: Transcript) -> str:
    payload = f"{EXTRACTION_VERSION}\n{transcript.render()}".encode()
    return hashlib.sha256(payload).hexdigest()


def load_cached_insights(
    cache_dir: Path, transcripts: Sequence[Transcript]
) -> dict[str, MeetingInsights]:
    """Insights whose cache entry still matches the current transcript."""
    found: dict[str, MeetingInsights] = {}
    for transcript in transcripts:
        path = cache_dir / f"{transcript.meeting_id}.json"
        if not path.exists():
            continue
        try:
            record = InsightsRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError:
            logger.warning("ignoring unreadable insights cache %s", path)
            continue
        if record.transcript_sha256 == transcript_fingerprint(transcript):
            found[transcript.meeting_id] = record.insights
    return found


class MeetingExtractor:
    def __init__(self, llm: LLM, cache_dir: Path, *, concurrency: int = 3) -> None:
        self.llm = llm
        self.cache_dir = cache_dir
        self._semaphore = asyncio.Semaphore(concurrency)

    async def _extract(self, transcript: Transcript) -> MeetingInsights:
        async with self._semaphore:
            insights = await self.llm.structured(
                system=EXTRACTION_SYSTEM,
                prompt=transcript.render(),
                schema=MeetingInsights,
                effort="medium",
            )
        record = InsightsRecord(
            meeting_id=transcript.meeting_id,
            transcript_sha256=transcript_fingerprint(transcript),
            extraction_version=EXTRACTION_VERSION,
            model=self.llm.model,
            extracted_at=datetime.now(UTC).isoformat(timespec="seconds"),
            insights=insights,
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / f"{transcript.meeting_id}.json").write_text(
            record.model_dump_json(indent=2), encoding="utf-8"
        )
        return insights

    async def extract_all(
        self, transcripts: Sequence[Transcript], *, force: bool = False
    ) -> tuple[dict[str, MeetingInsights], int]:
        """Return insights for every transcript and how many were freshly extracted."""
        cached = {} if force else load_cached_insights(self.cache_dir, transcripts)
        missing = [t for t in transcripts if t.meeting_id not in cached]
        fresh = await asyncio.gather(*(self._extract(t) for t in missing))
        return {**cached, **{t.meeting_id: i for t, i in zip(missing, fresh, strict=True)}}, len(
            missing
        )


def _duration_minutes(transcript: Transcript) -> int | None:
    stamps = [t.start_s for t in transcript.turns if t.start_s is not None]
    return round((max(stamps) - min(stamps)) / 60) + 1 if len(stamps) > 1 else None


def write_meeting_records(
    database_path: Path,
    transcripts: Sequence[Transcript],
    insights: dict[str, MeetingInsights] | None = None,
) -> dict[str, int]:
    """(Re)create the meeting tables in the analytics database. Returns row counts."""
    insights = insights or {}
    with closing(sqlite3.connect(database_path)) as connection:
        connection.executescript(MEETING_SCHEMA)
        for t in transcripts:
            found = insights.get(t.meeting_id)
            speakers = {turn.speaker for turn in t.turns}
            connection.execute(
                "INSERT INTO meetings VALUES (?, ?, ?, ?, ?, ?, ?)",
                (t.meeting_id, t.title, t.date.isoformat(), t.meeting_type, t.language,
                 _duration_minutes(t), found.summary if found else None),
            )  # fmt: skip
            connection.executemany(
                "INSERT INTO meeting_participants VALUES (?, ?, ?, ?, ?)",
                [(t.meeting_id, p.name, p.role, p.organisation, int(p.name in speakers))
                 for p in t.participants],
            )  # fmt: skip
            if found is None:
                continue
            connection.executemany(
                "INSERT INTO decisions VALUES (?, ?, ?)",
                [(t.meeting_id, d.decision, d.decided_by) for d in found.decisions],
            )
            connection.executemany(
                "INSERT INTO action_items VALUES (?, ?, ?, ?)",
                [(t.meeting_id, a.owner, a.task, a.due_date) for a in found.action_items],
            )
            connection.executemany(
                "INSERT INTO risks VALUES (?, ?)", [(t.meeting_id, r) for r in found.risks]
            )
        connection.commit()
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
            for table in ("meetings", "meeting_participants", "decisions", "action_items", "risks")
        }
    return counts
