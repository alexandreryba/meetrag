"""Meeting catalog: turns what the router understood ("the last steering committee", "what
Sophie said in March") into concrete search filters over transcript chunks."""

from __future__ import annotations

import unicodedata
from collections import Counter
from datetime import date
from typing import TYPE_CHECKING

from pydantic import BaseModel

from meetrag.domain.models import SearchFilter, Source

if TYPE_CHECKING:
    from collections.abc import Iterable

    from meetrag.domain.models import Chunk, MeetingFilter


class MeetingInfo(BaseModel):
    doc_id: str
    title: str
    date: date
    meeting_type: str
    participants: list[str]


def _fold(value: str) -> str:
    """Lower-case, accent-free form used to compare names and types."""
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower().strip()


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


class MeetingCatalog:
    def __init__(self, meetings: Iterable[MeetingInfo]) -> None:
        self.meetings = sorted(meetings, key=lambda m: (m.date, m.doc_id))
        self.people = sorted({p for m in self.meetings for p in m.participants})

    @classmethod
    def from_chunks(cls, chunks: Iterable[Chunk]) -> MeetingCatalog:
        seen: dict[str, MeetingInfo] = {}
        for chunk in chunks:
            if chunk.source is not Source.MEETINGS or chunk.doc_id in seen:
                continue
            meta = chunk.metadata
            seen[chunk.doc_id] = MeetingInfo(
                doc_id=chunk.doc_id,
                title=chunk.title.rsplit(" (", 1)[0],
                date=date.fromisoformat(meta["date"]),
                meeting_type=meta.get("meeting_type", "meeting"),
                participants=list(meta.get("participants", [])),
            )
        return cls(seen.values())

    def __len__(self) -> int:
        return len(self.meetings)

    def describe(self) -> str:
        """Compact overview given to the router, so it can phrase usable filters."""
        if not self.meetings:
            return "No meeting transcripts are indexed."
        types = Counter(m.meeting_type for m in self.meetings)
        return (
            f"{len(self.meetings)} meetings from {self.meetings[0].date} to "
            f"{self.meetings[-1].date}. Meeting types: "
            + ", ".join(f"{kind} ({count})" for kind, count in sorted(types.items()))
            + ". People: "
            + ", ".join(self.people)
            + "."
        )

    def resolve_people(self, names: Iterable[str]) -> set[str]:
        """Map partial names ("Sophie", "bernard") to the participants' full names."""
        resolved: set[str] = set()
        for name in names:
            tokens = _fold(name).split()
            if not tokens:
                continue
            for person in self.people:
                person_tokens = _fold(person).split()
                if all(token in person_tokens for token in tokens):
                    resolved.add(person)
        return resolved

    def resolve(self, meeting_filter: MeetingFilter | None) -> tuple[SearchFilter, str | None]:
        """Return the search filter and a human-readable note on how it was resolved."""
        base = SearchFilter(source=Source.MEETINGS)
        if meeting_filter is None or meeting_filter.is_empty or not self.meetings:
            return base, None

        candidates = self.meetings
        if meeting_filter.meeting_type:
            wanted = _fold(meeting_filter.meeting_type)
            typed = [
                m for m in candidates if wanted in _fold(m.meeting_type) or wanted in _fold(m.title)
            ]
            candidates = typed or candidates
        start, end = _parse_date(meeting_filter.date_from), _parse_date(meeting_filter.date_to)
        if start:
            candidates = [m for m in candidates if m.date >= start]
        if end:
            candidates = [m for m in candidates if m.date <= end]
        speakers = self.resolve_people(meeting_filter.speakers) or None
        if speakers:
            candidates = [m for m in candidates if speakers.intersection(m.participants)]
        if meeting_filter.latest_only and candidates:
            candidates = [max(candidates, key=lambda m: (m.date, m.doc_id))]
        if not candidates:  # an over-constrained filter must not hide every transcript
            return base, "no meeting matches the filter; searching all meetings"

        doc_ids = frozenset(m.doc_id for m in candidates)
        note = f"{len(doc_ids)} meeting(s)"
        if len(candidates) <= 3:
            note += ": " + ", ".join(f"{m.title} ({m.date})" for m in candidates)
        if speakers:
            note += "; speakers: " + ", ".join(sorted(speakers))
        all_meetings = len(doc_ids) == len(self.meetings)
        return (
            SearchFilter(
                source=Source.MEETINGS,
                doc_ids=None if all_meetings else doc_ids,
                speakers=frozenset(speakers) if speakers else None,
            ),
            note,
        )
