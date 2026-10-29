from datetime import date

from meetrag.domain.models import Chunk, MeetingFilter, SearchFilter, Source
from meetrag.retrieval.bm25 import BM25Index
from meetrag.retrieval.meetings import MeetingCatalog, MeetingInfo


def _meeting(doc_id: str, day: str, kind: str, people: list[str]) -> MeetingInfo:
    return MeetingInfo(
        doc_id=doc_id,
        title=doc_id,
        date=date.fromisoformat(day),
        meeting_type=kind,
        participants=people,
    )


CATALOG = MeetingCatalog(
    [
        _meeting(
            "meetings/sc1", "2026-05-12", "steering committee", ["Sophie Bernard", "Hugo Lambert"]
        ),
        _meeting("meetings/ws", "2026-03-16", "workshop", ["Léa Moreau", "Hugo Lambert"]),
        _meeting(
            "meetings/sc3", "2026-10-06", "steering committee", ["Sophie Bernard", "Marc Lefèvre"]
        ),
    ]
)


def test_describe_lists_types_period_and_people() -> None:
    text = CATALOG.describe()
    assert "3 meetings from 2026-03-16 to 2026-10-06" in text
    assert "steering committee (2)" in text and "Léa Moreau" in text


def test_partial_and_accent_free_names_are_resolved() -> None:
    assert CATALOG.resolve_people(["sophie"]) == {"Sophie Bernard"}
    assert CATALOG.resolve_people(["Lea Moreau", "nobody"]) == {"Léa Moreau"}


def test_latest_steering_committee() -> None:
    where, note = CATALOG.resolve(MeetingFilter(meeting_type="steering", latest_only=True))
    assert where.doc_ids == frozenset({"meetings/sc3"})
    assert note is not None and "2026-10-06" in note


def test_date_range_and_speaker() -> None:
    where, _ = CATALOG.resolve(
        MeetingFilter(date_from="2026-04-01", date_to="2026-12-31", speakers=["Sophie"])
    )
    assert where.doc_ids == frozenset({"meetings/sc1", "meetings/sc3"})
    assert where.speakers == frozenset({"Sophie Bernard"})


def test_impossible_filter_falls_back_to_all_meetings() -> None:
    where, note = CATALOG.resolve(MeetingFilter(date_from="2030-01-01"))
    assert where == SearchFilter(source=Source.MEETINGS)
    assert note is not None and "searching all meetings" in note


def test_empty_filter_is_source_only() -> None:
    assert CATALOG.resolve(None) == (SearchFilter(source=Source.MEETINGS), None)


def _chunk(chunk_id: str, speakers: list[str], text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id=chunk_id.split("#")[0], source=Source.MEETINGS,
        title="t", section="", text=text, metadata={"speakers": speakers},
    )  # fmt: skip


def test_bm25_applies_speaker_and_document_filters() -> None:
    chunks = [
        _chunk("meetings/sc1#0", ["Sophie Bernard"], "budget approved 480 000"),
        _chunk("meetings/sc3#0", ["Hugo Lambert"], "budget consumed 462 000"),
    ]
    index = BM25Index(chunks)
    by_sophie = index.search(
        "budget", limit=5, where=SearchFilter(speakers=frozenset({"Sophie Bernard"}))
    )
    assert [h.chunk.chunk_id for h in by_sophie] == ["meetings/sc1#0"]
    in_sc3 = index.search(
        "budget", limit=5, where=SearchFilter(doc_ids=frozenset({"meetings/sc3"}))
    )
    assert [h.chunk.chunk_id for h in in_sc3] == ["meetings/sc3#0"]
