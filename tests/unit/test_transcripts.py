from pathlib import Path

import pytest

from meetrag.domain.models import Source
from meetrag.ingestion.transcripts import (
    TranscriptChunker,
    Turn,
    clean_turns,
    parse_transcript,
    redact_pii,
)

TEXT = """---
title: Steering committee #9
date: 2026-05-12
type: steering committee
participants:
  - Sophie Bernard | CFO | Voltaria
  - Hugo Lambert | Engagement Manager | Consulting team
---
[00:00:05] Hugo Lambert: Hello everyone. Um, let's start.
[00:00:20] Sophie Bernard: The budget is 480 000 euros.
It covers the campaign tooling.
Next point: the pilot.
[00:01:02] Hugo Lambert: Agreed.
"""

VTT = """WEBVTT

NOTE
title: Atelier données
participants:
- Thomas Girard | Data Platform Lead | Voltaria

00:00:04.000 --> 00:00:09.000
<v Thomas Girard>Bonjour à tous, euh, on commence.</v>

00:00:10.500 --> 00:00:15.000
<v Léa Moreau>Le the rapport est prêt.</v>

00:00:16.000 --> 00:00:18.000
Et il est partagé.
"""

SRT = """1
00:00:03,000 --> 00:00:12,000
Karim Haddad: The AUC is 0.82.

2
00:00:13,000 --> 00:00:20,000
Claire Dubois: You can use the billing ticket count.
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_text_transcript_with_front_matter(tmp_path: Path) -> None:
    t = parse_transcript(_write(tmp_path, "sc9.md", TEXT))
    assert t.title == "Steering committee #9" and t.meeting_type == "steering committee"
    assert t.role_of("Sophie Bernard") == "CFO"
    assert [turn.speaker for turn in t.turns] == ["Hugo Lambert", "Sophie Bernard", "Hugo Lambert"]
    # continuation lines are glued to the turn, "Next point:" is not mistaken for a speaker
    assert (
        t.turns[1].text
        == "The budget is 480 000 euros. It covers the campaign tooling. Next point: the pilot."
    )
    assert t.turns[0].text == "Hello everyone. let's start."
    assert t.turns[2].start_s == 62


def test_vtt_with_voice_tags_and_note_metadata(tmp_path: Path) -> None:
    t = parse_transcript(_write(tmp_path, "2026-03-16_data-workshop.vtt", VTT))
    assert t.date.isoformat() == "2026-03-16" and t.meeting_type == "workshop"
    assert t.language == "fr"
    assert [p.name for p in t.participants] == ["Thomas Girard", "Léa Moreau"]
    assert t.turns[0].text == "Bonjour à tous, on commence."  # filler removed
    # a cue without speaker continues the previous speaker, stutters are collapsed
    assert t.turns[1].speaker == "Léa Moreau" and "Le the rapport" in t.turns[1].text
    assert t.turns[1].text.endswith("Et il est partagé.")


def test_srt_uses_speaker_prefixes_and_file_name(tmp_path: Path) -> None:
    t = parse_transcript(_write(tmp_path, "2026-06-02_churn-model-review.srt", SRT))
    assert t.title == "Churn model review" and t.meeting_type == "review"
    assert t.turns[1].speaker == "Claire Dubois" and t.turns[1].start_s == 13
    assert t.doc_id == "meetings/2026-06-02-churn-model-review"


def test_missing_date_is_an_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no meeting date"):
        parse_transcript(_write(tmp_path, "notes.txt", "Hugo Lambert: hi"))


def test_cleaning_merges_turns_and_collapses_stutters() -> None:
    turns = [
        Turn(speaker="A", start_s=1, text="the the plan, uh, is ready"),
        Turn(speaker="A", start_s=5, text="and shared"),
        Turn(speaker="B", start_s=9, text="um"),
    ]
    assert clean_turns(turns) == [
        Turn(speaker="A", start_s=1, text="the plan, is ready and shared")
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("call 06 12 34 56 78 or +33 6 12 34 56 78", "call [phone] or [phone]"),
        ("write to claire.dubois@voltaria.example", "write to [email]"),
        ("a budget of 450 000 euros on 2026-03-02", "a budget of 450 000 euros on 2026-03-02"),
        ("AUC 0,81 then 0.82 for 14,000 customers", "AUC 0,81 then 0.82 for 14,000 customers"),
    ],
)
def test_pii_redaction_keeps_figures(text: str, expected: str) -> None:
    assert redact_pii(text) == expected


def test_chunker_keeps_turns_whole_and_records_speakers(tmp_path: Path) -> None:
    t = parse_transcript(_write(tmp_path, "sc9.md", TEXT))
    chunks = TranscriptChunker(max_chars=120, overlap_turns=1).split(t)
    assert len(chunks) == 3
    assert all(c.source is Source.MEETINGS and c.doc_id == t.doc_id for c in chunks)
    assert chunks[0].text.startswith("[00:00:05] Hugo Lambert (Engagement Manager): Hello")
    assert chunks[0].metadata["speakers"] == ["Hugo Lambert"]
    assert chunks[0].metadata["date"] == "2026-05-12"
    assert chunks[0].title == "Steering committee #9 (2026-05-12)"
    assert chunks[1].section == "00:00:20-00:00:20"


def test_chunker_overlaps_on_the_boundary_turn(tmp_path: Path) -> None:
    t = parse_transcript(_write(tmp_path, "sc9.md", TEXT))
    first, second = TranscriptChunker(max_chars=10_000).split(t)[0], None
    assert second is None and first.metadata["speakers"] == ["Hugo Lambert", "Sophie Bernard"]
    windows = TranscriptChunker(max_chars=220, overlap_turns=1).split(t)
    assert windows[0].text.splitlines()[-1] in windows[1].text
