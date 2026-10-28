"""Meeting transcripts: parsing, cleaning and turn-aware chunking.

Supported inputs, as exported by the usual meeting tools:

* plain text / Markdown, one turn per line: ``[00:12:34] Sophie Bernard: ...``,
  ``Sophie Bernard (00:12:34): ...`` or ``Sophie Bernard: ...``, with an optional front
  matter (``title``, ``date``, ``type``, ``language``, ``participants``);
* WebVTT (Teams, Zoom, Google Meet): ``<v Speaker>`` voice tags or ``Speaker:`` prefixes,
  metadata in ``NOTE`` blocks;
* SRT subtitles with ``Speaker:`` prefixes.

The meeting date and title fall back to the file name (``2026-03-02_kick-off.vtt``).
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from polyrag.domain.models import Chunk, Source

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

TRANSCRIPT_SUFFIXES = {".txt", ".md", ".vtt", ".srt"}

_TS = r"\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?"
_TIMED_LINES = (
    re.compile(
        rf"^\[?(?P<ts>{_TS})\]?\s*[-–]?\s*(?P<speaker>[^:\[\]()]{{1,60}}?)\s*:\s*(?P<text>.*)$"  # noqa: RUF001 (en dash)
    ),
    re.compile(
        rf"^(?P<speaker>[^:\[\]()]{{1,60}}?)\s*[(\[](?P<ts>{_TS})[)\]]\s*:?\s*(?P<text>.*)$"
    ),
)
_UNTIMED_LINE = re.compile(r"^(?P<speaker>[A-ZÀ-Ý][\w.'’\- ]{0,59}?)\s*:\s*(?P<text>.+)$")  # noqa: RUF001
_CUE_TIMING = re.compile(rf"^(?P<ts>{_TS})\s*-->\s*{_TS}")
_VOICE = re.compile(r"<v(?:\.[\w.-]+)?\s+([^>]+)>(.*?)(?:</v>|$)", re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_FILE_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})[_-]?(.*)$")

_FILLERS = re.compile(r"\b(?:u+m+|u+h+|erm|uhm|hm+|euh+|heu+|hum+)\b[,.]?\s*", re.IGNORECASE)
_REPEATS = re.compile(r"\b(\w+)(?:\s+\1\b)+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Phone candidates start with "+" or "0" (international or national format); amounts such as
# "450 000 euros" or dates never do. The digit count is checked in ``redact_pii``.
_PHONE = re.compile(r"(?<![\w+(])(?:\+|\(?0)[\d\s().-]{7,20}\d(?!\w)")

_MEETING_TYPES = (
    (r"steering|comit[ée] de pilotage|\bcopil\b", "steering committee"),
    (r"kick-?off|lancement", "kick-off"),
    (r"workshop|atelier", "workshop"),
    (r"weekly|\bsync\b|point hebdo", "weekly sync"),
    (r"review|revue", "review"),
)
_FRENCH_HINTS = frozenset(
    [
        "le",
        "la",
        "les",
        "des",
        "est",
        "nous",
        "vous",
        "pour",
        "que",
        "une",
        "sur",
        "avec",
        "pas",
        "dans",
    ]
)
_ENGLISH_HINTS = frozenset(
    ["the", "is", "are", "we", "you", "for", "that", "with", "on", "not", "in", "this"]
)


class Participant(BaseModel):
    name: str
    role: str | None = None
    organisation: str | None = None


class Turn(BaseModel):
    speaker: str
    start_s: float | None = None
    text: str


class Transcript(BaseModel):
    meeting_id: str
    title: str
    date: date
    meeting_type: str
    language: str
    participants: list[Participant] = Field(default_factory=list)
    turns: list[Turn]
    path: str | None = None

    @property
    def doc_id(self) -> str:
        return f"{Source.MEETINGS.value}/{self.meeting_id}"

    @property
    def display_title(self) -> str:
        return f"{self.title} ({self.date.isoformat()})"

    def role_of(self, name: str) -> str | None:
        return next((p.role for p in self.participants if p.name == name), None)

    def render(self) -> str:
        """Plain-text rendering, used for LLM extraction and content hashing."""
        header = [
            f"Meeting: {self.title}",
            f"Date: {self.date.isoformat()}",
            f"Type: {self.meeting_type}",
            "Participants: "
            + "; ".join(
                ", ".join(x for x in (p.name, p.role, p.organisation) if x)
                for p in self.participants
            ),
        ]
        return "\n".join(header + [""] + [format_turn(t, self) for t in self.turns])


def format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def format_turn(turn: Turn, transcript: Transcript) -> str:
    role = transcript.role_of(turn.speaker)
    who = f"{turn.speaker} ({role})" if role else turn.speaker
    stamp = f"[{format_timestamp(turn.start_s)}] " if turn.start_s is not None else ""
    return f"{stamp}{who}: {turn.text}"


def _seconds(stamp: str) -> float:
    parts = [float(p) for p in stamp.replace(",", ".").split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


# --- Parsing ----------------------------------------------------------------------------------


def _front_matter(text: str) -> tuple[dict[str, str | list[str]], str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    meta = _key_values(text[3:end].splitlines())
    return meta, text[end + 4 :]


def _key_values(lines: Iterable[str]) -> dict[str, str | list[str]]:
    meta: dict[str, str | list[str]] = {}
    current: str | None = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("- ") and current:
            values = meta.setdefault(current, [])
            if isinstance(values, list):
                values.append(line[2:].strip())
            continue
        key, sep, value = line.partition(":")
        if sep:
            current = key.strip().lower()
            meta[current] = value.strip() if value.strip() else []
    return meta


def _match_turn(line: str, known_speakers: set[str] | None = None) -> Turn | None:
    for pattern in _TIMED_LINES:
        match = pattern.match(line)
        if match:
            return Turn(
                speaker=match["speaker"].strip(),
                start_s=_seconds(match["ts"]),
                text=match["text"].strip(),
            )
    match = _UNTIMED_LINE.match(line)
    if not match:
        return None
    speaker = match["speaker"].strip()
    # Without a timestamp, "Next point: budget" must not be read as a speaker called
    # "Next point": accept short names only, and known participants when the list exists.
    if len(speaker.split()) > 4 or (known_speakers and speaker not in known_speakers):
        return None
    return Turn(speaker=speaker, start_s=None, text=match["text"].strip())


def _parse_lines(lines: Iterable[str], known_speakers: set[str] | None = None) -> list[Turn]:
    turns: list[Turn] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        turn = _match_turn(line, known_speakers)
        if turn is not None:
            turns.append(turn)
        elif turns:  # continuation of the previous turn
            turns[-1] = turns[-1].model_copy(update={"text": f"{turns[-1].text} {line}"})
    return turns


def _parse_cues(text: str) -> tuple[dict[str, str | list[str]], list[Turn]]:
    """Shared by WebVTT and SRT: blocks of ``[id] / timing / text``."""
    meta: dict[str, str | list[str]] = {}
    turns: list[Turn] = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        lines = [line for line in block.strip().splitlines() if line.strip()]
        if not lines:
            continue
        if lines[0].startswith("NOTE"):
            meta.update(_key_values([lines[0][4:], *lines[1:]]))
            continue
        timing_index = next((i for i, line in enumerate(lines) if _CUE_TIMING.match(line)), None)
        if timing_index is None:
            continue
        timing = _CUE_TIMING.match(lines[timing_index])
        start = _seconds(timing["ts"]) if timing else None
        body = "\n".join(lines[timing_index + 1 :])
        voices = _VOICE.findall(body)
        if voices:
            for speaker, said in voices:
                turns.append(Turn(speaker=speaker.strip(), start_s=start, text=_TAG.sub("", said)))
            continue
        plain = " ".join(_TAG.sub("", line).strip() for line in body.splitlines())
        parsed = _parse_lines([plain])
        if parsed and parsed[0].text:
            turns.append(parsed[0].model_copy(update={"start_s": start}))
        elif turns:  # cue without a speaker: same speaker as the previous cue
            turns.append(turns[-1].model_copy(update={"start_s": start, "text": plain}))
    return meta, turns


def _participants(raw: str | list[str] | None, turns: list[Turn]) -> list[Participant]:
    entries = raw if isinstance(raw, list) else [p for p in (raw or "").split(";") if p.strip()]
    participants = []
    for entry in entries:  # "Name | Role | Organisation", role and organisation optional
        name, role, organisation = (
            [part.strip() or None for part in entry.split("|")] + [None] * 2
        )[:3]
        if name:
            participants.append(Participant(name=name, role=role, organisation=organisation))
    known = {p.name for p in participants}
    for turn in turns:  # anyone who speaks is a participant
        if turn.speaker not in known:
            participants.append(Participant(name=turn.speaker))
            known.add(turn.speaker)
    return participants


def _meeting_type(title: str) -> str:
    lowered = title.lower()
    return next(
        (kind for pattern, kind in _MEETING_TYPES if re.search(pattern, lowered)), "meeting"
    )


def _language(turns: list[Turn]) -> str:
    words = re.findall(r"[a-zà-ÿ]+", " ".join(t.text for t in turns[:40]).lower())
    french = sum(w in _FRENCH_HINTS for w in words)
    english = sum(w in _ENGLISH_HINTS for w in words)
    return "fr" if french > english else "en"


def redact_pii(text: str) -> str:
    """Mask email addresses and phone numbers (9 to 15 digits)."""

    def phone(match: re.Match[str]) -> str:
        digits = sum(char.isdigit() for char in match.group())
        return "[phone]" if 9 <= digits <= 15 else match.group()

    return _PHONE.sub(phone, _EMAIL.sub("[email]", text))


def clean_turns(turns: Iterable[Turn], *, redact: bool = True) -> list[Turn]:
    """Remove fillers and stutters, optionally mask emails and phone numbers, and merge
    consecutive turns of the same speaker."""
    cleaned: list[Turn] = []
    for turn in turns:
        text = _FILLERS.sub("", turn.text)
        text = _REPEATS.sub(r"\1", text)
        if redact:
            text = redact_pii(text)
        text = re.sub(r"\s+", " ", text).strip(" ,")
        if not text:
            continue
        if cleaned and cleaned[-1].speaker == turn.speaker:
            cleaned[-1] = cleaned[-1].model_copy(update={"text": f"{cleaned[-1].text} {text}"})
        else:
            cleaned.append(turn.model_copy(update={"text": text}))
    return cleaned


def _slug(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_value.lower()).strip("-")


def parse_transcript(path: Path, *, redact: bool = True) -> Transcript:
    raw = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()
    if suffix in {".vtt", ".srt"}:
        meta, turns = _parse_cues(raw.removeprefix("WEBVTT"))
    else:
        meta, body = _front_matter(raw)
        declared = {p.name for p in _participants(meta.get("participants"), [])}
        turns = _parse_lines(body.splitlines(), declared or None)
    if not turns:
        raise ValueError(f"no speaker turns found in {path.name}")

    file_match = _FILE_DATE.match(path.stem)
    raw_date = str(meta.get("date") or (file_match.group(1) if file_match else ""))
    if not raw_date:
        raise ValueError(f"no meeting date in {path.name}: add 'date:' or name it YYYY-MM-DD_...")
    slug_title = (
        (file_match.group(2) if file_match else path.stem).replace("-", " ").replace("_", " ")
    )
    title = str(meta.get("title") or slug_title.strip().capitalize())
    turns = clean_turns(turns, redact=redact)
    return Transcript(
        meeting_id=_slug(path.stem),
        title=title,
        date=date.fromisoformat(raw_date),
        meeting_type=str(meta.get("type") or _meeting_type(title)),
        language=str(meta.get("language") or _language(turns)),
        participants=_participants(meta.get("participants"), turns),
        turns=turns,
        path=path.name,
    )


def load_transcripts(folder: Path, *, redact: bool = True) -> list[Transcript]:
    if not folder.is_dir():
        return []
    transcripts = [
        parse_transcript(path, redact=redact)
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in TRANSCRIPT_SUFFIXES
    ]
    return sorted(transcripts, key=lambda t: (t.date, t.meeting_id))


# --- Chunking ---------------------------------------------------------------------------------


class TranscriptChunker:
    """Groups consecutive speaker turns into chunks of about ``max_chars``.

    A turn is never split from its speaker; consecutive chunks share their boundary turn so
    that a question and its answer stay retrievable together. Each chunk records the speakers
    it contains and its time range, which the retrievers use as filters.
    """

    def __init__(self, max_chars: int = 1100, overlap_turns: int = 1) -> None:
        self.max_chars = max_chars
        self.overlap_turns = overlap_turns

    def _pieces(self, transcript: Transcript) -> list[tuple[Turn, str]]:
        pieces: list[tuple[Turn, str]] = []
        for turn in transcript.turns:
            line = format_turn(turn, transcript)
            if len(line) <= self.max_chars:
                pieces.append((turn, line))
                continue
            sentences = re.split(r"(?<=[.!?])\s+", turn.text)
            buffer = ""
            for sentence in sentences:  # very long monologue: split on sentences
                if buffer and len(buffer) + len(sentence) > self.max_chars * 0.8:
                    part = turn.model_copy(update={"text": buffer.strip()})
                    pieces.append((part, format_turn(part, transcript)))
                    buffer = ""
                buffer += f" {sentence}"
            if buffer.strip():
                part = turn.model_copy(update={"text": buffer.strip()})
                pieces.append((part, format_turn(part, transcript)))
        return pieces

    def split(self, transcript: Transcript) -> list[Chunk]:
        pieces = self._pieces(transcript)
        chunks: list[Chunk] = []
        start = 0
        while start < len(pieces):
            end, size = start, 0
            while end < len(pieces) and (
                end == start or size + len(pieces[end][1]) <= self.max_chars
            ):
                size += len(pieces[end][1]) + 1
                end += 1
            window = pieces[start:end]
            turns = [turn for turn, _ in window]
            begin, finish = turns[0].start_s, turns[-1].start_s
            section = (
                f"{format_timestamp(begin)}-{format_timestamp(finish)}" if begin is not None else ""
            )
            chunks.append(
                Chunk(
                    chunk_id=f"{transcript.doc_id}#{len(chunks):03d}",
                    doc_id=transcript.doc_id,
                    source=Source.MEETINGS,
                    title=transcript.display_title,
                    section=section,
                    text="\n".join(line for _, line in window),
                    metadata={
                        "date": transcript.date.isoformat(),
                        "meeting_type": transcript.meeting_type,
                        "language": transcript.language,
                        "participants": [p.name for p in transcript.participants],
                        "speakers": sorted({t.speaker for t in turns}),
                        "start_s": begin,
                        "end_s": finish,
                    },
                )
            )
            if end >= len(pieces):
                break
            start = max(end - self.overlap_turns, start + 1)
        return chunks

    def split_all(self, transcripts: Iterable[Transcript]) -> list[Chunk]:
        return [chunk for transcript in transcripts for chunk in self.split(transcript)]
