"""Structure-aware chunking.

Documents are first split along Markdown headings, so a chunk never mixes two sections,
then long sections are packed into ~``chunk_size`` characters on paragraph / sentence
boundaries with a small overlap. Markdown tables are never split, so every row keeps its
header.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from polyrag.domain.models import Chunk

if TYPE_CHECKING:
    from collections.abc import Iterable

    from polyrag.domain.models import Document

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9*(\"'])")


@dataclass(frozen=True)
class _Section:
    path: str
    text: str


class MarkdownChunker:
    def __init__(self, chunk_size: int = 900, overlap: int = 120) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in self._sections(document.text):
            for piece in self._pack(section.text):
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}#{len(chunks):03d}",
                        doc_id=document.doc_id,
                        source=document.source,
                        title=document.title,
                        section=section.path,
                        text=piece,
                    )
                )
        return chunks

    def split_all(self, documents: Iterable[Document]) -> list[Chunk]:
        return [chunk for document in documents for chunk in self.split(document)]

    @staticmethod
    def _sections(text: str) -> list[_Section]:
        """Group lines by heading path, ignoring the H1 (it is the document title)."""
        sections: list[_Section] = []
        stack: list[str] = []
        buffer: list[str] = []

        def flush() -> None:
            body = "\n".join(buffer).strip()
            if body:
                sections.append(_Section(" > ".join(stack), body))
            buffer.clear()

        for line in text.splitlines():
            match = _HEADING.match(line)
            if not match:
                buffer.append(line)
                continue
            flush()
            level, heading = len(match.group(1)), match.group(2).strip()
            if level == 1:
                stack = []
                continue
            del stack[level - 2 :]
            stack.append(heading)
        flush()
        return sections

    def _pack(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        units: list[str] = []
        for block in re.split(r"\n\s*\n", text):
            block = block.strip()
            if not block:
                continue
            if len(block) <= self.chunk_size or block.lstrip().startswith("|"):
                units.append(block)
            else:
                units.extend(s.strip() for s in _SENTENCE_END.split(block) if s.strip())

        pieces: list[str] = []
        current: list[str] = []
        size = 0
        for unit in units:
            if current and size + len(unit) + 2 > self.chunk_size:
                pieces.append("\n\n".join(current))
                current, size = self._overlap_tail(current)
            current.append(unit)
            size += len(unit) + 2
        if current:
            pieces.append("\n\n".join(current))
        return pieces

    def _overlap_tail(self, units: list[str]) -> tuple[list[str], int]:
        """Carry the last units (up to ``overlap`` characters) into the next chunk."""
        tail: list[str] = []
        size = 0
        for unit in reversed(units):
            if size + len(unit) > self.overlap:
                break
            tail.insert(0, unit)
            size += len(unit) + 2
        return tail, size
