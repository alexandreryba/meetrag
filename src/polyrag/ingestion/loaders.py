"""Load documents from a corpus folder organised as ``<corpus>/<source>/<file>``."""

from __future__ import annotations

import html
import logging
import re
from typing import TYPE_CHECKING

from polyrag.domain.models import DOCUMENT_SOURCES, Document, Source

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}
_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_TAG = re.compile(r"<[^>]+>")
_BLOCK_TAG = re.compile(r"</?(p|div|br|li|tr|h[1-6]|section|article)[^>]*>", re.IGNORECASE)
_SCRIPT = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)


def _read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    raw = path.read_text(encoding="utf-8")
    if suffix in {".html", ".htm"}:
        raw = _SCRIPT.sub("", raw)
        raw = _BLOCK_TAG.sub("\n", raw)
        raw = html.unescape(_TAG.sub("", raw))
    return raw


def _title(text: str, path: Path) -> str:
    match = _H1.search(text)
    return match.group(1) if match else path.stem.replace("-", " ").replace("_", " ").title()


def load_corpus(corpus_dir: Path) -> Iterator[Document]:
    """Yield every supported document; the sub-folder name decides the ``Source``."""
    for source in DOCUMENT_SOURCES:
        folder = corpus_dir / source.value
        if not folder.is_dir():
            logger.warning("no folder for source %s in %s", source.value, corpus_dir)
            continue
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() not in SUPPORTED_SUFFIXES or not path.is_file():
                continue
            text = _read_text(path).strip()
            if not text:
                logger.warning("skipping empty document %s", path)
                continue
            yield Document(
                doc_id=f"{source.value}/{path.relative_to(folder).with_suffix('').as_posix()}",
                source=Source(source),
                title=_title(text, path),
                text=text,
                path=path.relative_to(corpus_dir).as_posix(),
            )
