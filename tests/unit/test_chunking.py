from polyrag.domain.models import Document, Source
from polyrag.ingestion.chunking import MarkdownChunker

DOC = """# Remote Work Policy

Intro paragraph.

## Eligibility

Employees are eligible after probation.

## Equipment

### Allowances

The allowance is EUR 45 per month.
"""


def _doc(text: str) -> Document:
    return Document(
        doc_id="policies/test", source=Source.POLICIES, title="Remote Work Policy", text=text
    )


def test_sections_follow_heading_hierarchy() -> None:
    chunks = MarkdownChunker().split(_doc(DOC))
    assert [c.section for c in chunks] == ["", "Eligibility", "Equipment > Allowances"]
    assert chunks[2].text == "The allowance is EUR 45 per month."
    assert chunks[0].chunk_id == "policies/test#000"


def test_contextualized_text_prepends_heading_path() -> None:
    chunk = MarkdownChunker().split(_doc(DOC))[2]
    assert chunk.contextualized_text.startswith("Remote Work Policy > Equipment > Allowances")


def test_long_sections_are_packed_under_chunk_size_with_overlap() -> None:
    sentences = " ".join(f"Sentence number {i} explains a rule." for i in range(80))
    chunks = MarkdownChunker(chunk_size=300, overlap=80).split(_doc(f"## Rules\n\n{sentences}"))
    assert len(chunks) > 5
    assert all(len(c.text) <= 300 for c in chunks)
    # the next chunk starts with the last sentences of the previous one
    first_sentences = chunks[0].text.split("\n\n")
    second_sentences = chunks[1].text.split("\n\n")
    assert second_sentences[0] in first_sentences
    assert first_sentences[-1] in second_sentences


def test_tables_are_kept_in_one_piece() -> None:
    table = "\n".join(
        ["| City | Cap |", "|---|---|"] + [f"| City {i} | EUR {i} |" for i in range(30)]
    )
    chunks = MarkdownChunker(chunk_size=200, overlap=50).split(
        _doc(f"## Hotels\n\nIntro.\n\n{table}")
    )
    assert any("| City | Cap |" in c.text and "| City 29 | EUR 29 |" in c.text for c in chunks)
