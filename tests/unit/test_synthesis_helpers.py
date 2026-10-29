from meetrag.agents.prompts import ABSTAIN_MESSAGE
from meetrag.agents.synthesis import extract_citations, is_abstention, unique_evidence
from meetrag.domain.models import Evidence, Source


def _evidence(ref: str) -> Evidence:
    return Evidence(
        kind="document", source=Source.SUPPORT, title=ref, content=f"text {ref}", reference=ref
    )


def test_unique_evidence_keeps_first_occurrence_order() -> None:
    items = [_evidence("a"), _evidence("b"), _evidence("a")]
    assert [e.reference for e in unique_evidence(items)] == ["a", "b"]


def test_extract_citations_ignores_out_of_range_markers() -> None:
    evidence = [_evidence("a"), _evidence("b")]
    citations = extract_citations("Yes [2], and also [1][2]. Not [7].", evidence)
    assert [c.index for c in citations] == [1, 2]
    assert citations[1].reference == "b"


def test_abstention_detection() -> None:
    assert is_abstention(ABSTAIN_MESSAGE)
    assert is_abstention(f'"{ABSTAIN_MESSAGE}"')
    assert not is_abstention("The hotel cap is EUR 180 [1].")
