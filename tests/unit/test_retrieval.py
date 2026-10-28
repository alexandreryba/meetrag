from polyrag.domain.models import Chunk, ScoredChunk, SearchFilter, Source
from polyrag.infrastructure.embeddings import HashingEmbedder
from polyrag.infrastructure.vectorstores import InMemoryVectorStore
from polyrag.retrieval.bm25 import BM25Index, tokenize
from polyrag.retrieval.fusion import reciprocal_rank_fusion
from polyrag.retrieval.hybrid import HybridRetriever, RetrievalMode


def _chunk(chunk_id: str, text: str, source: Source = Source.PRODUCTS) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        source=source,
        title="T",
        section="",
        text=text,
    )


CHUNKS = [
    _chunk("a#0", "Voltaria Fixe 3 ans has a fixed price for 36 months (code FIXE3)."),
    _chunk("b#0", "Voltaria Verte supplies renewable electricity from hydropower."),
    _chunk("c#0", "Hotel caps are EUR 180 in Paris.", Source.POLICIES),
    _chunk("d#0", "Off-peak hours run from 23:00 to 07:00 for electric vehicles."),
]


def test_tokenize_normalizes_accents_plurals_and_stopwords() -> None:
    assert tokenize("The Offers in Île-de-France") == ["offer", "ile", "de", "france"]
    assert "0.2016" in tokenize("price EUR 0.2016 per kWh")


def test_bm25_ranks_exact_code_first_and_filters_by_source() -> None:
    index = BM25Index(CHUNKS)
    assert index.search("FIXE3 contract", limit=1)[0].chunk.chunk_id == "a#0"
    results = index.search("Paris hotel", limit=5, where=SearchFilter(source=Source.PRODUCTS))
    assert all(r.chunk.source is Source.PRODUCTS for r in results)


def test_rrf_rewards_items_ranked_high_by_both_retrievers() -> None:
    a, b, c = (ScoredChunk(chunk=CHUNKS[i], score=1.0) for i in range(3))
    fused = reciprocal_rank_fusion([[a, b, c], [b, a]], k=60)
    assert [s.chunk.chunk_id for s in fused][:2] in (["a#0", "b#0"], ["b#0", "a#0"])
    assert fused[-1].chunk.chunk_id == "c#0"
    assert fused[0].score == fused[1].score


def test_hybrid_retriever_modes() -> None:
    embedder, store = HashingEmbedder(), InMemoryVectorStore()
    store.reset(embedder.dimension)
    store.upsert(CHUNKS, embedder.embed_documents([c.contextualized_text for c in CHUNKS]))
    retriever = HybridRetriever(bm25=BM25Index(CHUNKS), embedder=embedder, store=store)
    assert retriever.default_mode is RetrievalMode.HYBRID
    for mode in (RetrievalMode.BM25, RetrievalMode.DENSE, RetrievalMode.HYBRID):
        hits = retriever.retrieve("renewable hydropower electricity", top_k=2, mode=mode)
        assert hits[0].chunk.chunk_id == "b#0"
    policy_hits = retriever.retrieve("hotel", top_k=3, where=SearchFilter(source=Source.POLICIES))
    assert {h.chunk.source for h in policy_hits} == {Source.POLICIES}
