"""Tests for retrieval components (BM25 sparse retriever)."""

import pytest

from src.chunking.base import Chunk
from src.retrieval.sparse import SparseRetriever


def _make_chunks(texts: list[str], doc_id: str = "doc1") -> list[Chunk]:
    """Helper to create Chunk objects from plain text strings."""
    return [
        Chunk(
            text=text,
            doc_id=doc_id,
            chunk_index=i,
            start_char=0,
            end_char=len(text),
            metadata={},
        )
        for i, text in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# BM25 indexing
# ---------------------------------------------------------------------------

class TestSparseRetrieverIndexing:
    def test_index_stores_chunks(self):
        retriever = SparseRetriever()
        chunks = _make_chunks(["hello world", "foo bar baz"])
        retriever.index(chunks)

        assert len(retriever.chunks) == 2
        assert len(retriever.doc_freqs) == 2
        assert retriever.avg_doc_len > 0

    def test_idf_computed(self):
        retriever = SparseRetriever()
        chunks = _make_chunks(["the cat sat", "the dog ran", "the bird flew"])
        retriever.index(chunks)

        # "the" appears in all 3 docs — should have lower IDF
        # "cat" appears in only 1 — should have higher IDF
        assert retriever.idf["the"] < retriever.idf["cat"]

    def test_tokenize_lowercases(self):
        retriever = SparseRetriever()
        tokens = retriever._tokenize("Hello World FOO")
        assert tokens == ["hello", "world", "foo"]


# ---------------------------------------------------------------------------
# BM25 search
# ---------------------------------------------------------------------------

class TestSparseRetrieverSearch:
    def setup_method(self):
        self.retriever = SparseRetriever()
        self.chunks = _make_chunks([
            "machine learning is a subset of artificial intelligence",
            "python programming language is widely used in data science",
            "deep learning neural networks require large datasets",
            "web development with javascript and html",
            "natural language processing uses machine learning techniques",
        ])
        self.retriever.index(self.chunks)

    def test_returns_ranked_results(self):
        results = self.retriever.search("machine learning", top_k=3)
        assert len(results) == 3
        # Scores should be descending
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_chunks_ranked_higher(self):
        results = self.retriever.search("machine learning", top_k=5)
        top_texts = [chunk.text for chunk, _ in results[:2]]
        # The two ML-related chunks should be in the top 2
        assert any("machine learning" in t for t in top_texts)

    def test_top_k_limits_results(self):
        results = self.retriever.search("learning", top_k=2)
        assert len(results) == 2

    def test_exact_match_scores_highest(self):
        results = self.retriever.search("javascript html web development", top_k=5)
        top_chunk = results[0][0]
        assert "javascript" in top_chunk.text

    def test_no_match_returns_zero_scores(self):
        results = self.retriever.search("quantum entanglement", top_k=3)
        # All scores should be 0 since no terms match
        for _, score in results:
            assert score == 0.0

    def test_search_empty_query(self):
        results = self.retriever.search("", top_k=3)
        assert len(results) == 3
        for _, score in results:
            assert score == 0.0


# ---------------------------------------------------------------------------
# BM25 parameter sensitivity
# ---------------------------------------------------------------------------

class TestBM25Parameters:
    def test_k1_zero_ignores_term_frequency(self):
        """With k1=0, term frequency doesn't matter — only IDF."""
        retriever = SparseRetriever(k1=0.0, b=0.75)
        chunks = _make_chunks([
            "cat cat cat cat cat",  # high tf for "cat"
            "cat dog",              # low tf for "cat"
        ])
        retriever.index(chunks)
        results = retriever.search("cat", top_k=2)

        # With k1=0, the BM25 numerator becomes 1 regardless of tf
        # Both docs contain "cat" so scores should be equal
        assert results[0][1] == pytest.approx(results[1][1], abs=0.01)

    def test_b_zero_ignores_doc_length(self):
        """With b=0, document length normalization is disabled."""
        retriever = SparseRetriever(k1=1.5, b=0.0)
        chunks = _make_chunks([
            "cat " * 100,  # very long doc
            "cat",         # very short doc
        ])
        retriever.index(chunks)
        results = retriever.search("cat", top_k=2)

        # The long doc has much higher tf, so it should score higher with b=0
        long_score = next(s for c, s in results if len(c.text) > 10)
        short_score = next(s for c, s in results if len(c.text) <= 10)
        assert long_score > short_score
