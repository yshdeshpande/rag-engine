"""Tests for chunking strategies."""

import pytest

from src.chunking.fixed import FixedChunker
from src.chunking.recursive import RecursiveChunker

# ---------------------------------------------------------------------------
# FixedChunker
# ---------------------------------------------------------------------------

class TestFixedChunker:
    def test_basic_chunking(self):
        chunker = FixedChunker(chunk_size=10, overlap=0)
        chunks = chunker.chunk("abcdefghij" * 3, doc_id="doc1")
        assert len(chunks) == 3
        for chunk in chunks:
            assert len(chunk.text) == 10

    def test_overlap(self):
        chunker = FixedChunker(chunk_size=10, overlap=5)
        text = "a" * 25
        chunks = chunker.chunk(text, doc_id="doc1")
        # step=5, starts: 0→10, 5→15, 10→20, 15→25 (hits end, breaks) → 4 chunks
        assert len(chunks) == 4

    def test_empty_text(self):
        chunker = FixedChunker(chunk_size=10, overlap=0)
        assert chunker.chunk("", doc_id="doc1") == []

    def test_text_shorter_than_chunk_size(self):
        chunker = FixedChunker(chunk_size=100, overlap=0)
        chunks = chunker.chunk("short text", doc_id="doc1")
        assert len(chunks) == 1
        assert chunks[0].text == "short text"

    def test_chunk_metadata(self):
        chunker = FixedChunker(chunk_size=10, overlap=0)
        chunks = chunker.chunk("abcdefghij", doc_id="test_doc", metadata={"year": 2024})
        assert chunks[0].doc_id == "test_doc"
        assert chunks[0].chunk_index == 0
        assert chunks[0].metadata == {"year": 2024}

    def test_start_end_char_offsets(self):
        chunker = FixedChunker(chunk_size=5, overlap=0)
        chunks = chunker.chunk("abcdefghij", doc_id="doc1")
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == 5
        assert chunks[1].start_char == 5
        assert chunks[1].end_char == 10

    def test_overlap_must_be_less_than_chunk_size(self):
        with pytest.raises(ValueError, match="Overlap"):
            FixedChunker(chunk_size=10, overlap=10)

    def test_sequential_chunk_indices(self):
        chunker = FixedChunker(chunk_size=5, overlap=0)
        chunks = chunker.chunk("a" * 20, doc_id="doc1")
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_whitespace_stripping(self):
        chunker = FixedChunker(chunk_size=10, overlap=0, strip_whitespace=True)
        chunks = chunker.chunk("  hello   world     end", doc_id="doc1")
        for chunk in chunks:
            assert chunk.text == chunk.text.strip()


# ---------------------------------------------------------------------------
# RecursiveChunker
# ---------------------------------------------------------------------------

class TestRecursiveChunker:
    def test_short_text_single_chunk(self):
        chunker = RecursiveChunker(max_chunk_size=100)
        chunks = chunker.chunk("This is short.", doc_id="doc1")
        assert len(chunks) == 1

    def test_splits_on_paragraph_boundary(self):
        chunker = RecursiveChunker(max_chunk_size=30)
        text = "First paragraph.\n\nSecond paragraph."
        chunks = chunker.chunk(text, doc_id="doc1")
        assert len(chunks) >= 2
        assert "First" in chunks[0].text
        assert "Second" in chunks[-1].text

    def test_falls_through_to_sentence_split(self):
        chunker = RecursiveChunker(max_chunk_size=30)
        # No paragraph breaks, but has sentence breaks
        text = "First sentence. Second sentence. Third sentence here."
        chunks = chunker.chunk(text, doc_id="doc1")
        assert len(chunks) >= 2

    def test_empty_text(self):
        chunker = RecursiveChunker(max_chunk_size=100)
        assert chunker.chunk("", doc_id="doc1") == []

    def test_respects_max_size(self):
        chunker = RecursiveChunker(max_chunk_size=50)
        text = "word " * 200
        chunks = chunker.chunk(text, doc_id="doc1")
        for chunk in chunks:
            assert len(chunk.text) <= 50

    def test_merge_small_pieces(self):
        chunker = RecursiveChunker(max_chunk_size=100)
        text = "A.\n\nB.\n\nC.\n\nD."
        chunks = chunker.chunk(text, doc_id="doc1")
        # With max_chunk_size=100, all these tiny pieces should merge
        assert len(chunks) == 1

    def test_chunk_indices_sequential(self):
        chunker = RecursiveChunker(max_chunk_size=30)
        text = "A long paragraph.\n\nAnother paragraph.\n\nThird paragraph."
        chunks = chunker.chunk(text, doc_id="doc1")
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_metadata_propagated(self):
        chunker = RecursiveChunker(max_chunk_size=10)
        chunks = chunker.chunk("Some longer text here.", doc_id="d", metadata={"k": "v"})
        for chunk in chunks:
            assert chunk.metadata == {"k": "v"}
