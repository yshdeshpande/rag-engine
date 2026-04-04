"""BM25 sparse retrieval — implemented from scratch.

BM25 (Best Matching 25) scores documents by keyword overlap with the query.
For each query term, it computes:

    score += IDF(term) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))

Where:
    - tf: how many times the term appears in this document
    - IDF: penalizes common terms, rewards rare ones (log-scaled)
    - k1: controls term frequency saturation (default 1.5)
    - b: controls document length normalization (default 0.75)
"""

import math
import re
from collections import Counter, defaultdict

from src.chunking.base import Chunk


class SparseRetriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        self.chunks: list[Chunk] = []
        self.doc_freqs: list[Counter] = []
        self.idf: dict[str, float] = {}
        self.doc_len: list[int] = []
        self.avg_doc_len: float = 0

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.doc_freqs = []
        self.doc_len = []
        term_doc_count: dict[str, int] = defaultdict(int)

        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            self.doc_len.append(len(tokens))

            for term in freq:
                term_doc_count[term] += 1

        self.avg_doc_len = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        N = len(chunks)
        self.idf = {
            term: math.log((N - df + 0.5) / (df + 0.5) + 1)
            for term, df in term_doc_count.items()
        }

    def _score(self, query_tokens: list[str], doc_index: int) -> float:
        score = 0.0
        freq = self.doc_freqs[doc_index]
        doc_length = self.doc_len[doc_index]

        for term in query_tokens:
            if term not in freq:
                continue

            tf = freq[term]
            idf = self.idf.get(term, 0)

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_length / self.avg_doc_len)
            )
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        query_tokens = self._tokenize(query)

        scored = [
            (self._score(query_tokens, i), i)
            for i in range(len(self.chunks))
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        return [(self.chunks[idx], score) for score, idx in scored[:top_k]]
