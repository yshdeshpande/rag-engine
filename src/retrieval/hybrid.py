"""Hybrid retrieval combining dense + sparse with Reciprocal Rank Fusion.

RRF combines ranked lists from multiple retrievers without needing to
normalize scores (which is hard — BM25 scores and cosine similarities
are on completely different scales).

For each document, its RRF score is:

    RRF(doc) = sum( 1 / (k + rank_i) )  for each retriever i

Where rank_i is the document's position in retriever i's results (1-indexed),
and k is a smoothing constant (default 60, from the original RRF paper).

A document ranked #1 by both retrievers: 1/61 + 1/61 = 0.0328
A document ranked #1 by dense, #50 by sparse: 1/61 + 1/110 = 0.0255
A document only in dense at #1: 1/61 = 0.0164

This naturally rewards documents that appear in multiple result lists
and are ranked highly in each.
"""

from collections import defaultdict

from src.chunking.base import Chunk
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ):
        self.dense = dense
        self.sparse = sparse
        self.k = k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def search(self, query: str, top_k: int = 10, fetch_k: int = 50) -> list[tuple[Chunk, float]]:
        """Run both retrievers, fuse with RRF, return top-k.

        fetch_k: how many results to pull from each retriever before fusion.
        Should be >= top_k. More means better recall but slower.
        """
        dense_results = self.dense.search(query, top_k=fetch_k)
        sparse_results = self.sparse.search(query, top_k=fetch_k)

        return self.reciprocal_rank_fusion(dense_results, sparse_results, top_k)

    def reciprocal_rank_fusion(
        self,
        dense_results: list[tuple[Chunk, float]],
        sparse_results: list[tuple[Chunk, float]],
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        """Combine two ranked lists using RRF."""

        # Map chunk doc_id+chunk_index to (Chunk, rrf_score)
        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_lookup: dict[str, Chunk] = {}

        # Score dense results
        for rank, (chunk, _score) in enumerate(dense_results, start=1):
            key = f"{chunk.doc_id}:{chunk.chunk_index}"
            rrf_scores[key] += self.dense_weight * (1.0 / (self.k + rank))
            chunk_lookup[key] = chunk

        # Score sparse results
        for rank, (chunk, _score) in enumerate(sparse_results, start=1):
            key = f"{chunk.doc_id}:{chunk.chunk_index}"
            rrf_scores[key] += self.sparse_weight * (1.0 / (self.k + rank))
            chunk_lookup[key] = chunk

        # Sort by RRF score descending
        sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        return [
            (chunk_lookup[key], rrf_scores[key])
            for key in sorted_keys[:top_k]
        ]
