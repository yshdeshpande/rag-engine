"""Cross-encoder re-ranking stage.

After the hybrid retriever returns its top candidates, a cross-encoder
scores each (query, chunk) pair jointly — attending to both sides at once
rather than comparing pre-computed embeddings. This is more accurate but
slower, so we only re-rank the top-N candidates from the first stage.

Uses sentence-transformers CrossEncoder under the hood, which supports
models like BAAI/bge-reranker-v2-m3.
"""

from sentence_transformers import CrossEncoder

from src.chunking.base import Chunk


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Re-score candidates with the cross-encoder and return top-k.

        Args:
            query: The user's question.
            candidates: (chunk, retrieval_score) pairs from the hybrid retriever.
            top_k: How many to keep after re-ranking.

        Returns:
            Top-k (chunk, cross_encoder_score) pairs, sorted by relevance.
        """
        if not candidates:
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, chunk.text) for chunk, _score in candidates]

        # Cross-encoder returns a relevance score per pair
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Zip scores back with chunks, sort descending
        scored = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [(chunk, float(ce_score)) for (chunk, _), ce_score in scored[:top_k]]
