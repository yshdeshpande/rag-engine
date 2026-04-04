"""Dense embedding retrieval using FAISS.

Embeds all chunks using a sentence-transformer model, stores vectors in a
FAISS index, and retrieves the nearest neighbors for a query embedding.

FAISS (Facebook AI Similarity Search) is an efficient library for similarity
search over dense vectors. We use an Inner Product index on L2-normalized
vectors, which is equivalent to cosine similarity.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunking.base import Chunk


class DenseRetriever:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", batch_size: int = 64):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.index = None  # FAISS index

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Embed all chunks and build a FAISS index."""
        import faiss

        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]

        # Encode in batches — returns (N, dim) float32 array
        self.embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2 normalize so inner product = cosine sim
        )

        # Build FAISS index using inner product (cosine sim on normalized vectors)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """Embed query, search FAISS index, return top-k (chunk, score) pairs."""
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            results.append((self.chunks[idx], float(score)))

        return results
