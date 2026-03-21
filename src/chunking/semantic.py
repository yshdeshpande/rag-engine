"""Semantic chunking — uses embedding similarity to find natural breakpoints."""

import numpy as np
import nltk
from sentence_transformers import SentenceTransformer

from src.chunking.base import BaseChunker, Chunk

nltk.download("punkt_tab", quiet=True)


class SemanticChunker(BaseChunker):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        max_chunk_chars: int = 1000,
        min_chunk_chars: int = 100,
    ):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars

    def _split_sentences_with_offsets(self, text: str):
        """Split text into sentences with character offsets using NLTK.

        Returns:
            List of tuples: (sentence, start_char, end_char)
        """
        sentences = []
        offset = 0
        for sent in nltk.sent_tokenize(text):
            start = text.find(sent, offset)
            if start == -1:
                start = offset
            end = start + len(sent)
            sentences.append((sent, start, end))
            offset = end
        return sentences

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: dict | None = None
    ) -> list[Chunk]:

        metadata = metadata or {}
        sentences = self._split_sentences_with_offsets(text)

        if not sentences:
            return []

        sentence_texts = [s[0] for s in sentences]
        embeddings = self.model.encode(sentence_texts)

        chunks = []
        current_sentences = [sentences[0]]
        current_length = len(sentences[0][0])

        chunk_index = 0

        for i in range(1, len(sentences)):
            prev_emb = embeddings[i - 1]
            curr_emb = embeddings[i]

            sim = self._cosine_similarity(prev_emb, curr_emb)

            sentence, start_char, end_char = sentences[i]

            should_split = (
                sim < self.similarity_threshold
                or current_length + len(sentence) > self.max_chunk_chars
            )

            if should_split and current_length >= self.min_chunk_chars:
                # finalize current chunk
                chunk_text = " ".join([s[0] for s in current_sentences])
                chunk_start = current_sentences[0][1]
                chunk_end = current_sentences[-1][2]

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        start_char=chunk_start,
                        end_char=chunk_end,
                        metadata=metadata.copy(),
                    )
                )

                chunk_index += 1
                current_sentences = [sentences[i]]
                current_length = len(sentence)

            else:
                current_sentences.append(sentences[i])
                current_length += len(sentence)

        # Final chunk
        if current_sentences:
            chunk_text = " ".join([s[0] for s in current_sentences])
            chunk_start = current_sentences[0][1]
            chunk_end = current_sentences[-1][2]

            chunks.append(
                Chunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    start_char=chunk_start,
                    end_char=chunk_end,
                    metadata=metadata.copy(),
                )
            )

        return chunks