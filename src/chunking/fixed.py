"""Fixed-size chunking with configurable overlap."""

from src.chunking.base import BaseChunker, Chunk


class FixedChunker(BaseChunker):

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        strip_whitespace: bool = True,
    ):
        if overlap >= chunk_size:
            raise ValueError(
                f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strip_whitespace = strip_whitespace

    def chunk(self, text: str, doc_id: str, metadata: dict | None = None) -> list[Chunk]:
        if not text:
            return []

        chunks = []
        step = self.chunk_size - self.overlap
        chunk_index = 0

        for start in range(0, len(text), step):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            if self.strip_whitespace:
                chunk_text = chunk_text.strip()

            if not chunk_text:
                continue

            chunks.append(Chunk(
                text=chunk_text,
                doc_id=doc_id,
                chunk_index=chunk_index,
                start_char=start,
                end_char=end,
                metadata=metadata or {},
            ))
            chunk_index += 1

            if end >= len(text):
                break

        return chunks
