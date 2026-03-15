"""Recursive/hierarchical chunking with configurable separators."""

from src.chunking.base import BaseChunker, Chunk

# Each separator is tried in order — coarsest to finest
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]


class RecursiveChunker(BaseChunker):

    def __init__(
        self,
        max_chunk_size: int = 512,
        separators: list[str] | None = None,
    ):
        self.max_chunk_size = max_chunk_size
        self.separators = separators or DEFAULT_SEPARATORS

    def chunk(self, text: str, doc_id: str, metadata: dict | None = None) -> list[Chunk]:
        if not text:
            return []

        raw_chunks = self._recursive_split(text, self.separators)

        # Merge small pieces back together so we don't get tiny chunks
        merged = self._merge_small(raw_chunks)

        chunks = []
        offset = 0
        for i, chunk_text in enumerate(merged):
            start = text.find(chunk_text, offset)
            if start == -1:
                start = offset
            end = start + len(chunk_text)
            offset = start

            chunks.append(Chunk(
                text=chunk_text,
                doc_id=doc_id,
                chunk_index=i,
                start_char=start,
                end_char=end,
                metadata=metadata or {},
            ))

        return chunks

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Split text on the first separator. If any piece is still too large,
        recurse with the next separator in the chain."""
        if len(text) <= self.max_chunk_size:
            return [text.strip()] if text.strip() else []

        if not separators:
            # No separators left — hard cut at max_chunk_size
            result = []
            for i in range(0, len(text), self.max_chunk_size):
                piece = text[i:i + self.max_chunk_size].strip()
                if piece:
                    result.append(piece)
            return result

        sep = separators[0]
        remaining_seps = separators[1:]
        pieces = text.split(sep)

        result = []
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            # Re-add the separator for ". " splits so sentences keep their period
            if sep == ". " and not piece.endswith("."):
                piece += "."
            if len(piece) <= self.max_chunk_size:
                result.append(piece)
            else:
                # Piece is still too big — recurse with a finer separator
                result.extend(self._recursive_split(piece, remaining_seps))

        return result

    def _merge_small(self, chunks: list[str]) -> list[str]:
        """Merge consecutive small chunks until they approach max_chunk_size."""
        if not chunks:
            return []

        merged = []
        current = chunks[0]

        for chunk in chunks[1:]:
            combined = current + "\n" + chunk
            if len(combined) <= self.max_chunk_size:
                current = combined
            else:
                merged.append(current)
                current = chunk

        merged.append(current)
        return merged
