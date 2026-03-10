"""Abstract base class for all chunking strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Chunk:
    """A single chunk of text with metadata about its origin."""

    text: str
    doc_id: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict


class BaseChunker(ABC):
    """All chunking strategies inherit from this."""

    @abstractmethod
    def chunk(self, text: str, doc_id: str, metadata: dict | None = None) -> list[Chunk]:
        """Split text into chunks. Must be implemented by subclasses."""
        ...
