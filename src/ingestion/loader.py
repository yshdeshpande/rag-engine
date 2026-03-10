"""Document loaders for PDF, HTML, and plain text formats."""

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf


@dataclass
class Document:
    """A loaded document with extracted text and metadata."""

    doc_id: str
    text: str
    path: Path
    metadata: dict = field(default_factory=dict)


def load_pdf(path: Path) -> Document | None:
    """Extract text from a single PDF and return a Document.

    Returns None if the PDF is corrupt or unreadable.
    """
    try:
        with pymupdf.open(path) as doc:
            text = "\n".join(page.get_text() for page in doc)
            metadata = doc.metadata or {}

        return Document(
            doc_id=path.stem,
            text=text,
            path=path,
            metadata=metadata,
        )
    except Exception as e:
        print(f"Skipping corrupt file {path.name}: {e}")
        return None


def load_directory(directory: Path) -> list[Document]:
    """Load all PDFs from a directory, skipping any that fail."""
    documents = []
    for pdf_file in sorted(directory.glob("*.pdf")):
        doc = load_pdf(pdf_file)
        if doc:
            documents.append(doc)
    return documents
