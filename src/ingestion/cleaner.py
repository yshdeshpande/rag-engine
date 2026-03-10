"""Text cleaning and preprocessing — handles OCR artifacts, encoding issues, normalization."""

import re

from src.ingestion.loader import Document


def clean_document(doc: Document) -> Document:
    """Clean extracted PDF text and return a new Document with cleaned text."""
    text = doc.text

    text = remove_margin_letters(text)
    text = remove_page_numbers(text)
    text = remove_header_footer(text)
    text = fix_whitespace(text)

    return Document(
        doc_id=doc.doc_id,
        text=text,
        path=doc.path,
        metadata=doc.metadata,
    )


def remove_margin_letters(text: str) -> str:
    """Remove the A-H column markers that appear on every page.

    SC PDFs have single letters A through H on their own lines,
    used as margin reference markers in the printed reports.
    """
    # Match lines that are just a single uppercase letter (A-H) with optional whitespace
    return re.sub(r"(?m)^\s*[A-H]\s*$", "", text)


def remove_page_numbers(text: str) -> str:
    """Remove standalone page numbers."""
    # Lines that are just a number (possibly with whitespace)
    return re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)


def remove_header_footer(text: str) -> str:
    """Remove recurring headers/footers from SC judgment PDFs.

    Common patterns:
    - "SUPREME COURT REPORTS  [YYYY] N S.C.R."
    - Case title lines repeated at top of pages
    """
    # Remove "SUPREME COURT REPORTS [YYYY] N S.C.R." header
    text = re.sub(
        r"(?m)^\s*SUPREME COURT REPORTS\s*\[\d{4}\]\s*\d{1,2}\s*S\.C\.R\.\s*$",
        "",
        text,
    )
    return text


def fix_whitespace(text: str) -> str:
    """Normalize whitespace: collapse blank lines, fix spacing."""
    # Replace 3+ consecutive newlines with 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace on each line
    text = re.sub(r"(?m)\s+$", "", text)
    # Strip leading/trailing whitespace from the whole document
    text = text.strip()
    return text


def clean_batch(documents: list[Document]) -> list[Document]:
    """Clean a list of documents."""
    return [clean_document(doc) for doc in documents]
