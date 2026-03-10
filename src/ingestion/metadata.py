"""Metadata extraction — joins Parquet metadata from the S3 dataset with loaded documents."""

from pathlib import Path

import pyarrow.parquet as pq

from src.ingestion.loader import Document


def load_metadata(parquet_path: Path) -> dict[str, dict]:
    """Load metadata from a Parquet file and return a dict keyed by filename stem.

    The Parquet file has columns: title, petitioner, respondent, judge,
    author_judge, citation, case_id, cnr, decision_date, disposal_nature, etc.

    The 'path' column contains values like "2020_4_552_564" which map to
    PDF filenames like "2020_4_552_564_EN.pdf" (stem: "2020_4_552_564_EN").
    """
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    metadata_by_id = {}
    for _, row in df.iterrows():
        # The 'path' column has the base name without _EN suffix
        base_path = row.get("path", "")
        doc_id = f"{base_path}_EN"  # Our PDFs are english, named with _EN suffix

        metadata_by_id[doc_id] = {
            "title": row.get("title", ""),
            "petitioner": row.get("petitioner", ""),
            "respondent": row.get("respondent", ""),
            "judge": row.get("judge", ""),
            "author_judge": row.get("author_judge", ""),
            "citation": row.get("citation", ""),
            "case_id": row.get("case_id", ""),
            "decision_date": row.get("decision_date", ""),
            "disposal_nature": row.get("disposal_nature", ""),
        }

    return metadata_by_id


def enrich_documents(
    documents: list[Document], parquet_path: Path
) -> list[Document]:
    """Merge Parquet metadata into Document.metadata for each document.

    The pymupdf metadata already in doc.metadata is kept; Parquet fields
    are added under a 'corpus' key to avoid collisions.
    """
    metadata_by_id = load_metadata(parquet_path)

    for doc in documents:
        corpus_meta = metadata_by_id.get(doc.doc_id, {})
        doc.metadata["corpus"] = corpus_meta

    return documents
