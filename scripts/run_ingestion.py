"""Run the full ingestion pipeline: load → clean → enrich → store.

Usage:
    python scripts/run_ingestion.py
    python scripts/run_ingestion.py --year 2020
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.ingestion.loader import load_directory
from src.ingestion.cleaner import clean_batch
from src.ingestion.metadata import enrich_documents

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def save_documents(documents: list, output_dir: Path) -> None:
    """Save each document as a JSON file with text + metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for doc in tqdm(documents, desc="Saving"):
        out = {
            "doc_id": doc.doc_id,
            "text": doc.text,
            "metadata": doc.metadata,
            "source_path": str(doc.path),
        }
        output_path = output_dir / f"{doc.doc_id}.json"
        output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))


def run(year: int | None = None) -> None:
    years = [year] if year else [int(d.name) for d in sorted(RAW_DIR.iterdir()) if d.is_dir() and d.name.isdigit()]

    for yr in years:
        pdf_dir = RAW_DIR / str(yr)
        parquet_path = RAW_DIR / "metadata" / f"{yr}.parquet"
        output_dir = PROCESSED_DIR / str(yr)

        if not pdf_dir.exists():
            print(f"Skipping {yr}: {pdf_dir} not found")
            continue

        print(f"\n[{yr}] Loading PDFs from {pdf_dir}...")
        documents = load_directory(pdf_dir)
        print(f"  Loaded {len(documents)} documents")

        print(f"  Cleaning text...")
        documents = clean_batch(documents)

        if parquet_path.exists():
            print(f"  Enriching with metadata...")
            documents = enrich_documents(documents, parquet_path)
        else:
            print(f"  Warning: no metadata at {parquet_path}, skipping enrichment")

        print(f"  Saving to {output_dir}...")
        save_documents(documents, output_dir)

        # Print stats
        total_chars = sum(len(d.text) for d in documents)
        avg_chars = total_chars // len(documents) if documents else 0
        print(f"  Done: {len(documents)} docs, {total_chars:,} total chars, {avg_chars:,} avg chars/doc")


def main():
    parser = argparse.ArgumentParser(description="Run the ingestion pipeline")
    parser.add_argument("--year", type=int, default=None, help="Process a specific year (default: all)")
    args = parser.parse_args()
    run(year=args.year)


if __name__ == "__main__":
    main()
