"""Download Indian Supreme Court judgments from AWS Open Data S3 bucket.

Source: https://registry.opendata.aws/indian-supreme-court-judgments/
License: CC-BY-4.0

Usage:
    # Download 2020-2024 (default — ~3,900 PDFs, ~1.6 GB)
    python scripts/download_data.py

    # Download specific years
    python scripts/download_data.py --years 2022 2023

    # Download metadata only (fast, ~3 MB per year)
    python scripts/download_data.py --metadata-only
"""

import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path

S3_BUCKET = "s3://indian-supreme-court-judgments"
DEFAULT_YEARS = list(range(2020, 2025))  # 2020-2024 inclusive

DATA_DIR = Path("data/raw")
METADATA_DIR = Path("data/raw/metadata")


def run_s3_cmd(args: list[str]) -> subprocess.CompletedProcess:
    """Run an AWS S3 CLI command with --no-sign-request."""
    cmd = ["aws", "s3", *args, "--no-sign-request"]
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def download_metadata(year: int) -> Path:
    """Download Parquet metadata for a given year."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    s3_path = f"{S3_BUCKET}/metadata/parquet/year={year}/metadata.parquet"
    local_path = METADATA_DIR / f"{year}.parquet"

    if local_path.exists():
        print(f"  [skip] Metadata for {year} already exists")
        return local_path

    print(f"  Downloading metadata for {year}...")
    run_s3_cmd(["cp", s3_path, str(local_path)])
    return local_path


def download_pdfs(year: int) -> Path:
    """Download and extract English PDF tar for a given year."""
    year_dir = DATA_DIR / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    tar_path = year_dir / "english.tar"
    s3_path = f"{S3_BUCKET}/data/tar/year={year}/english/english.tar"

    # Check if already extracted
    existing_pdfs = list(year_dir.glob("*.pdf"))
    if existing_pdfs:
        print(f"  [skip] {year}: {len(existing_pdfs)} PDFs already extracted")
        return year_dir

    # Download tar
    if not tar_path.exists():
        # Get file size first
        index_path = f"{S3_BUCKET}/data/tar/year={year}/english/english.index.json"
        result = run_s3_cmd(["cp", index_path, "-"])
        index = json.loads(result.stdout)
        size_mb = index["total_size"] / (1024 * 1024)
        file_count = index["file_count"]
        print(f"  Downloading {year}: {file_count} PDFs ({size_mb:.0f} MB)...")
        run_s3_cmd(["cp", s3_path, str(tar_path)])
    else:
        print(f"  [skip] Tar for {year} already downloaded, extracting...")

    # Extract
    print(f"  Extracting {year}...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(path=year_dir)

    # Remove tar to save disk space
    tar_path.unlink()

    extracted = list(year_dir.glob("*.pdf"))
    print(f"  Extracted {len(extracted)} PDFs for {year}")
    return year_dir


def main():
    parser = argparse.ArgumentParser(description="Download Indian SC judgments from AWS S3")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=DEFAULT_YEARS,
        help=f"Years to download (default: {DEFAULT_YEARS[0]}-{DEFAULT_YEARS[-1]})",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only metadata (Parquet files), skip PDFs",
    )
    args = parser.parse_args()

    # Check AWS CLI is installed
    try:
        subprocess.run(["aws", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: AWS CLI not found. Install with: brew install awscli")
        sys.exit(1)

    print(f"Downloading SC judgments for years: {args.years}")
    print(f"Target directory: {DATA_DIR.resolve()}\n")

    for year in sorted(args.years):
        print(f"[{year}]")
        download_metadata(year)
        if not args.metadata_only:
            download_pdfs(year)
        print()

    # Summary
    total_pdfs = sum(1 for _ in DATA_DIR.rglob("*.pdf"))
    total_meta = sum(1 for _ in METADATA_DIR.glob("*.parquet"))
    print(f"Done! {total_pdfs} PDFs, {total_meta} metadata files in {DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
