"""
ingest/parse.py

Converts an annual report PDF into page-level markdown text. This is the
first stage of ingestion and deliberately does no LLM calls — it's pure
text extraction, so it's fast, free, and safe to re-run as often as needed
while iterating on the extraction/critic stages downstream.

Verified against pymupdf4llm's actual output shape (page_chunks=True
returns a list of dicts with 'metadata', 'toc_items', 'page_boxes', 'text'
keys; metadata['page_number'] is 1-indexed).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import pymupdf4llm


@dataclass
class ParsedPage:
    page_number: int  # 1-indexed, matches the physical PDF page
    text: str


def parse_pdf(pdf_path: str | Path, show_progress: bool = True) -> list[ParsedPage]:
    """
    Parse a PDF into a list of per-page markdown text blocks.

    show_progress defaults to True: a large, OCR-heavy report (500+ pages
    with scanned sections) can take a few minutes here, and pymupdf4llm's
    to_markdown gives no other feedback during that time on its own — with
    this off, the terminal goes completely silent for minutes at a stretch,
    which is easy to mistake for a hang and stop. Set it to False for
    non-interactive use (tests, batch jobs writing to a log file) where a
    live progress bar isn't wanted.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    raw_pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, show_progress=show_progress)

    pages: list[ParsedPage] = []
    for raw_page in raw_pages:
        page_number = raw_page["metadata"]["page_number"]
        text = raw_page["text"].strip()
        pages.append(ParsedPage(page_number=page_number, text=text))

    return pages


def parse_pdf_to_json(pdf_path: str | Path, output_path: str | Path, show_progress: bool = True) -> list[ParsedPage]:
    """Parse a PDF and cache the page list as JSON (so we never re-parse a PDF twice)."""
    pages = parse_pdf(pdf_path, show_progress=show_progress)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in pages], f, ensure_ascii=False, indent=2)
    return pages


def _main() -> None:
    parser = argparse.ArgumentParser(description="Parse an annual report PDF into page-level text.")
    parser.add_argument("pdf_path", help="Path to the annual report PDF")
    parser.add_argument("--out", help="Optional path to write JSON output", default=None)
    args = parser.parse_args()

    pages = parse_pdf(args.pdf_path)
    print(f"Parsed {len(pages)} pages from {args.pdf_path}")
    if pages:
        print("--- Page 1 preview (first 300 chars) ---")
        print(pages[0].text[:300])

    if args.out:
        parse_pdf_to_json(args.pdf_path, args.out)
        print(f"Wrote JSON to {args.out}")


if __name__ == "__main__":
    _main()