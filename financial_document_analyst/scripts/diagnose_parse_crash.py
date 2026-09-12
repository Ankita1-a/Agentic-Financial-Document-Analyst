"""
scripts/diagnose_parse_crash.py

Calls pymupdf4llm.to_markdown directly (not through ingest/parse.py) with
show_progress=True, so a live progress indicator prints as each page is
processed. If the process crashes or hangs, whatever page number was last
shown is the culprit — that's the evidence we need before deciding how to
handle it (skip that page, disable OCR for it, etc.), rather than guessing.

Usage:
    python scripts/diagnose_parse_crash.py path/to/LT_FY2026.pdf
"""

import sys

import pymupdf4llm

if len(sys.argv) != 2:
    print("Usage: python scripts/diagnose_parse_crash.py path/to/report.pdf")
    sys.exit(1)

pdf_path = sys.argv[1]

print(f"Parsing {pdf_path} with live progress — watch the last page number shown if this stops...")
result = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, show_progress=True)
print(f"\nCompleted without crashing: {len(result)} pages processed.")