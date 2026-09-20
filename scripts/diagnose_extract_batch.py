"""
scripts/diagnose_extract_batch.py

Runs extraction on just a specific page range of a report, in isolation,
with full diagnostic output if _extract_batch_with_splitting fails to
catch an error it should have caught. Caches the parsed pages to a JSON
file next to the PDF on first run, so repeated debugging doesn't cost
another multi-minute parse each time.

Usage:
    python scripts/diagnose_extract_batch.py <pdf_path> <start_page> <end_page>

Example, for the batch that's actually failing:
    python scripts/diagnose_extract_batch.py data/raw_pdfs/LT/2026/LT_FY2026.pdf 741 754
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.parse import ParsedPage, parse_pdf_to_json
from ingest.extract import _extract_batch_with_splitting, _is_too_many_tokens_error
from mistralai.client import Mistral
from mistralai.client.errors import SDKError


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: python scripts/diagnose_extract_batch.py <pdf_path> <start_page> <end_page>")
        sys.exit(1)

    pdf_path, start_page, end_page = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    cache_path = Path(pdf_path).with_suffix(".parsed_cache.json")

    if cache_path.exists():
        print(f"Loading cached parse from {cache_path} (delete this file to force a re-parse)...")
        with open(cache_path, encoding="utf-8") as f:
            raw = json.load(f)
        pages = [ParsedPage(**p) for p in raw]
    else:
        print(f"No cache found — parsing {pdf_path} (this will take a few minutes; cached afterward)...")
        pages = parse_pdf_to_json(pdf_path, cache_path)

    batch = [p for p in pages if start_page <= p.page_number <= end_page]
    if not batch:
        print(f"No pages found in range {start_page}-{end_page} (parsed {len(pages)} pages total)")
        sys.exit(1)

    total_chars = sum(len(p.text) for p in batch)
    print(f"Testing extraction on pages {batch[0].page_number}-{batch[-1].page_number} "
          f"({len(batch)} pages, {total_chars} total chars)")

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    try:
        result = _extract_batch_with_splitting(client, batch, reasoning_effort=None)
        print(f"\nSUCCESS: {len(result.financial_facts)} facts, {len(result.page_sections)} page sections")
    except SDKError as e:
        print("\n=== _extract_batch_with_splitting did NOT catch this error — diagnostic info ===")
        print("type(e):", type(e))
        print("type(e).__module__:", type(e).__module__)
        print("isinstance of the SDKError extract.py imports and checks against:", isinstance(e, SDKError))
        print("status code:", e.raw_response.status_code if e.raw_response is not None else None)
        print("e.body:", repr(e.body))
        print("_is_too_many_tokens_error(e) returns:", _is_too_many_tokens_error(e))
    except Exception as e:
        print(f"\n=== A DIFFERENT exception type entirely: {type(e).__module__}.{type(e).__name__} ===")
        print(e)


if __name__ == "__main__":
    main()