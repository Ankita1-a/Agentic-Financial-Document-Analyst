"""
ingest/build_index.py

Ties the whole ingestion pipeline together for one report — parse ->
extract -> critique -> store — and runs it over a batch of reports from a
JSON manifest, so populating the stores is one command instead of hand-
assembling the pipeline every time (as the last few turns of this project
had to do manually).

Idempotent by design: re-running this on a report you've already ingested
corrects the stores in place rather than duplicating data.
  - financials_db.py already upserts on (company, fiscal_year, metric).
  - vectorstore.py's chunk ids are now content-hashed (fixed alongside this
    file, since this is exactly where the old random-id bug would have
    surfaced as duplicate chunks piling up on every re-run). To also catch
    the case where a report's PARSED TEXT itself changed (e.g. after a
    parsing/chunking bugfix, so the same page produces different chunks),
    every report's existing chunks are deleted before its new ones are
    added — see delete_company_year() in vectorstore.py.

Error containment, matching the same principle used in the CBG pipeline
this project's design borrowed from: one bad report (corrupt PDF, an API
failure) does not stop the rest of a batch from ingesting. Each report's
outcome — success, partial (some facts flagged), or complete failure — is
recorded and returned, never silently swallowed.
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from chromadb.api.types import EmbeddingFunction

from ingest.critic import critique_facts
from ingest.extract import extract_report
from ingest.parse import ParsedPage, parse_pdf_to_json
from stores.financials_db import FinancialFact, upsert_facts_bulk
from stores.vectorstore import add_chunks, delete_company_year, get_collection


@dataclass
class ManifestEntry:
    pdf_path: str
    company: str
    fiscal_year: int
    source_report: str | None = None  # defaults to the PDF's filename if not given


@dataclass
class IngestResult:
    company: str
    fiscal_year: int
    source_report: str
    pages_parsed: int = 0
    facts_verified: int = 0
    facts_flagged: int = 0
    chunks_stored: int = 0
    flagged_facts: list[FinancialFact] = field(default_factory=list)
    error: str | None = None  # set if the whole report failed; other fields stay at defaults


def _parse_with_cache(pdf_path: str | Path, show_progress: bool) -> list[ParsedPage]:
    """
    Parses a PDF, or loads it from a cache file next to it if one already
    exists — same convention scripts/diagnose_extract_batch.py uses. A
    large, OCR-heavy report can take a few minutes to parse; there's no
    reason to pay that cost again on every retry while iterating on
    extraction, since parsing itself doesn't change between attempts.
    Delete the cache file (<pdf>.parsed_cache.json) to force a re-parse
    after a real parsing/chunking change.
    """
    cache_path = Path(pdf_path).with_suffix(".parsed_cache.json")
    if cache_path.exists():
        print(f"  Loading cached parse from {cache_path} (delete this file to force a re-parse)...")
        with open(cache_path, encoding="utf-8") as f:
            raw = json.load(f)
        return [ParsedPage(**p) for p in raw]

    print(f"  Parsing (large or scanned reports can take a few minutes — "
          f"watch the progress bar below, don't stop this early)...")
    return parse_pdf_to_json(pdf_path, cache_path, show_progress=show_progress)


def ingest_report(
    pdf_path: str | Path,
    company: str,
    fiscal_year: int,
    financials_db_path: str | Path,
    vectorstore_dir: str | Path,
    source_report: str | None = None,
    mistral_api_key: str | None = None,
    embedding_function: EmbeddingFunction | None = None,
    show_parse_progress: bool = True,
) -> IngestResult:
    """
    Runs the full pipeline for one report and writes the result into both
    stores. Facts that fail the critic's grounding check are NOT written to
    financials_db.py — they're returned in flagged_facts for review, since
    a wrong number in the store would silently poison every future trend
    query for that company/metric.

    show_parse_progress defaults to True so a large/OCR-heavy report shows
    live per-page progress instead of going silent for a few minutes — that
    silence is easy to mistake for a hang (it has been, in practice) and
    stop prematurely, when the parser was actually working the whole time.
    """
    source_report = source_report or Path(pdf_path).name

    pages = _parse_with_cache(pdf_path, show_progress=show_parse_progress)
    facts, chunks = extract_report(pages, company, fiscal_year, source_report=source_report,
                                    api_key=mistral_api_key)
    critique = critique_facts(facts, pages)

    upsert_facts_bulk(financials_db_path, critique.verified_facts)

    collection = get_collection(vectorstore_dir, embedding_function=embedding_function)
    delete_company_year(collection, company, fiscal_year)
    add_chunks(collection, chunks)

    return IngestResult(
        company=company,
        fiscal_year=fiscal_year,
        source_report=source_report,
        pages_parsed=len(pages),
        facts_verified=len(critique.verified_facts),
        facts_flagged=len(critique.flagged_facts),
        chunks_stored=len(chunks),
        flagged_facts=critique.flagged_facts,
    )


def ingest_folder(
    manifest: list[ManifestEntry],
    financials_db_path: str | Path,
    vectorstore_dir: str | Path,
    mistral_api_key: str | None = None,
    embedding_function: EmbeddingFunction | None = None,
) -> list[IngestResult]:
    """
    Runs ingest_report for every entry in the manifest. A failure on one
    entry (bad PDF, API error, anything) is caught and recorded in that
    entry's IngestResult.error — it does not stop the remaining entries
    from being processed.
    """
    results: list[IngestResult] = []

    for entry in manifest:
        print(f"Ingesting {entry.company} FY{entry.fiscal_year} ({entry.pdf_path})...")
        try:
            result = ingest_report(
                entry.pdf_path, entry.company, entry.fiscal_year,
                financials_db_path, vectorstore_dir,
                source_report=entry.source_report,
                mistral_api_key=mistral_api_key,
                embedding_function=embedding_function,
            )
            print(f"  -> {result.facts_verified} facts verified, {result.facts_flagged} flagged, "
                  f"{result.chunks_stored} chunks stored")
        except Exception as e:
            print(f"  -> FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            result = IngestResult(
                company=entry.company, fiscal_year=entry.fiscal_year,
                source_report=entry.source_report or Path(entry.pdf_path).name,
                error=f"{type(e).__name__}: {e}",
            )
        results.append(result)

    return results


def _print_summary(results: list[IngestResult]) -> None:
    print("\n=== Ingestion summary ===")
    total_verified = total_flagged = total_chunks = 0
    for r in results:
        if r.error:
            print(f"  {r.company} FY{r.fiscal_year}: FAILED — {r.error}")
            continue
        print(f"  {r.company} FY{r.fiscal_year}: {r.facts_verified} verified, "
              f"{r.facts_flagged} flagged, {r.chunks_stored} chunks")
        total_verified += r.facts_verified
        total_flagged += r.facts_flagged
        total_chunks += r.chunks_stored
    failed = sum(1 for r in results if r.error)
    print(f"\nTotals: {total_verified} facts verified, {total_flagged} flagged, "
          f"{total_chunks} chunks stored, {failed}/{len(results)} reports failed")

    all_flagged = [f for r in results for f in r.flagged_facts]
    if all_flagged:
        print(f"\n{len(all_flagged)} flagged facts worth a manual glance:")
        for f in all_flagged[:20]:
            print(f"  {f.company} FY{f.fiscal_year}: {f.metric} = {f.value} {f.unit} (page {f.source_page})")
        if len(all_flagged) > 20:
            print(f"  ... and {len(all_flagged) - 20} more")


def _load_manifest(manifest_path: str | Path) -> list[ManifestEntry]:
    with open(manifest_path, encoding="utf-8") as f:
        raw = json.load(f)
    return [ManifestEntry(**entry) for entry in raw]


def _main() -> None:
    if len(sys.argv) != 4:
        print("Usage: python -m ingest.build_index <manifest.json> <financials_db_path> <vectorstore_dir>")
        print('manifest.json format: [{"pdf_path": "...", "company": "...", "fiscal_year": 2025}, ...]')
        sys.exit(1)

    manifest_path, financials_db_path, vectorstore_dir = sys.argv[1:4]
    manifest = _load_manifest(manifest_path)
    results = ingest_folder(manifest, financials_db_path, vectorstore_dir)
    _print_summary(results)


def _run_smoke_test() -> None:
    """
    Verifies the orchestration logic itself — wiring, idempotent re-runs,
    and per-report error containment — using real critique_facts/
    financials_db/vectorstore code, with parse_pdf and extract_report
    mocked out since those already have their own tested logic elsewhere
    and this file's job is only to wire everything together correctly.
    """
    import tempfile
    from unittest.mock import patch

    from ingest.parse import ParsedPage
    from stores.financials_db import get_financial_series
    from stores.vectorstore import _FakeEmbeddingFunction, search

    fake_page_text = "Consolidated revenue for the year was Rs 2,32,140 crores."
    fake_pages = [ParsedPage(page_number=18, text=fake_page_text)]

    def fake_extract_report(pages, company, fiscal_year, source_report=None, api_key=None):
        facts = [
            FinancialFact(company, fiscal_year, "revenue", 232140.0, "INR crore", source_page=18),
            # deliberately wrong value -> critique_facts should flag, not store, this one
            FinancialFact(company, fiscal_year, "net_profit", 999999.0, "INR crore", source_page=18),
        ]
        from stores.vectorstore import TextChunk
        chunks = [TextChunk(company, fiscal_year, page=18, section="md&a", text=fake_page_text)]
        return facts, chunks

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "financials.sqlite"
        chroma_dir = Path(tmp) / "chroma_db"
        embedding_fn = _FakeEmbeddingFunction()

        with patch(f"{__name__}._parse_with_cache", return_value=fake_pages), \
             patch(f"{__name__}.extract_report", side_effect=fake_extract_report):

            result = ingest_report(
                "fake/tatasteel.pdf", "Tata Steel", 2025, db_path, chroma_dir,
                embedding_function=embedding_fn,
            )

            assert result.facts_verified == 1, f"expected 1 verified fact, got {result.facts_verified}"
            assert result.facts_flagged == 1, f"expected 1 flagged fact, got {result.facts_flagged}"
            assert result.flagged_facts[0].metric == "net_profit"
            assert result.chunks_stored == 1
            print(f"ingest_report: {result.facts_verified} verified, {result.facts_flagged} flagged, "
                  f"{result.chunks_stored} chunks — matches the mocked extraction exactly")

            series = get_financial_series(db_path, "Tata Steel", "revenue")
            assert series == [(2025, 232140.0)], f"verified fact wasn't actually written to financials_db: {series}"

            collection = get_collection(chroma_dir, embedding_function=embedding_fn)
            hits = search(collection, "revenue", company="Tata Steel")
            assert len(hits) == 1, f"chunk wasn't actually written to the vector store: {hits}"
            print("ingest_report: verified fact and chunk both actually landed in the real stores")

            # --- re-running the same report must not duplicate anything ---
            result2 = ingest_report(
                "fake/tatasteel.pdf", "Tata Steel", 2025, db_path, chroma_dir,
                embedding_function=embedding_fn,
            )
            series_after_rerun = get_financial_series(db_path, "Tata Steel", "revenue")
            assert series_after_rerun == [(2025, 232140.0)], "re-running duplicated a financial fact row"
            hits_after_rerun = search(collection, "revenue", company="Tata Steel")
            assert len(hits_after_rerun) == 1, "re-running duplicated a chunk instead of replacing it"
            print("ingest_report: re-running the same report is idempotent — no duplicate facts or chunks")

            # --- batch run: one entry fails, the other must still succeed ---
            def flaky_parse_pdf(path, show_progress=True):
                if "broken" in str(path):
                    raise ValueError("simulated corrupt PDF")
                return fake_pages

            with patch(f"{__name__}._parse_with_cache", side_effect=flaky_parse_pdf):
                manifest = [
                    ManifestEntry("fake/lt.pdf", "L&T", 2025),
                    ManifestEntry("fake/broken.pdf", "Broken Co", 2025),
                ]
                batch_results = ingest_folder(manifest, db_path, chroma_dir, embedding_function=embedding_fn)

            assert len(batch_results) == 2, "one failed entry should not stop the rest of the batch"
            lt_result, broken_result = batch_results
            assert lt_result.error is None and lt_result.facts_verified == 1
            assert broken_result.error is not None and "simulated corrupt PDF" in broken_result.error
            print("ingest_folder: a failed report is recorded with its error; the rest of the batch still runs")

    print("\nAll build_index.py smoke tests passed.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _main()
    else:
        _run_smoke_test()