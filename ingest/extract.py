"""
ingest/extract.py

Calls Mistral to turn raw parsed pages into two things:
  1. Structured financial facts (revenue, profit, segment figures, etc.)
     ready for stores/financials_db.py
  2. A section label per page (risk_factors, md&a, segment_overview, ...)
     used to tag chunks before they go into stores/vectorstore.py

Pages are processed in batches, not one call per page — Mistral's free
"Experiment" tier rate-limits are conservative (roughly 1 request/second,
per-model token caps that aren't published exactly), so a 250-page report
processed one page at a time would take a very long time and burn through
budget fast.

Mistral's job here is deliberately narrow: classify pages and pull out
exact numbers with a source page attached. No summarizing or
interpretation happens in this file — narrative reasoning is the agent's
job at query time, not ingestion's.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Literal

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mistralai.client import Mistral
from mistralai.client.errors import SDKError
from pydantic import BaseModel

from ingest.parse import ParsedPage
from stores.financials_db import FinancialFact
from stores.vectorstore import TextChunk, chunk_page_text

# mistral-small-latest and mistral-medium-latest (Premier tier) both return a
# hard 0 requests/minute limit on the free "Experiment" tier — confirmed
# empirically, not documented anywhere. Only Mistral's open-weight models
# (Apache 2.0: Nemo, Ministral) get real free-tier access. open-mistral-nemo
# (12B) is the better of the two available for this task — more capable than
# the 3B Ministral at pulling precise numbers out of dense financial text.
MODEL_NAME = "open-mistral-nemo"
BATCH_SIZE_PAGES = 20
MAX_RETRIES = 5

SectionLabel = Literal["risk_factors", "md&a", "segment_overview", "financial_statements", "other"]

CANONICAL_METRIC_EXAMPLES = [
    "revenue", "net_profit", "ebitda", "total_assets", "total_equity",
    "segment_revenue:<segment name>", "geography_revenue:<region name>",
]


class _ExtractedFact(BaseModel):
    metric: str
    value: float
    unit: str | None = None
    page: int


class _PageSection(BaseModel):
    page: int
    section: SectionLabel


class _BatchExtraction(BaseModel):
    financial_facts: list[_ExtractedFact]
    page_sections: list[_PageSection]


def _build_prompt(pages: list[ParsedPage]) -> str:
    page_blocks = "\n\n".join(f"[Page {p.page_number}]\n{p.text}" for p in pages)
    return f"""You are extracting data from pages of a company's annual report.

For EVERY page in this batch, do two things:

1. Classify its dominant content into exactly one section label:
   - "risk_factors": discusses business/financial/operational risks
   - "md&a": management discussion and analysis, outlook, narrative commentary
   - "segment_overview": describes business segments or divisions
   - "financial_statements": balance sheet, P&L, cash flow, notes to accounts
   - "other": cover pages, table of contents, unrelated content
   Every page must get exactly one label, even if it's "other".

2. Extract every numeric financial fact you can find with high confidence —
   revenue, net profit, EBITDA, total assets, total equity, and segment or
   geography breakdowns. Use consistent metric names across pages, e.g.
   {", ".join(CANONICAL_METRIC_EXAMPLES)}.
   Only extract a figure for the CURRENT reporting year, not prior-year
   comparatives shown alongside it in the same table.
   Skip a number if you are not confident what it represents — a missed
   fact is far better than a wrong one.

Pages:

{page_blocks}
"""


def _get_http_status(e: BaseException) -> int | None:
    """
    Duck-types the HTTP status code out of an exception without relying on
    isinstance(e, SDKError). Real evidence: the exception actually raised
    by client.chat.parse() in production does NOT satisfy
    isinstance(e, SDKError) against the class this file imports from
    mistralai.client.errors, even though it prints identically ("SDKError:
    API error occurred: Status 400. Body: {...}") — confirmed by adding an
    `except SDKError` print that never fired on a real, reproducible
    failure. mistralai actually ships three separate SDKError class
    definitions (plain client, azure, gcp subpackages); rather than chase
    exactly which one is involved or why, duck-typing on the attributes an
    HTTP-backed SDK error would have sidesteps class identity entirely.
    """
    raw_response = getattr(e, "raw_response", None)
    return getattr(raw_response, "status_code", None)


def _get_http_body_text(e: BaseException) -> str | None:
    body = getattr(e, "body", None)
    if body:
        return body
    raw_response = getattr(e, "raw_response", None)
    return getattr(raw_response, "text", None)


def _get_http_headers(e: BaseException) -> dict:
    raw_response = getattr(e, "raw_response", None)
    return getattr(raw_response, "headers", None) or {}


def _is_too_many_tokens_error(e: BaseException) -> bool:
    """
    Detects Mistral's "too many tokens overall, split into more batches"
    error specifically (observed: HTTP 400, code "3210", type
    "invalid_request_prompt"), so only THIS failure is treated as
    retryable/splittable below — any other 400 (a genuinely malformed
    request, an auth problem) should still fail loudly.

    Matched against a single real observed error body — no documentation
    for this error code was found. If Mistral's wording ever changes, this
    needs updating against a fresh real example. Works via _get_http_*
    duck-typing, not isinstance, so it still matches regardless of the
    exact exception class involved.
    """
    if _get_http_status(e) != 400:
        return False
    body_text = _get_http_body_text(e)
    try:
        body = json.loads(body_text) if body_text else {}
    except (json.JSONDecodeError, TypeError):
        return False
    return body.get("code") == "3210" or body.get("type") == "invalid_request_prompt"


def _call_mistral_with_retry(
    client: Mistral, prompt: str, reasoning_effort: str | None = None
) -> _BatchExtraction:
    """
    Retries on rate limits (429), transient server errors (5xx), and
    Mistral's "too many tokens overall, split into more batches" 400 error.

    Catches Exception broadly, not a specific SDK exception class — real
    evidence (see _get_http_status's docstring) shows the exception this
    SDK actually raises doesn't reliably satisfy isinstance checks against
    the class this file imports, so status/body/headers are duck-typed off
    whatever object comes back instead. Anything that doesn't look like an
    HTTP-backed SDK error at all (no status code found — e.g. our own "no
    parseable JSON" ValueError below) is re-raised immediately rather than
    guessed at.

    The "too many tokens" 400 is included here deliberately, not just
    handled as a batch-size problem elsewhere: on two separate real
    reports, this failure hit exactly the LAST batch of a long run of
    otherwise-identical-shaped requests, with wildly different page counts
    (2 pages, then separately 14 pages) and different content — and
    re-running the exact same "too large" batch in complete isolation
    immediately succeeded. That pattern doesn't fit a per-request size
    limit; it fits a cumulative per-minute token budget that a fast run of
    many consecutive requests can exhaust near the end, which Mistral
    appears to report as this 400 rather than the 429 you'd expect for a
    rate condition. So the fix is the same one that already works for
    429s: wait, then retry the SAME request — not shrink it.
    (extract_report still falls back to splitting, in
    _extract_batch_with_splitting, if retrying here is eventually
    exhausted — kept as a second line of defense in case a batch is ever
    genuinely too large on its own, not just rate-limited.)

    reasoning_effort defaults to None and is only included in the request
    when set: open-mistral-nemo is a base model, not one of Mistral's
    reasoning-tier models, so sending a reasoning_effort value it doesn't
    understand risks a 400 rather than being silently ignored. Keep this
    None for Nemo/Ministral; only set it if MODEL_NAME is later switched to
    a model that actually documents support for it.

    Confirmed against a real 429 response body: Mistral's actual rate-limit
    headers are `x-ratelimit-limit-req-minute` and
    `x-ratelimit-remaining-req-minute` (not the `-requests`/`-tokens` split
    used by some other providers, and not the singular `X-RateLimit-Remaining`
    their own docs page mentions either — this is what the API actually sends).

    A limit of "0" means the organization has zero request budget for this
    model on the current plan — that's an account/tier issue, not a burst,
    and no amount of retrying fixes it. This fails fast in that case instead
    of burning through MAX_RETRIES with backoff for something that can only
    be resolved on Mistral's dashboard.
    """
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = dict(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format=_BatchExtraction,
            )
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
            response = client.chat.parse(**kwargs)
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise ValueError("Mistral returned no parseable JSON")
            return parsed
        except BaseException as e:
            if not isinstance(e, Exception):
                raise

            status = _get_http_status(e)
            if status is None:
                # Doesn't look like an HTTP-backed SDK error at all (e.g. the
                # ValueError raised two lines up) — nothing here knows how to
                # retry it usefully, so don't guess; let it propagate as-is.
                raise

            last_error = e
            headers = _get_http_headers(e)
            limit_per_min = headers.get("x-ratelimit-limit-req-minute")
            remaining_per_min = headers.get("x-ratelimit-remaining-req-minute")
            retry_after = headers.get("retry-after")
            too_many_tokens = _is_too_many_tokens_error(e)

            if status == 429 and limit_per_min == "0":
                raise RuntimeError(
                    f"Mistral reports a hard 0 requests/minute limit for model '{MODEL_NAME}' "
                    f"on your account. This is not a timing issue — no retry or backoff can fix it. "
                    f"Check https://admin.mistral.ai/plateforme/limits for your organization's actual "
                    f"per-model limits, and confirm the free tier is fully activated (phone verification, "
                    f"etc.) or try a different model."
                ) from e

            if status == 429 or (500 <= status < 600) or too_many_tokens:
                if retry_after is not None:
                    wait = float(retry_after)
                    wait_source = "server Retry-After"
                elif too_many_tokens:
                    # No rate-limit headers are given on this error at all (checked: they're
                    # absent), so there's no real number to go on — backing off longer than a
                    # typical per-minute window, growing per attempt, is a deliberate guess,
                    # not a documented value. If this guess is wrong, MAX_RETRIES bounds the
                    # total wasted time to a few minutes before falling through to raise.
                    wait = 20 * (attempt + 1)
                    wait_source = "assumed per-minute token budget (no header to confirm the real window)"
                elif status == 429:
                    wait = 60  # this is a per-minute window, not per-second — a short backoff can't outlast it
                    wait_source = "per-minute rate limit, no Retry-After given"
                else:
                    wait = min(2 ** attempt, 60)
                    wait_source = "exponential backoff (server error)"

                print(f"  Mistral call failed ({status}, {type(e).__name__}). "
                      f"limit_per_min={limit_per_min} remaining_per_min={remaining_per_min}. "
                      f"Waiting {wait:.0f}s ({wait_source}), attempt {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(wait)
                continue
            raise  # 4xx other than 429/too-many-tokens (bad request, auth) won't be fixed by retrying

    raise RuntimeError(
        f"Mistral extraction failed after {MAX_RETRIES} retries. If every attempt above showed "
        f"remaining_requests=0 or remaining_tokens=0, this is quota exhaustion, not a timing issue — "
        f"check admin.mistral.ai -> Limits before re-running the whole ingestion job."
    ) from last_error


def _extract_batch_with_splitting(
    client: Mistral, pages: list[ParsedPage], reasoning_effort: str | None = None
) -> _BatchExtraction:
    """
    Extracts one batch of pages. _call_mistral_with_retry already retries
    the "too many tokens" error in place (see its docstring — real evidence
    points to this being a transient per-minute token budget, not a fixed
    size limit), so in practice this function's splitting logic is a
    SECOND line of defense: it only activates if MAX_RETRIES retries at
    the same size were already exhausted and the error is still happening.
    At that point, halving the batch is a reasonable fallback in case a
    batch is ever genuinely too large content-wise, not just rate-limited.

    Catches Exception broadly (same duck-typing reasoning as
    _call_mistral_with_retry) and unwraps __cause__ when the incoming
    exception is the "exhausted retries" RuntimeError that function raises,
    so the underlying too-many-tokens condition can still be recognized
    after it's been wrapped.

    A single page that's STILL too large alone (or still failing after
    retries) is skipped, with a printed warning, rather than aborting the
    entire extraction run — consistent with this file's existing "a missed
    fact is better than a wrong one" principle: a missed page is better
    than a failed report.
    """
    prompt = _build_prompt(pages)
    try:
        return _call_mistral_with_retry(client, prompt, reasoning_effort)
    except Exception as e:
        underlying = e.__cause__ if isinstance(e, RuntimeError) and e.__cause__ is not None else e
        if not _is_too_many_tokens_error(underlying):
            raise

        if len(pages) == 1:
            print(f"  Page {pages[0].page_number} is too large for a single request even alone — "
                  f"skipping it (its content, if any, won't be captured).")
            return _BatchExtraction(financial_facts=[], page_sections=[])

        mid = len(pages) // 2
        first_half, second_half = pages[:mid], pages[mid:]
        print(f"  Batch too large (pages {pages[0].page_number}-{pages[-1].page_number}), splitting into "
              f"{first_half[0].page_number}-{first_half[-1].page_number} and "
              f"{second_half[0].page_number}-{second_half[-1].page_number} and retrying...")
        result1 = _extract_batch_with_splitting(client, first_half, reasoning_effort)
        result2 = _extract_batch_with_splitting(client, second_half, reasoning_effort)
        return _BatchExtraction(
            financial_facts=result1.financial_facts + result2.financial_facts,
            page_sections=result1.page_sections + result2.page_sections,
        )


def extract_report(
    pages: list[ParsedPage],
    company: str,
    fiscal_year: int,
    source_report: str | None = None,
    api_key: str | None = None,
    batch_size: int = BATCH_SIZE_PAGES,
    reasoning_effort: str | None = None,
) -> tuple[list[FinancialFact], list[TextChunk]]:
    """
    Runs extraction over an entire report's pages, batched to control both
    request count (Mistral's free-tier RPS cap) and per-call context size.
    "other"-labeled pages (covers, TOC) are dropped rather than indexed.
    """
    api_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Set MISTRAL_API_KEY (or pass api_key=) before calling extract_report()")
    client = Mistral(api_key=api_key)

    all_facts: list[FinancialFact] = []
    all_chunks: list[TextChunk] = []

    batches = [pages[i:i + batch_size] for i in range(0, len(pages), batch_size)]
    for batch_num, batch in enumerate(batches, start=1):
        print(f"Extracting batch {batch_num}/{len(batches)} "
              f"(pages {batch[0].page_number}-{batch[-1].page_number})...")
        result = _extract_batch_with_splitting(client, batch, reasoning_effort)

        for fact in result.financial_facts:
            all_facts.append(FinancialFact(
                company=company, fiscal_year=fiscal_year, metric=fact.metric,
                value=fact.value, unit=fact.unit, source_page=fact.page,
                source_report=source_report,
            ))

        section_by_page = {s.page: s.section for s in result.page_sections}
        for page in batch:
            section = section_by_page.get(page.page_number, "other")
            if section == "other":
                continue
            for chunk_text in chunk_page_text(page.text):
                all_chunks.append(TextChunk(
                    company=company, fiscal_year=fiscal_year, page=page.page_number,
                    text=chunk_text, section=section, source_report=source_report,
                ))

    return all_facts, all_chunks


def _run_smoke_test() -> None:
    """
    Verifies prompt-building, batching, retry logic, and the
    Mistral-output -> FinancialFact/TextChunk conversion — everything except
    the actual round trip to Mistral, which needs a real MISTRAL_API_KEY and
    is meant to be tested live, the same way parse.py needed a real PDF.
    """
    from unittest.mock import MagicMock, patch
    import httpx

    fake_pages = [ParsedPage(page_number=i, text=f"Page {i} content") for i in range(1, 4)]
    prompt = _build_prompt(fake_pages)
    assert "[Page 1]" in prompt and "[Page 3]" in prompt
    assert "segment_revenue:<segment name>" in prompt
    print("_build_prompt: page markers and metric guidance present")

    # --- _call_mistral_with_retry: a real (non-zero) per-minute limit with
    # Retry-After present is respected, not guessed ---
    fake_429_response = httpx.Response(
        429, headers={"retry-after": "2", "x-ratelimit-limit-req-minute": "60",
                      "x-ratelimit-remaining-req-minute": "0"},
    )
    rate_limit_error = SDKError("Rate limit exceeded", raw_response=fake_429_response, body="{}")
    success_result = _BatchExtraction(financial_facts=[], page_sections=[])

    call_count = {"n": 0}

    def fake_parse(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise rate_limit_error
        response = MagicMock()
        response.choices[0].message.parsed = success_result
        return response

    fake_client = MagicMock()
    fake_client.chat.parse.side_effect = fake_parse

    with patch("time.sleep") as mock_sleep:
        result = _call_mistral_with_retry(fake_client, "some prompt")

    assert result is success_result, "should succeed on the second attempt after one 429"
    assert call_count["n"] == 2
    mock_sleep.assert_called_once_with(2.0)  # the Retry-After value, not a guessed backoff
    print("_call_mistral_with_retry: respected Retry-After=2s header instead of guessing a backoff")

    # --- a hard 0 requests/minute limit fails fast, with no retrying at all ---
    zero_quota_response = httpx.Response(
        429, headers={"x-ratelimit-limit-req-minute": "0", "x-ratelimit-remaining-req-minute": "0"},
    )
    zero_quota_error = SDKError("Rate limit exceeded", raw_response=zero_quota_response, body="{}")
    zero_quota_client = MagicMock()
    zero_quota_client.chat.parse.side_effect = zero_quota_error

    with patch("time.sleep") as mock_sleep_zero:
        try:
            _call_mistral_with_retry(zero_quota_client, "some prompt")
            raise AssertionError("expected RuntimeError for a hard 0 req/minute limit")
        except RuntimeError as e:
            assert "0 requests/minute" in str(e)
            assert "admin.mistral.ai" in str(e)

    mock_sleep_zero.assert_not_called()  # must fail fast, not burn through backoff first
    assert zero_quota_client.chat.parse.call_count == 1, "must not retry a hard 0-quota limit"
    print("_call_mistral_with_retry: a hard 0 req/minute limit fails immediately, no wasted retries")

    # --- _is_too_many_tokens_error: matches the real observed error, rejects lookalikes ---
    too_many_tokens_response = httpx.Response(400, headers={})
    too_many_tokens_body = ('{"object":"error","message":"Too many tokens overall, split into more '
                             'batches.","type":"invalid_request_prompt","param":null,"code":"3210"}')
    too_many_tokens_error = SDKError("API error occurred", too_many_tokens_response, too_many_tokens_body)
    assert _is_too_many_tokens_error(too_many_tokens_error) is True

    other_400_response = httpx.Response(400, headers={})
    other_400_error = SDKError("API error occurred", other_400_response,
                                '{"message":"Invalid model specified","type":"invalid_request_error","code":"1500"}')
    assert _is_too_many_tokens_error(other_400_error) is False, "a different 400 must not be treated as splittable"

    a_429_error = SDKError("API error occurred", httpx.Response(429, headers={}), '{"code":"1300"}')
    assert _is_too_many_tokens_error(a_429_error) is False, "wrong status code must not match"
    print("_is_too_many_tokens_error: matches the real error signature, rejects other 400s and non-400s")

    # --- _call_mistral_with_retry: the "too many tokens" 400 is retried in place
    # (waiting, then retrying the SAME prompt), not treated as an immediate failure —
    # this is the actual fix: real evidence (isolated re-runs of the "too large" batch
    # succeeding immediately) showed this is a transient condition, not a fixed size limit ---
    retry_call_count = {"n": 0}

    def fake_parse_transient_then_success(**kwargs):
        retry_call_count["n"] += 1
        if retry_call_count["n"] == 1:
            raise too_many_tokens_error
        response = MagicMock()
        response.choices[0].message.parsed = success_result
        return response

    transient_client = MagicMock()
    transient_client.chat.parse.side_effect = fake_parse_transient_then_success

    with patch("time.sleep") as mock_sleep_transient:
        transient_result = _call_mistral_with_retry(transient_client, "some prompt")

    assert transient_result is success_result
    assert retry_call_count["n"] == 2, "should retry once after the 'too many tokens' error, then succeed"
    mock_sleep_transient.assert_called_once_with(20.0)  # attempt 0 -> 20 * (0 + 1)
    print("_call_mistral_with_retry: retries the 'too many tokens' error in place instead of failing immediately")

    # --- the actual real-world scenario: an exception that is NOT an instance
    # of the SDKError class this file imports (simulating the class-identity
    # mismatch confirmed in production — `except SDKError` never fired there),
    # but has the same shape. Duck-typing must still catch and retry it. ---
    class _UnrelatedErrorClass(Exception):
        """Deliberately NOT a subclass of SDKError — same shape, different identity."""
        def __init__(self, message, raw_response, body):
            super().__init__(message)
            self.raw_response = raw_response
            self.body = body

    assert not issubclass(_UnrelatedErrorClass, SDKError), "test setup: this must NOT be an SDKError subclass"

    lookalike_error = _UnrelatedErrorClass("API error occurred", too_many_tokens_response, too_many_tokens_body)
    assert _is_too_many_tokens_error(lookalike_error) is True, \
        "duck-typing must recognize this even though it isn't an SDKError instance"

    duck_typed_call_count = {"n": 0}

    def fake_parse_unrelated_class_then_success(**kwargs):
        duck_typed_call_count["n"] += 1
        if duck_typed_call_count["n"] == 1:
            raise lookalike_error
        response = MagicMock()
        response.choices[0].message.parsed = success_result
        return response

    duck_typed_client = MagicMock()
    duck_typed_client.chat.parse.side_effect = fake_parse_unrelated_class_then_success

    with patch("time.sleep"):
        duck_typed_result = _call_mistral_with_retry(duck_typed_client, "some prompt")

    assert duck_typed_result is success_result
    assert duck_typed_call_count["n"] == 2, \
        "must retry an exception with the right shape even when it's not an SDKError instance"
    print("_call_mistral_with_retry: retries correctly via duck-typing even for a non-SDKError exception class")

    # --- _extract_batch_with_splitting: still recognizes the condition after
    # _call_mistral_with_retry wraps it in "exhausted retries" RuntimeError ---
    def fake_call_that_exhausts_retries(client, prompt, reasoning_effort=None):
        raise RuntimeError("Mistral extraction failed after 5 retries.") from too_many_tokens_error

    with patch(f"{__name__}._call_mistral_with_retry", side_effect=fake_call_that_exhausts_retries):
        wrapped_skip_result = _extract_batch_with_splitting(
            client=None, pages=[ParsedPage(page_number=754, text="dense page")], reasoning_effort=None,
        )
    assert wrapped_skip_result.financial_facts == [], \
        "must unwrap __cause__ to recognize too-many-tokens even after it's wrapped in RuntimeError"
    print("_extract_batch_with_splitting: still recognizes the condition after it's wrapped in a RuntimeError")

    # --- _extract_batch_with_splitting: real scenario — a 4-page batch where only the
    # last 2 pages (mimicking the actual dense-annexure case) are "too large" together,
    # forcing exactly one split before both halves succeed ---
    split_pages = [ParsedPage(page_number=i, text=f"Page {i}") for i in range(1, 5)]  # pages 1-4
    split_call_log = []

    def fake_call_that_rejects_large_batches(client, prompt, reasoning_effort=None):
        split_call_log.append(prompt)
        page_count_in_prompt = prompt.count("[Page ")
        if page_count_in_prompt > 2:
            raise too_many_tokens_error
        # one fact per page actually in this sub-batch, keyed off its first page number
        first_page = int(prompt.split("[Page ")[1].split("]")[0])
        return _BatchExtraction(
            financial_facts=[_ExtractedFact(metric="revenue", value=float(first_page), page=first_page)],
            page_sections=[_PageSection(page=first_page, section="risk_factors")],
        )

    with patch(f"{__name__}._call_mistral_with_retry", side_effect=fake_call_that_rejects_large_batches):
        merged = _extract_batch_with_splitting(client=None, pages=split_pages, reasoning_effort=None)

    assert len(split_call_log) == 3, f"expected 3 calls (1 rejected + 2 successful halves), got {len(split_call_log)}"
    assert len(merged.financial_facts) == 2, f"expected facts from both halves after merging, got {merged.financial_facts}"
    print(f"_extract_batch_with_splitting: a too-large 4-page batch split into 2+2, "
          f"both halves succeeded and merged into {len(merged.financial_facts)} facts")

    # --- a single page that's STILL too large alone is skipped, not fatal ---
    def always_reject(client, prompt, reasoning_effort=None):
        raise too_many_tokens_error

    with patch(f"{__name__}._call_mistral_with_retry", side_effect=always_reject):
        skipped_result = _extract_batch_with_splitting(
            client=None, pages=[ParsedPage(page_number=581, text="huge annexure")], reasoning_effort=None,
        )
    assert skipped_result.financial_facts == [] and skipped_result.page_sections == []
    print("_extract_batch_with_splitting: a single page too large even alone is skipped, not fatal")

    # --- an unrelated error is never treated as splittable ---
    def raise_unrelated_error(client, prompt, reasoning_effort=None):
        raise other_400_error

    with patch(f"{__name__}._call_mistral_with_retry", side_effect=raise_unrelated_error):
        try:
            _extract_batch_with_splitting(client=None, pages=split_pages, reasoning_effort=None)
            raise AssertionError("expected the unrelated 400 to propagate, not be caught as splittable")
        except SDKError:
            pass
    print("_extract_batch_with_splitting: an unrelated error propagates instead of being silently split")

    pages = [ParsedPage(page_number=i, text=f"Page {i} text") for i in range(1, 46)]  # 45 pages

    call_log = []

    def fake_call_with_retry(client, prompt, reasoning_effort="low"):
        call_log.append(prompt)
        batch_index = len(call_log)
        first_page_in_batch = pages[(batch_index - 1) * BATCH_SIZE_PAGES].page_number
        return _BatchExtraction(
            financial_facts=[_ExtractedFact(metric="revenue", value=100000.0 + batch_index,
                                             unit="INR crore", page=first_page_in_batch)],
            page_sections=[
                _PageSection(page=first_page_in_batch, section="risk_factors"),
                _PageSection(page=first_page_in_batch + 1, section="other"),
            ],
        )

    with patch(f"{__name__}._call_mistral_with_retry", side_effect=fake_call_with_retry):
        facts, chunks = extract_report(
            pages, company="Tata Steel", fiscal_year=2024,
            source_report="tatasteel_fy24.pdf", api_key="fake-key-for-smoke-test",
        )

    expected_batches = 3  # 45 pages / 20 per batch -> 20, 20, 5
    assert len(call_log) == expected_batches, f"expected {expected_batches} Mistral calls, got {len(call_log)}"
    assert len(facts) == expected_batches, f"expected {expected_batches} facts, got {len(facts)}"
    assert facts[0].company == "Tata Steel" and facts[0].fiscal_year == 2024
    assert facts[0].source_report == "tatasteel_fy24.pdf"

    assert len(chunks) == expected_batches, f"expected {expected_batches} chunk groups, got {len(chunks)}"
    assert all(c.section == "risk_factors" for c in chunks)
    other_pages_indexed = [c for c in chunks if c.page not in
                            [pages[(b) * BATCH_SIZE_PAGES].page_number for b in range(expected_batches)]]
    assert not other_pages_indexed, "an 'other'-labeled page was indexed and should have been dropped"

    print(f"extract_report: {expected_batches} batches called, {len(facts)} facts extracted, "
          f"{len(chunks)} chunks kept, 'other' pages correctly dropped")

    print("\nAll extract.py smoke tests passed (real Mistral call untested — needs a live MISTRAL_API_KEY).")


if __name__ == "__main__":
    _run_smoke_test()