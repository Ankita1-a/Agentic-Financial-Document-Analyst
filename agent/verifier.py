"""
agent/verifier.py

Grounds the agent's draft final answer against what its own tools actually
returned during that turn — catching a hallucinated or garbled number
before it reaches the user. Different from ingest/critic.py, which checks
facts once, at ingestion time, against the source PDF page; this checks
the live conversational answer against the live tool-call results from
the same turn, every time.

Built in direct response to a real, reproduced failure: open-mistral-nemo
correctly retrieved Tata Steel's revenue (232139.94 crore, matching
independently-verified published figures) via compute_trend, then stated
"₹23,213.99 crores" in its prose answer — a garbled transcription of a
correct number, not a data pipeline error. Confirmed by checking
financials_db.sqlite directly: the stored value was right, so the bug was
purely in text generation, not extraction or critique. This module cannot
fix that generation error, but it can catch it before the answer reaches
the user, which is the whole point.

This is a plain number-matching heuristic, not a semantic fact-checker —
it does not judge whether the agent's INTERPRETATION of the data is
sound, only whether the numbers it cites actually appeared somewhere in
its own tool results this turn. False negatives are possible (a
legitimately-derived number, like an average the agent computed across
two tool results, won't match either result alone) — but catching outright
transcription errors on the current turn's own data is squarely this
heuristic's strength, and it needs no LLM call to do it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUMBER_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?")
_URL_PATTERN = re.compile(r"https?://\S+")


def _strip_urls(text: str) -> str:
    """A URL's digits (path segments, slugs, tracking IDs) are never a
    financial figure — e.g. a real answer citing a Business Standard
    article at .../kalinganagar-project-expansion-by-dec-2024-123121700377_1.html
    had "123,121,700,377" flagged as an unmatched figure before this existed,
    since it passed every other exclusion (too large for a year, too large
    for a small count). Applied before number-extraction on BOTH the
    agent's answer text and tool results' string fields — the latter
    matters too, or a URL's digits could end up as "valid grounding
    evidence" for a citation of that same URL, which would just move this
    bug to the other side of the check instead of fixing it."""
    return _URL_PATTERN.sub("", text)


def _extract_numbers_from_text(text: str) -> list[float]:
    """
    Pulls out numeric figures mentioned in a piece of text, handling comma
    grouping (Indian or Western — both just get stripped) and decimals.

    Three deliberate exclusions to avoid flooding the caller with false
    positives on numbers that were never meant to be verified against data:
    - A bare whole number in the 1900-2100 range is treated as a calendar
      year ("FY2024", "in 2024"), not a figure to check.
    - A bare whole number under 10 is treated as a likely count or
      list-position ("2 companies", "the 3rd point"), not a financial
      figure. A number with a genuine decimal point is never excluded this
      way, since small decimals are almost always meaningful (a CAGR of
      "3.5%" still deserves checking).
    - A whole-number multiple of 5 in the 0-100 range is treated as a
      likely illustrative threshold, not a specific claim. Confirmed
      necessary after a real response mixed a genuine data point (a
      correct, and correctly unflagged, dividend figure) with a general
      explanation of what a payout ratio means, using round rule-of-thumb
      percentages ("a ratio below 30%", "above 70%", "exceeds 100%") that
      have no tool result to match because they were never claims about
      the company at all. This is a real trade-off, not a clean fix:
      distinguishing "a fact about this company" from "an illustrative
      example while explaining a concept" is a semantic judgment regex
      can't make reliably. Every genuine transcription error caught so
      far (23,213.99; 23,213.94) has been an oddly-precise wrong number,
      never a clean round one — so this exclusion trades a small, accepted
      risk of missing a genuinely wrong ROUND percentage for a large
      reduction in false alarms on ordinary explanatory content, which
      would otherwise erode trust in the warning through overuse.

    The regex only treats "." as a decimal point when at least one digit
    follows it immediately — confirmed necessary after a real false
    positive: a markdown numbered list ("1. Do X", "2. Do Y") was matching
    "1." as a decimal number 1.0, which has a "." in it and so skipped the
    small-integer exclusion above, flagging list markers as unmatched
    "figures" on every response that used one.
    """
    numbers = []
    for match in _NUMBER_TOKEN.finditer(_strip_urls(text)):
        raw = match.group().replace(",", "")
        if raw in ("", "."):
            continue
        try:
            value = float(raw)
        except ValueError:
            continue

        is_whole_number = "." not in raw
        if is_whole_number and 1900 <= value <= 2100:
            continue
        if is_whole_number and value < 10:
            continue
        if is_whole_number and 0 <= value <= 100 and value % 5 == 0:
            continue

        numbers.append(value)
    return numbers


def _extract_all_numbers_from_string(text: str) -> list[float]:
    """
    Pulls every digit sequence out of a string as potential grounding
    evidence — deliberately WITHOUT the exclusions in
    _extract_numbers_from_text (calendar years, small counts, round
    illustrative percentages). Those exclusions are tuned for what's worth
    FLAGGING in the agent's own prose; here we're building the set of
    numbers the agent is allowed to cite, where maximum recall only ever
    makes the check more lenient, never introduces a false alarm. URLs are
    still stripped first — see _strip_urls.
    """
    numbers = []
    for match in _NUMBER_TOKEN.finditer(_strip_urls(text)):
        raw = match.group().replace(",", "")
        if raw in ("", "."):
            continue
        try:
            numbers.append(float(raw))
        except ValueError:
            continue
    return numbers


def _extract_numbers_from_value(value) -> list[float]:
    """Recursively collects every numeric figure from a (possibly nested)
    tool result — both real int/float leaves (get_financial_series,
    compute_trend, ...) AND numbers embedded in free-text string fields
    (web_search's "snippet"/"title", vector_search's retrieved text). Without
    the string case, a real, correctly-cited figure from a web search
    result (e.g. "£1.25 billion" in a snippet) would always be flagged as
    unmatched, since it never appears as a clean numeric value anywhere in
    that tool's JSON — confirmed by reproducing this exact failure with a
    real web_search-style result below."""
    numbers: list[float] = []
    if isinstance(value, bool):
        return numbers  # bool is a subclass of int in Python — never a figure worth matching
    if isinstance(value, (int, float)):
        numbers.append(float(value))
    elif isinstance(value, str):
        numbers.extend(_extract_all_numbers_from_string(value))
    elif isinstance(value, dict):
        for v in value.values():
            numbers.extend(_extract_numbers_from_value(v))
    elif isinstance(value, list):
        for v in value:
            numbers.extend(_extract_numbers_from_value(v))
    return numbers


def _matches_any(value: float, candidates: list[float], relative_tolerance: float = 0.01) -> bool:
    """
    True if value is within 1% of any candidate — allows for the model's
    own reasonable rounding when writing prose (e.g. "roughly 232,140" for
    an underlying 232139.94), without being loose enough to wave through a
    genuinely different number.
    """
    for c in candidates:
        if c == 0:
            if value == 0:
                return True
            continue
        if abs(value - c) / abs(c) <= relative_tolerance:
            return True
    return False


@dataclass
class VerificationResult:
    verified: bool
    unmatched_numbers: list[float] = field(default_factory=list)


def verify_answer(final_text: str, tool_results: list[dict]) -> VerificationResult:
    """
    Checks every number-like figure mentioned in final_text against every
    numeric value that appeared anywhere in this turn's tool results.
    Numbers that don't match anything are returned as unmatched_numbers —
    this function only detects a mismatch, it doesn't correct one; the
    caller (orchestrator.py's ask()) decides what to do with the result.
    """
    candidate_numbers: list[float] = []
    for result in tool_results:
        candidate_numbers.extend(_extract_numbers_from_value(result))

    text_numbers = _extract_numbers_from_text(final_text)
    unmatched = [n for n in text_numbers if not _matches_any(n, candidate_numbers)]

    return VerificationResult(verified=not unmatched, unmatched_numbers=unmatched)


def _run_smoke_test() -> None:
    # --- the actual real bug, reproduced exactly ---
    real_tool_result = {
        "company": "Tata Steel", "metric": "revenue",
        "series": [{"fiscal_year": 2025, "value": 232139.94}],
        "yoy_growth_pct": [{"fiscal_year": 2025, "growth_pct": None}],
        "cagr_pct": None,
    }
    garbled_answer = "Tata Steel's revenue for FY2025 was approximately ₹23,213.99 crores."
    result = verify_answer(garbled_answer, [real_tool_result])
    assert result.verified is False
    assert any(abs(n - 23213.99) < 0.01 for n in result.unmatched_numbers)
    print(f"verify_answer: catches the exact real garbled-number bug (unmatched: {result.unmatched_numbers})")

    # --- a correct answer, including reasonable rounding, passes cleanly ---
    correct_answer = "Tata Steel's revenue for FY2025 was approximately ₹2,32,140 crores (232139.94 exactly)."
    result2 = verify_answer(correct_answer, [real_tool_result])
    assert result2.verified is True, f"a correct, reasonably-rounded answer should verify, got {result2.unmatched_numbers}"
    print("verify_answer: a correct answer (with reasonable rounding) verifies cleanly")

    # --- calendar years and small counts are never flagged ---
    text_with_year_and_count = "In FY2024, we compared 2 companies across their revenue trends."
    result3 = verify_answer(text_with_year_and_count, [])
    assert result3.verified is True, f"years/small counts should never be flagged, got {result3.unmatched_numbers}"
    print("verify_answer: calendar years and small bare counts are correctly ignored")

    # --- REGRESSION: markdown numbered list markers must never be flagged ---
    # Real false positive: "1. Do X\n2. Do Y\n3. Do Z" was matching "1." as
    # the decimal number 1.0, which bypassed the small-integer exclusion
    # above (since it "had a decimal point"), flagging 1.00/2.00/3.00 as
    # unmatched figures on every response that used a numbered list.
    numbered_list_text = (
        "Would you like me to:\n1. Analyze a specific segment?\n2. Search the MD&A?\n3. Check another metric?"
    )
    result_list = verify_answer(numbered_list_text, [])
    assert result_list.verified is True, \
        f"numbered list markers must not be flagged as figures, got {result_list.unmatched_numbers}"
    print("verify_answer: markdown numbered list markers (1. 2. 3.) are correctly ignored, not treated as decimals")

    # --- a legitimate small decimal (a CAGR percentage) IS checked, and matches ---
    cagr_result = {"company": "Tata Steel", "metric": "revenue", "cagr_pct": 11.8}
    cagr_answer = "Tata Steel's revenue grew at an 11.8% CAGR."
    result4 = verify_answer(cagr_answer, [cagr_result])
    assert result4.verified is True
    print("verify_answer: a small but meaningful decimal (CAGR%) is checked and correctly matches")

    # --- a wrong CAGR (small decimal) is still caught, proving the "ignore small numbers" rule
    # only applies to bare whole numbers, not decimals ---
    wrong_cagr_answer = "Tata Steel's revenue grew at a 25.3% CAGR."
    result5 = verify_answer(wrong_cagr_answer, [cagr_result])
    assert result5.verified is False
    assert any(abs(n - 25.3) < 0.01 for n in result5.unmatched_numbers)
    print("verify_answer: a wrong small decimal (CAGR%) is still caught, not swept in with the small-int exclusion")

    # --- REGRESSION: a correct data point mixed with a general conceptual explanation using
    # round illustrative percentages must not get flooded with false positives ---
    dividend_result = {"company": "Tata Steel", "metric": "dividend_per_share", "unit": None,
                        "series": [{"fiscal_year": 2025, "value": 4.0}]}
    mixed_answer = (
        "The dividend per share for Tata Steel in FY25 was ₹4.00.\n\n"
        "A payout ratio below 30% suggests the company retains earnings for growth, while a ratio "
        "above 70% indicates aggressive dividend payments. If earnings fall while dividends stay "
        "flat, the ratio could exceed 100%, forcing a cut."
    )
    result_mixed = verify_answer(mixed_answer, [dividend_result])
    assert result_mixed.verified is True, \
        f"round illustrative percentages in a general explanation must not be flagged, got {result_mixed.unmatched_numbers}"
    print("verify_answer: a correct data point + general conceptual explanation (round illustrative "
          "percentages) verifies cleanly, no false alarms")

    # --- numbers spread across multiple tool results (a comparison) are all checked correctly ---
    tata_result = {"company": "Tata Steel", "cagr_pct": 11.8}
    lt_result = {"company": "L&T", "cagr_pct": 15.8}
    comparison_answer = "Tata Steel grew at 11.8% CAGR, while L&T grew faster at 15.8% CAGR over the same period."
    result6 = verify_answer(comparison_answer, [tata_result, lt_result])
    assert result6.verified is True
    print("verify_answer: numbers correctly matched against results from MULTIPLE tool calls in one turn")

    # --- REGRESSION: the exact real bug from a live run — a URL embedded in a markdown
    # citation link had a long digit sequence in its slug (.../dec-2024-123121700377_1.html)
    # flagged as an unmatched "figure" of 123,121,700,377.00, even though it's a URL fragment ---
    web_result = {
        "title": "Tata Steel Annual Report 2023-2024",
        "snippet": "In September 2023, Tata Steel reached an agreement with the UK government to "
                   "jointly invest £1.25 billion (including a £500 million grant) in a new EAF.",
        "url": "https://www.business-standard.com/companies/news/tata-steel-aims-to-complete-"
               "kalinganagar-project-expansion-by-dec-2024-123121700377_1.html",
    }
    answer_with_citation = (
        "Tata Steel has partnered with the UK government to invest **£1.25 billion** (including a "
        "£500 million grant) in a new electric arc furnace project.\n"
        "*Source: [Business Standard](https://www.business-standard.com/companies/news/tata-steel-"
        "aims-to-complete-kalinganagar-project-expansion-by-dec-2024-123121700377_1.html)*"
    )
    result7 = verify_answer(answer_with_citation, [web_result])
    assert result7.verified is True, (
        f"a web-search-grounded answer with a URL citation must verify cleanly, "
        f"got unmatched: {result7.unmatched_numbers}"
    )
    assert not any(n > 1_000_000 for n in result7.unmatched_numbers), \
        "no URL-slug-sized number should ever reach unmatched_numbers"
    print("verify_answer: REGRESSION — the real URL-false-positive bug (123,121,700,377 from a URL "
          "slug) is fixed, and the genuinely-cited £1.25bn/£500m figures from the web_search "
          "snippet correctly verify as grounded")

    # --- a number that ISN'T actually in the snippet, but LOOKS like a plausible web citation,
    # must still be caught — confirms the fix didn't make grounding checks toothless ---
    fabricated_answer = "Tata Steel is investing £2.75 billion in the new EAF project."
    result8 = verify_answer(fabricated_answer, [web_result])
    assert result8.verified is False and any(abs(n - 2.75) < 0.01 for n in result8.unmatched_numbers)
    print("verify_answer: a genuinely fabricated web-sourced figure (£2.75bn, not in the snippet) "
          "is still correctly caught — the fix adds recall without losing precision")

    # --- a URL's own digits must NOT become valid "grounding evidence" for citing that
    # same number as if it were a real figure (the fix must not just move the bug) ---
    url_as_fake_citation = "Tata Steel's investment was approximately 123,121,700,377."
    result9 = verify_answer(url_as_fake_citation, [web_result])
    assert result9.verified is False, (
        "a URL's digits must never validate a citation of that same number as a real figure — "
        f"got verified=True with candidates apparently including the URL slug"
    )
    print("verify_answer: a URL's own digits are correctly EXCLUDED from valid grounding evidence too "
          "— citing '123,121,700,377' as a real figure is still (correctly) flagged")

    print("\nAll verifier.py smoke tests passed.")


if __name__ == "__main__":
    _run_smoke_test()