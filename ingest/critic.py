"""
ingest/critic.py

Verifies extracted financial facts before they're trusted enough to store:

1. Metric name normalization — collapses report-specific naming variants
   (e.g. "turnover_india", "ebitda_india") onto a fixed canonical vocabulary
   so get_financial_series() works across different reports/companies
   without silently returning an empty series because of a naming mismatch.

2. Value grounding — checks that each fact's numeric value actually appears
   (in some plausible textual form — Indian or Western digit grouping, with
   or without decimals) on the page it claims to be sourced from. A fact
   whose value can't be found on its cited page is flagged for review
   rather than silently trusted or silently dropped.

This is deliberately all deterministic, code-only logic — no second LLM
call. It's cheap, instant, and either finds the exact digits on the page
or it doesn't; there's nothing here that benefits from asking a model to
look again.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ingest.parse import ParsedPage
from stores.financials_db import FinancialFact

_METRIC_SYNONYMS = {
    "turnover": "revenue",
    "total_revenue": "revenue",
    "total_income": "revenue",
    "net_sales": "revenue",
    "pat": "net_profit",
    "profit_after_tax": "net_profit",
    "profit_for_the_year": "net_profit",
    "net_income": "net_profit",
    "shareholders_equity": "total_equity",
    "net_worth": "total_equity",
    "cash_flow_from_operating_activities": "cash_flow_from_operations",
    "operating_cash_flow": "cash_flow_from_operations",
}

_KNOWN_REGIONS = {
    "india", "europe", "uk", "netherlands", "us", "usa", "china",
    "thailand", "southeast_asia", "north_america",
}
_GEOGRAPHY_ELIGIBLE_BASES = {"revenue", "ebitda", "net_profit"}


def normalize_metric_name(metric: str) -> str:
    """
    Collapses known naming variants onto a fixed canonical vocabulary.
    Unrecognized names are returned unchanged rather than guessed at —
    callers should treat those as worth a human glance, not as an error.
    """
    metric = metric.strip().lower().replace(" ", "_")

    if ":" in metric:
        prefix, _, suffix = metric.partition(":")
        prefix = _METRIC_SYNONYMS.get(prefix, prefix)
        return f"{prefix}:{suffix}"

    if metric in _METRIC_SYNONYMS:
        return _METRIC_SYNONYMS[metric]

    base, sep, suffix = metric.rpartition("_")
    if sep and suffix in _KNOWN_REGIONS:
        canonical_base = _METRIC_SYNONYMS.get(base, base)
        if canonical_base in _GEOGRAPHY_ELIGIBLE_BASES:
            return f"geography_{canonical_base}:{suffix}"

    return metric


def format_indian_grouping(value: float) -> str:
    """
    Formats a number using Indian digit grouping (last 3 digits together,
    then pairs going left): 232140 -> "2,32,140", 34848 -> "34,848",
    140302 -> "1,40,302". Matches how Indian annual reports print rupee
    figures.
    """
    is_negative = value < 0
    value = abs(value)
    int_part = int(value)
    frac = round(value - int_part, 2)

    s = str(int_part)
    if len(s) <= 3:
        grouped = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        groups: list[str] = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        grouped = ",".join(groups) + "," + last3

    if frac:
        grouped += f"{frac:.2f}".lstrip("0")

    return ("-" if is_negative else "") + grouped


def format_western_grouping(value: float) -> str:
    """Standard comma-every-3-digits formatting: 232140 -> "232,140"."""
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:,.2f}"


def _candidate_representations(value: float) -> list[str]:
    candidates = {
        format_indian_grouping(value),
        format_western_grouping(value),
        str(int(value)) if value == int(value) else str(value),
    }
    if value != int(value):
        candidates.add(f"{value:.1f}")
        candidates.add(f"{value:.2f}")
    return list(candidates)


def verify_fact_against_page(value: float, page_text: str) -> bool:
    """
    Returns True if some plausible textual representation of `value`
    appears on the given page's text. This is a grounding heuristic, not a
    proof of correctness — formatting quirks (a number split across a
    markdown table, unusual rounding) can produce a false "not found" on an
    actually-correct fact. Treat a False result as "worth a human glance",
    not as "definitely wrong".
    """
    normalized_text = re.sub(r"\s+", " ", page_text)
    return any(candidate in normalized_text for candidate in _candidate_representations(value))


@dataclass
class CritiqueResult:
    verified_facts: list[FinancialFact]
    flagged_facts: list[FinancialFact]


def critique_facts(facts: list[FinancialFact], pages: list[ParsedPage]) -> CritiqueResult:
    """
    Normalizes every fact's metric name, then checks its value against the
    text of the page it claims to be sourced from. Verified facts are ready
    for financials_db.py; flagged facts should be reviewed, not discarded —
    a formatting-driven false flag is more likely than an actual extraction
    error, but only a human (or a follow-up LLM pass) can tell the two apart.
    """
    page_text_by_number = {p.page_number: p.text for p in pages}
    verified: list[FinancialFact] = []
    flagged: list[FinancialFact] = []

    for fact in facts:
        normalized = FinancialFact(
            company=fact.company, fiscal_year=fact.fiscal_year,
            metric=normalize_metric_name(fact.metric), value=fact.value,
            unit=fact.unit, source_page=fact.source_page, source_report=fact.source_report,
        )
        page_text = page_text_by_number.get(fact.source_page, "")
        if verify_fact_against_page(normalized.value, page_text):
            verified.append(normalized)
        else:
            flagged.append(normalized)

    return CritiqueResult(verified_facts=verified, flagged_facts=flagged)