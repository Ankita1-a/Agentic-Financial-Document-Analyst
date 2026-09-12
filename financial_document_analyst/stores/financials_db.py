"""
stores/financials_db.py

Structured, queryable storage for numeric facts pulled out of annual
reports (revenue, net profit, segment figures, etc.) — one row per
(company, fiscal_year, metric).

This is the layer that makes "analyze the revenue trend" trustworthy: the
agent never estimates growth from retrieved text, it calls
get_financial_series() and computes YoY growth / CAGR here with plain
arithmetic, then only asks the LLM to narrate the result.

No LLM calls happen in this file.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FinancialFact:
    company: str
    fiscal_year: int          # e.g. 2024, meaning FY2024
    metric: str                # e.g. "revenue", "net_profit", "segment_revenue:steel"
    value: float
    unit: str | None = None    # e.g. "INR crore"
    source_page: int | None = None
    source_report: str | None = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS financial_facts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    company       TEXT NOT NULL,
    fiscal_year   INTEGER NOT NULL,
    metric        TEXT NOT NULL,
    value         REAL NOT NULL,
    unit          TEXT,
    source_page   INTEGER,
    source_report TEXT,
    UNIQUE(company, fiscal_year, metric)
);
"""


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def upsert_fact(db_path: str | Path, fact: FinancialFact) -> None:
    """Insert a fact, or overwrite it if the same (company, fiscal_year, metric) already exists.

    Overwrite-on-conflict matters because re-ingesting a report (e.g. after
    fixing an extraction bug) should correct the stored number, not duplicate it.
    """
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO financial_facts (company, fiscal_year, metric, value, unit, source_page, source_report)
            VALUES (:company, :fiscal_year, :metric, :value, :unit, :source_page, :source_report)
            ON CONFLICT(company, fiscal_year, metric) DO UPDATE SET
                value = excluded.value,
                unit = excluded.unit,
                source_page = excluded.source_page,
                source_report = excluded.source_report
            """,
            fact.__dict__,
        )


def upsert_facts_bulk(db_path: str | Path, facts: list[FinancialFact]) -> int:
    """Upsert many facts in one transaction. Returns the count written."""
    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO financial_facts (company, fiscal_year, metric, value, unit, source_page, source_report)
            VALUES (:company, :fiscal_year, :metric, :value, :unit, :source_page, :source_report)
            ON CONFLICT(company, fiscal_year, metric) DO UPDATE SET
                value = excluded.value,
                unit = excluded.unit,
                source_page = excluded.source_page,
                source_report = excluded.source_report
            """,
            [f.__dict__ for f in facts],
        )
    return len(facts)


def list_companies(db_path: str | Path) -> list[str]:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT company FROM financial_facts ORDER BY company").fetchall()
    return [r["company"] for r in rows]


def list_metrics(db_path: str | Path, company: str) -> list[str]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT metric FROM financial_facts WHERE company = ? ORDER BY metric",
            (company,),
        ).fetchall()
    return [r["metric"] for r in rows]


def get_metric_unit(db_path: str | Path, company: str, metric: str) -> str | None:
    """
    Returns the stored unit (e.g. "INR crore") for one company/metric, or
    None if there's no matching row or the unit was never recorded.

    Added after a real run showed the agent fabricating a unit ("million")
    and an incorrect unit conversion ("billion", off by 100x) when asked
    to state a revenue figure — the tool output it was working from never
    included the actual unit at all, even though financials_db has always
    stored it. Kept as its own small query rather than folded into
    get_financial_series() so that function's (year, value) return shape —
    which compute_yoy_growth/compute_cagr and existing callers all
    depend on — doesn't need to change.
    """
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT unit FROM financial_facts WHERE company = ? AND metric = ? LIMIT 1",
            (company, metric),
        ).fetchone()
    return row["unit"] if row else None


def get_financial_series(
    db_path: str | Path,
    company: str,
    metric: str,
    start_year: int | None = None,
    end_year: int | None = None,
) -> list[tuple[int, float]]:
    """Returns [(fiscal_year, value), ...] sorted ascending by year, optionally bounded."""
    query = "SELECT fiscal_year, value FROM financial_facts WHERE company = ? AND metric = ?"
    params: list = [company, metric]
    if start_year is not None:
        query += " AND fiscal_year >= ?"
        params.append(start_year)
    if end_year is not None:
        query += " AND fiscal_year <= ?"
        params.append(end_year)
    query += " ORDER BY fiscal_year ASC"

    with get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [(r["fiscal_year"], r["value"]) for r in rows]


def compute_yoy_growth(series: list[tuple[int, float]]) -> list[tuple[int, float | None]]:
    """
    Year-over-year % growth for each year relative to the previous one.
    First year in the series has no prior point, so its growth is None.
    Also returns None (instead of raising) if the prior value is zero, since
    percent growth from zero is undefined.
    """
    growth: list[tuple[int, float | None]] = []
    for i, (year, value) in enumerate(series):
        if i == 0:
            growth.append((year, None))
            continue
        prev_value = series[i - 1][1]
        if prev_value == 0:
            growth.append((year, None))
        else:
            growth.append((year, (value - prev_value) / prev_value * 100))
    return growth


def compute_cagr(series: list[tuple[int, float]]) -> float | None:
    """
    Compound annual growth rate between the first and last point in the series.
    Returns None if there are fewer than 2 points, the span is zero years, or
    the start value isn't positive (CAGR is undefined for zero/negative bases).
    """
    if len(series) < 2:
        return None

    start_year, start_value = series[0]
    end_year, end_value = series[-1]
    num_years = end_year - start_year

    if num_years <= 0 or start_value <= 0:
        return None

    return ((end_value / start_value) ** (1 / num_years) - 1) * 100


def _run_smoke_test() -> None:
    """Exercises every function above against a throwaway in-memory-style db file."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_financials.sqlite"

        facts = [
            FinancialFact("Tata Steel", 2021, "revenue", 156294, "INR crore", source_page=42),
            FinancialFact("Tata Steel", 2022, "revenue", 243959, "INR crore", source_page=45),
            FinancialFact("Tata Steel", 2023, "revenue", 227972, "INR crore", source_page=44),
            FinancialFact("Tata Steel", 2024, "revenue", 218577, "INR crore", source_page=43),
            FinancialFact("L&T", 2021, "revenue", 143633, "INR crore", source_page=30),
            FinancialFact("L&T", 2022, "revenue", 162034, "INR crore", source_page=31),
            FinancialFact("L&T", 2023, "revenue", 183341, "INR crore", source_page=29),
            FinancialFact("L&T", 2024, "revenue", 222866, "INR crore", source_page=28),
        ]
        written = upsert_facts_bulk(db_path, facts)
        assert written == 8, f"expected 8 facts written, got {written}"

        companies = list_companies(db_path)
        assert companies == ["L&T", "Tata Steel"], f"unexpected companies: {companies}"

        metrics = list_metrics(db_path, "Tata Steel")
        assert metrics == ["revenue"], f"unexpected metrics: {metrics}"

        tata_series = get_financial_series(db_path, "Tata Steel", "revenue")
        assert tata_series == [(2021, 156294), (2022, 243959), (2023, 227972), (2024, 218577)]

        # get_metric_unit: added after a real run showed the agent fabricating a wrong unit
        # ("million") and a wrong conversion ("billion") when the tool output never told it
        # the actual unit at all.
        unit = get_metric_unit(db_path, "Tata Steel", "revenue")
        assert unit == "INR crore", f"expected the stored unit, got {unit!r}"
        missing_unit = get_metric_unit(db_path, "Tata Steel", "nonexistent_metric")
        assert missing_unit is None, "a metric with no rows should return None, not raise"
        print(f"get_metric_unit: correctly returns the stored unit ({unit!r}), None for a missing metric")

        # Bounded query
        bounded = get_financial_series(db_path, "Tata Steel", "revenue", start_year=2022, end_year=2023)
        assert bounded == [(2022, 243959), (2023, 227972)], f"bounded query wrong: {bounded}"

        # Re-upsert the same key with a corrected value -> should overwrite, not duplicate
        upsert_fact(db_path, FinancialFact("Tata Steel", 2024, "revenue", 218578, "INR crore"))
        corrected = get_financial_series(db_path, "Tata Steel", "revenue")
        assert len(corrected) == 4, "overwrite created a duplicate row instead of updating"
        assert corrected[-1] == (2024, 218578), f"overwrite did not take effect: {corrected[-1]}"

        growth = compute_yoy_growth(tata_series)
        print("Tata Steel YoY revenue growth (%):")
        for year, g in growth:
            print(f"  FY{year}: {'n/a' if g is None else f'{g:+.1f}%'}")

        cagr = compute_cagr(tata_series)
        print(f"Tata Steel revenue CAGR FY2021-FY2024: {cagr:.1f}%")

        lt_series = get_financial_series(db_path, "L&T", "revenue")
        lt_cagr = compute_cagr(lt_series)
        print(f"L&T revenue CAGR FY2021-FY2024: {lt_cagr:.1f}%")

        # Edge cases: too few points, zero start value
        assert compute_cagr([(2024, 100)]) is None, "CAGR should be None with a single point"
        assert compute_cagr([(2021, 0), (2024, 100)]) is None, "CAGR should be None with a zero base"
        zero_growth = compute_yoy_growth([(2021, 0), (2022, 50)])
        assert zero_growth[1][1] is None, "YoY growth from a zero base should be None, not a crash"

        print("\nAll financials_db smoke tests passed.")


if __name__ == "__main__":
    _run_smoke_test()