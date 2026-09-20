"""
agent/tools.py

The functions the agent (in agent/orchestrator.py) can call. Each one is
a thin wrapper over stores/financials_db.py or stores/vectorstore.py —
this file does no reasoning of its own, it only fetches data and, for
trend/comparison tools, runs the deterministic math already built in
financials_db.py.

make_agent_tools() returns a plain dict of {tool_name: callable}, each
closed over one session's specific store paths via a closure rather than
module-level globals — so multiple stores (e.g. one per test, or one per
concurrent session later) never share state by accident.

Every tool returns plain JSON-serializable data (dicts/lists/strings), not
dataclasses or tuples, since this is what gets handed back to the agent as a
function-call result.

Two tools exist purely to help the agent reason about what's available
before guessing: list_companies() and list_available_metrics(). A user
will type "tatasteel" or "L&T"; rather than hard-coding fuzzy-matching
logic here, the agent itself resolves that against list_companies()'s
exact stored names — that's exactly the kind of judgment call a "thinking"
agent should make, not something to pre-decide in code.

IMPORTANT: deliberately no `from __future__ import annotations` here,
unlike every other file in this project. Confirmed by reading
google-genai's actual automatic-function-calling code
(_extra_utils.convert_if_exist_pydantic_model) and reproducing the failure
offline: it validates a tool call's arguments with
isinstance(value, param.annotation) using the REAL annotation object from
inspect.signature(). With the future import, every annotation in this file
would be a string ('str' instead of the str type), and
isinstance(value, 'str') raises TypeError — which broke every single tool
call in a real run. This file is the one place in the project where that
otherwise-harmless style choice actually matters at runtime, because it's
the only file whose signatures get introspected by someone else's code.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import builtins
import contextlib
import io
import math
import statistics

from chromadb.api.types import EmbeddingFunction

from stores import financials_db
from stores.vectorstore import get_collection, search

# --- run_python: a lightweight sandbox for custom calculations ------------
# Restricted-builtins exec() is a reasonable defense for THIS threat model —
# a single local user, with an LLM that's cooperating on the task rather
# than deliberately attacking it — but it is NOT a hardened sandbox. It has
# no protection against determined interpreter-escape techniques (e.g.
# walking ().__class__.__bases__ to reach arbitrary classes), which real
# isolation (a subprocess with resource limits, or a dedicated sandboxing
# library) would close off. If this tool is ever exposed to untrusted
# multi-user input, replace this with real process isolation before doing
# so — don't extend the pattern below to cover that case.

_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
    "len", "list", "max", "min", "print", "range", "round", "sorted",
    "str", "sum", "tuple", "zip",
]
_SAFE_BUILTINS = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES if hasattr(builtins, name)}
_SAFE_BUILTINS.update({"True": True, "False": False, "None": None})

_BLOCKED_SUBSTRINGS = [
    "__", "import", "open(", "eval(", "exec(", "compile(",
    "getattr(", "setattr(", "delattr(", "globals(", "locals(", "vars(",
]


def _execute_sandboxed_python(code: str, scratchpad: dict) -> dict:
    """
    Executes a short Python snippet with no file/network/OS access and no
    dynamic-import or introspection builtins. `scratchpad` is available
    inside the code (read it, don't need to reassign it) with every
    previous tool result from this turn, keyed by call id. Assign the
    final answer to a variable named `result`; anything printed is
    captured too.
    """
    lowered = code.lower()
    for pattern in _BLOCKED_SUBSTRINGS:
        if pattern in lowered:
            return {"error": f"Blocked: code contains disallowed pattern {pattern!r}. "
                              f"This sandbox only supports basic arithmetic/data processing "
                              f"over `scratchpad` and the math/statistics modules — no imports, "
                              f"attribute introspection, or dynamic execution."}

    output_buffer = io.StringIO()
    sandbox_globals = {
        "__builtins__": _SAFE_BUILTINS,
        "math": math,
        "statistics": statistics,
        "scratchpad": scratchpad,
    }
    sandbox_locals: dict = {}
    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, sandbox_globals, sandbox_locals)
        return {"stdout": output_buffer.getvalue(), "result": sandbox_locals.get("result")}
    except Exception as e:
        return {"stdout": output_buffer.getvalue(), "error": f"{type(e).__name__}: {e}"}


def _web_search_impl(query: str, max_results: int = 5) -> list[dict]:
    """
    Free, no-API-key web search via ddgs (a metasearch aggregator, not an
    official/guaranteed-uptime API). Treat a failure here as "unavailable
    right now" — not a sign that anything else is broken — since this
    depends on an unofficial scraping-style backend that can be flaky or
    rate-limited independent of anything in this codebase. Untested against
    a live network call in the environment this was built in (blocked
    there) — confirm it actually returns results on your own machine.
    """
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=max_results)
        return [
            {"title": r.get("title"), "snippet": r.get("body"), "url": r.get("href")}
            for r in results
        ]
    except Exception as e:
        return [{"error": f"Web search unavailable: {type(e).__name__}: {e}"}]


def make_agent_tools(
    vectorstore_dir: str | Path,
    financials_db_path: str | Path,
    embedding_function: EmbeddingFunction | None = None,
    scratchpad: dict | None = None,
) -> dict[str, callable]:
    """
    Builds the tool set for one session's stores.

    embedding_function is injectable (same pattern as vectorstore.py) so
    this can be smoke-tested offline with a fake embedder — pass None to
    use real Mistral embeddings in production.

    scratchpad is a dict the orchestrator writes every tool call's result
    into (keyed by call id) — passed in here (rather than created fresh)
    so run_python sees the SAME dict the orchestrator is populating, not a
    private copy. Pass None to get a fresh, empty one (fine for a session
    where run_python won't be used, e.g. most existing tests).
    """
    if scratchpad is None:
        scratchpad = {}
    collection = get_collection(vectorstore_dir, embedding_function=embedding_function)

    def list_companies() -> list[str]:
        """Lists every company currently ingested, by exact stored name.

        Call this first when the user names a company informally
        ("tatasteel", "L&T Ltd") to resolve it to the exact name the other
        tools expect, instead of guessing a spelling.
        """
        return financials_db.list_companies(financials_db_path)

    def list_available_metrics(company: str) -> list[str]:
        """Lists every financial metric available for one company, by exact name.

        Call this before get_financial_series/compute_trend if you're not
        sure a metric (e.g. "geography_revenue:india") exists for this
        company — avoids guessing a metric name that returns no data.
        """
        return financials_db.list_metrics(financials_db_path, company)

    def _missing_data_error(company: str, metric: str, start_year: int | None, end_year: int | None) -> dict:
        """
        Builds an error message that distinguishes two very different
        problems a caller needs to react to differently: the metric/company
        combination genuinely has no data at all, vs. it has data but not
        within the specific year range that was requested. A real run
        showed this distinction matters a lot in practice — a generic
        "no data found" message for the second case sent the agent off
        guessing different metric NAMES for several tool calls, when the
        actual fix was just to drop the year bounds, since every ingested
        report so far covers a single fiscal year.
        """
        if start_year is not None or end_year is not None:
            unbounded = financials_db.get_financial_series(financials_db_path, company, metric)
            if unbounded:
                available_years = [year for year, _ in unbounded]
                return {"error": f"No data for company={company!r}, metric={metric!r} in fiscal years "
                                  f"{start_year}-{end_year} specifically. This metric DOES exist for "
                                  f"this company — data is available for fiscal year(s) {available_years}. "
                                  f"Call again without start_year/end_year, or with a range that includes "
                                  f"those years — the metric name is correct, don't try a different one."}
        return {"error": f"No data found for company={company!r}, metric={metric!r}. "
                          f"Try list_companies() and list_available_metrics() to check exact names."}

    def get_financial_series(
        company: str,
        metric: str,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> dict:
        """Returns one company's raw values for one metric across fiscal years,
        together with the unit those values are actually denominated in.

        Use compute_trend instead if you need growth rates — don't compute
        them yourself from this raw series. Leave start_year/end_year unset
        on your first call for a given metric — only add them afterward if
        you need to narrow down results you've already seen.

        ALWAYS state the value in the exact unit given here — never convert
        it to a different unit (crore/million/billion/etc.) yourself.
        """
        series = financials_db.get_financial_series(financials_db_path, company, metric, start_year, end_year)
        if not series:
            return _missing_data_error(company, metric, start_year, end_year)
        return {
            "company": company,
            "metric": metric,
            "unit": financials_db.get_metric_unit(financials_db_path, company, metric),
            "series": [{"fiscal_year": year, "value": value} for year, value in series],
        }

    def compute_trend(
        company: str,
        metric: str,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> dict:
        """Computes year-over-year growth (%) and CAGR (%) for one company's metric.

        This is the tool to use for "analyze the trend of X" — the growth
        numbers are computed here with plain arithmetic on stored data, not
        estimated. A null yoy_growth_pct for a given year means there was no
        prior-year value to compare against (or the prior value was zero).
        A null cagr means fewer than 2 data points, a zero-year span, or a
        non-positive starting value — all cases where CAGR is undefined,
        not zero. Leave start_year/end_year unset on your first call — only
        add them afterward if you need to narrow down years you've already seen.

        ALWAYS state series values in the exact unit given here — never
        convert to a different unit yourself. CAGR/growth are already
        percentages and need no unit conversion at all.
        """
        series = financials_db.get_financial_series(financials_db_path, company, metric, start_year, end_year)
        if not series:
            return _missing_data_error(company, metric, start_year, end_year)
        yoy = financials_db.compute_yoy_growth(series)
        cagr = financials_db.compute_cagr(series)
        return {
            "company": company,
            "metric": metric,
            "unit": financials_db.get_metric_unit(financials_db_path, company, metric),
            "series": [{"fiscal_year": year, "value": value} for year, value in series],
            "yoy_growth_pct": [{"fiscal_year": year, "growth_pct": growth} for year, growth in yoy],
            "cagr_pct": cagr,
        }

    def compare_companies(
        companies: list[str],
        metric: str,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> dict:
        """Computes the same trend (YoY growth + CAGR) for the same metric across several companies.

        Use this for "which of X and Y grew faster" style questions instead
        of calling compute_trend separately and comparing yourself — this
        guarantees both companies were queried over the same year range.
        """
        return {company: compute_trend(company, metric, start_year, end_year) for company in companies}

    def vector_search(
        query: str,
        company: str | None = None,
        fiscal_year: int | None = None,
    ) -> list[dict]:
        """Semantic search over qualitative report text (risk factors, MD&A, segment commentary).

        Leave company=None to search across every ingested company at once
        (e.g. for "which companies mention supply chain risk"). Set it to
        scope the search to one company once you know which one you need.
        """
        return search(collection, query, company=company, fiscal_year=fiscal_year, n_results=5)

    def run_python(code: str) -> dict:
        """Executes a short Python snippet for a custom calculation none of the other tools cover
        (e.g. a 3-year rolling average, excluding an outlier year, a custom ratio).

        Every result from a tool call earlier in this turn is available inside your code as
        `scratchpad`, keyed by call id (e.g. scratchpad['call_2_compute_trend']) — reference
        numbers from there instead of re-typing them from memory, since re-typing is exactly how
        transcription errors happen. Only the math/statistics modules and basic builtins are
        available — no imports, no file/network access. Assign your final answer to a variable
        named `result`; anything you print is returned too.
        """
        return _execute_sandboxed_python(code, scratchpad)

    def web_search(query: str) -> list[dict]:
        """Searches the web for information NOT in the ingested reports — current stock price,
        industry benchmarks, recent news, competitor context not covered by an ingested report.

        Free, no API key — but unofficial and not guaranteed-uptime. A failure here means try
        again or answer without it, not that something else is broken.
        """
        return _web_search_impl(query)

    return {
        "list_companies": list_companies,
        "list_available_metrics": list_available_metrics,
        "get_financial_series": get_financial_series,
        "compute_trend": compute_trend,
        "compare_companies": compare_companies,
        "vector_search": vector_search,
        "run_python": run_python,
        "web_search": web_search,
    }


def _run_smoke_test() -> None:
    import tempfile

    from stores.financials_db import FinancialFact, upsert_facts_bulk
    from stores.vectorstore import TextChunk, add_chunks, _FakeEmbeddingFunction

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "financials.sqlite"
        chroma_dir = Path(tmp) / "chroma_db"

        # Same known revenue figures used in financials_db.py's own smoke test,
        # so the CAGR/YoY numbers here are already independently verified.
        upsert_facts_bulk(db_path, [
            FinancialFact("Tata Steel", 2021, "revenue", 156294, "INR crore"),
            FinancialFact("Tata Steel", 2022, "revenue", 243959, "INR crore"),
            FinancialFact("Tata Steel", 2023, "revenue", 227972, "INR crore"),
            FinancialFact("Tata Steel", 2024, "revenue", 218577, "INR crore"),
            FinancialFact("L&T", 2021, "revenue", 143633, "INR crore"),
            FinancialFact("L&T", 2024, "revenue", 222866, "INR crore"),
        ])

        tools = make_agent_tools(chroma_dir, db_path, embedding_function=_FakeEmbeddingFunction(), scratchpad={})
        collection = get_collection(chroma_dir, embedding_function=_FakeEmbeddingFunction())
        add_chunks(collection, [
            TextChunk("Tata Steel", 2024, page=12, section="risk_factors",
                      text="Raw material price volatility is a key risk for Tata Steel."),
            TextChunk("L&T", 2024, page=8, section="risk_factors",
                      text="Execution delays on infrastructure projects are a risk for L&T."),
        ])

        companies = tools["list_companies"]()
        assert companies == ["L&T", "Tata Steel"], f"unexpected companies: {companies}"
        print(f"list_companies: {companies}")

        metrics = tools["list_available_metrics"]("Tata Steel")
        assert metrics == ["revenue"], f"unexpected metrics: {metrics}"
        print(f"list_available_metrics('Tata Steel'): {metrics}")

        series_result = tools["get_financial_series"]("Tata Steel", "revenue")
        assert series_result["series"][0] == {"fiscal_year": 2021, "value": 156294}
        assert series_result["unit"] == "INR crore", \
            "the unit must be included — its absence is what caused the agent to fabricate one ('million')"
        print(f"get_financial_series: {len(series_result['series'])} points returned, "
              f"unit={series_result['unit']!r}")

        missing_result = tools["get_financial_series"]("Tata Steel", "nonexistent_metric")
        assert "error" in missing_result, "a missing metric should return a helpful error, not crash"
        print(f"get_financial_series (missing metric): {missing_result['error'][:60]}...")

        # --- REGRESSION: a real run showed a metric that DOES exist but not in the
        # requested year range produced the same generic "no data found" message as a
        # genuinely wrong metric name — the agent then wasted 6 tool calls guessing
        # different metric NAMES when the real fix was just dropping the year bounds.
        # (fixture data covers 2021-2024, so 2018-2020 is genuinely outside it)
        out_of_range_result = tools["get_financial_series"]("Tata Steel", "revenue", 2018, 2020)
        assert "error" in out_of_range_result
        assert "DOES exist" in out_of_range_result["error"], \
            "an out-of-range query for an otherwise-valid metric must say so explicitly, not look identical to a wrong metric name"
        assert "2021" in out_of_range_result["error"]
        print(f"get_financial_series (valid metric, wrong year range): "
              f"{out_of_range_result['error'][:90]}...")

        out_of_range_trend = tools["compute_trend"]("Tata Steel", "revenue", 2018, 2020)
        assert "DOES exist" in out_of_range_trend["error"]
        print("compute_trend (valid metric, wrong year range): same precise error, not a generic 'not found'")

        trend = tools["compute_trend"]("Tata Steel", "revenue")
        assert trend["cagr_pct"] is not None
        assert abs(trend["cagr_pct"] - 11.8) < 0.1, f"CAGR mismatch: {trend['cagr_pct']}"
        assert trend["unit"] == "INR crore"
        print(f"compute_trend('Tata Steel', 'revenue'): CAGR={trend['cagr_pct']:.1f}%, unit={trend['unit']!r}")

        comparison = tools["compare_companies"](["Tata Steel", "L&T"], "revenue")
        assert set(comparison.keys()) == {"Tata Steel", "L&T"}
        assert comparison["L&T"]["cagr_pct"] is not None
        print(f"compare_companies: Tata Steel CAGR={comparison['Tata Steel']['cagr_pct']:.1f}%, "
              f"L&T CAGR={comparison['L&T']['cagr_pct']:.1f}%")

        tata_hits = tools["vector_search"]("risk factors", company="Tata Steel")
        assert len(tata_hits) == 1 and tata_hits[0]["company"] == "Tata Steel"
        all_hits = tools["vector_search"]("risk factors")
        assert len(all_hits) == 2, f"expected 2 cross-company hits, got {len(all_hits)}"
        print(f"vector_search: scoped -> {len(tata_hits)} hit, unscoped -> {len(all_hits)} hits")

        # --- run_python: normal use over scratchpad data (simulating what the
        # orchestrator would have already written there from an earlier tool call) ---
        seeded_scratchpad = {"call_1_compute_trend": trend}
        tools_with_data = make_agent_tools(chroma_dir, db_path, embedding_function=_FakeEmbeddingFunction(),
                                            scratchpad=seeded_scratchpad)
        python_result = tools_with_data["run_python"](
            "result = scratchpad['call_1_compute_trend']['cagr_pct'] * 2"
        )
        assert python_result.get("error") is None, f"unexpected error: {python_result}"
        assert abs(python_result["result"] - trend["cagr_pct"] * 2) < 0.01
        print(f"run_python: computed over scratchpad data correctly -> {python_result['result']:.2f}")

        # --- run_python: the same escape/blocklist protections as the module-level tests,
        # exercised through the actual tool interface an agent would call ---
        blocked = tools_with_data["run_python"]("import os; result = os.getcwd()")
        assert "error" in blocked and "Blocked" in blocked["error"]
        print("run_python (via tool interface): disallowed patterns are still blocked")

        # --- web_search: can't hit a live backend in this sandboxed test environment
        # (same as every other live-network step in this project) — this just confirms
        # the tool is wired up and fails cleanly rather than raising, so a real but down
        # or blocked search doesn't crash the agent loop mid-conversation ---
        search_result = tools["web_search"]("Tata Steel current share price")
        assert isinstance(search_result, list) and len(search_result) >= 1
        print(f"web_search: wired up and returns cleanly (live result quality needs checking "
              f"with real network access): {str(search_result)[:100]}")

        # --- REGRESSION GUARD: every tool's signature must have real type
        # objects as annotations, not strings. A real run once broke every
        # single tool call with a TypeError, traced to `from __future__
        # import annotations` making every annotation a string at runtime —
        # google-genai's invoke_function_from_dict_args (used here purely
        # as a rigorous offline signature-correctness check, regardless of
        # which provider's orchestrator is in production) does
        # isinstance(value, annotation) using the real annotation object,
        # and isinstance(x, 'str') raises TypeError. The same underlying
        # correctness matters for _build_tool_schema's schema generation in
        # orchestrator.py too: a string annotation there would silently
        # produce wrong or missing JSON schema types, not just crash loudly
        # like this does. This exercises that exact code path directly so
        # the bug can't silently come back (e.g. if this file's missing
        # future-annotations import is later "fixed" for stylistic
        # consistency with the rest of the project — don't do that here).
        from google.genai._extra_utils import invoke_function_from_dict_args

        afc_test_args = {
            "list_companies": {},
            "list_available_metrics": {"company": "Tata Steel"},
            "get_financial_series": {"company": "Tata Steel", "metric": "revenue"},
            "compute_trend": {"company": "Tata Steel", "metric": "revenue"},
            "compare_companies": {"companies": ["Tata Steel", "L&T"], "metric": "revenue"},
            "vector_search": {"query": "risk factors"},
            "run_python": {"code": "result = 1 + 1"},
            "web_search": {"query": "test query"},
        }
        for tool_name, call_args in afc_test_args.items():
            invoke_function_from_dict_args(call_args, tools[tool_name])
        print(f"signature regression guard: all {len(afc_test_args)} tools have real (non-string) "
              f"type annotations, verified via a rigorous real argument-conversion path")

    print("\nAll tools.py smoke tests passed.")


if __name__ == "__main__":
    _run_smoke_test()