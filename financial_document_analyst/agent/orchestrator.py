"""
agent/orchestrator.py

Wires agent/tools.py into a manual Mistral tool-calling loop.

Unlike the earlier Gemini-based version of this file, Mistral's SDK has no
"automatic function calling" convenience — chat.complete() accepts
tools=[...] and returns tool_calls for the caller to execute, but nothing
in the SDK executes them or continues the conversation on its own
(confirmed by inspecting the client: c.agents / c.beta.conversations are
Mistral's own hosted-agent platform, a different paradigm requiring
server-registered agents, not a helper for local Python callables). This
file implements the loop itself: send messages -> if the model requests
tool calls, execute them and feed results back as ToolMessage entries ->
repeat until the model returns a final text answer, or
MAX_TOOL_ITERATIONS is reached.

Model choice: open-mistral-nemo, the same one used for extraction —
Mistral's reasoning-capable models (Magistral, Medium) are Premier-tier
and return a hard 0 requests/minute limit on the free tier (confirmed
empirically earlier in this project). This means the agent has no
explicit "thinking mode" the way Gemini 3 Flash had — it's a capable
general model doing tool-calling, not a dedicated reasoning specialist.
reasoning_effort is therefore not set here, same reasoning as extract.py:
sending it to a non-reasoning-tier model risks a 400 rather than being
silently ignored.

AgentSession holds the conversation as a plain list of typed Mistral
message objects, since the client has no stateful chat-session object
equivalent to Gemini's `Chat` — this is the direct replacement for that,
and the thing ask() rolls back on a failed attempt (same reasoning as the
Gemini version's history-rollback fix: a failure partway through the tool
loop could otherwise leave a dangling, incomplete turn that a naive retry
would build on top of, confusing the model).
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from chromadb.api.types import EmbeddingFunction
from google.genai._automatic_function_calling_util import parse_function_declaration_json_schema
from mistralai.client import Mistral
from mistralai.client.models.assistantmessage import AssistantMessage
from mistralai.client.models.function import Function
from mistralai.client.models.systemmessage import SystemMessage
from mistralai.client.models.tool import Tool
from mistralai.client.models.toolmessage import ToolMessage
from mistralai.client.models.usermessage import UserMessage

from agent.tools import make_agent_tools
from agent.verifier import verify_answer

MODEL_NAME = "open-mistral-nemo"
MAX_RETRIES = 5
MAX_TOOL_ITERATIONS = 10

SYSTEM_INSTRUCTION = """You are a financial analyst agent that answers questions about companies \
using their annual reports.

You have tools to look up ingested companies, their available financial metrics, raw financial \
data, computed trends (YoY growth, CAGR), and qualitative report text (risk factors, MD&A, \
segment commentary).

Rules:
- The user will refer to companies informally (e.g. "tatasteel", "L&T"). Call list_companies() \
  and match against the exact names returned before calling any other tool with a company name. \
  Never guess a company's exact stored name.
- If a metric lookup returns an error, call list_available_metrics() for that company before \
  trying a different metric name yourself — don't guess a metric name that might not exist.
- Call get_financial_series() and compute_trend() WITHOUT start_year/end_year the first time for \
  a given company/metric. Only add a year range afterward, once you've seen what years are \
  actually available — a metric can exist but simply not cover the years you guessed, which \
  looks identical to a wrong metric name if you narrow the range before checking.
- For any question involving growth, change over time, or "trend", use compute_trend() or \
  compare_companies() — never estimate growth by eyeballing get_financial_series() output \
  yourself.
- When you use vector_search() results in your answer, cite the company and page number the \
  claim came from.
- If a tool returns an "error" field, treat it as information to act on (resolve a name, try a \
  different metric), not as a reason to give up on the question.
- Be precise and quantitative. State the actual numbers, not just directional language like \
  "grew significantly" — this is for real financial analysis, not casual conversation.
- When you state a numeric value that came from a tool result, copy the digits exactly as given \
  in the tool's JSON output — do not round, reformat, or rewrite the number from memory. If you \
  want to present it with comma grouping for readability, group the exact same digits; never \
  change how many digits it has.
- Every financial figure comes with a "unit" field (e.g. "INR crore"). State it exactly as given \
  — never convert it to a different unit (crore to million, million to billion, etc.) yourself. \
  You do not reliably get this conversion right; stating the original unit correctly is far \
  better than a wrong "helpful" conversion.
"""


# --- duck-typed Mistral error helpers -----------------------------------------
# Same detection logic as ingest/extract.py's _call_mistral_with_retry — see
# that file's docstring for the full evidence (real 429/400 bodies, the
# isinstance(e, SDKError) mismatch that forced duck-typing). Duplicated here
# rather than shared, per this project's existing precedent
# (stores/vectorstore.py did the same for its embeddings retry logic).

def _get_http_status(e: BaseException) -> int | None:
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
    if _get_http_status(e) != 400:
        return False
    body_text = _get_http_body_text(e)
    try:
        body = json.loads(body_text) if body_text else {}
    except (json.JSONDecodeError, TypeError):
        return False
    return body.get("code") == "3210" or body.get("type") == "invalid_request_prompt"


@dataclass
class AgentSession:
    """One conversation's state — the replacement for Gemini's stateful Chat
    object, since Mistral's client has nothing equivalent. `messages` is a
    plain list of typed Mistral message objects (SystemMessage, UserMessage,
    AssistantMessage, ToolMessage)."""
    client: Mistral
    tools: dict
    tool_schemas: list = field(default_factory=list)
    messages: list = field(default_factory=list)


def _build_tool_schema(name: str, fn) -> Tool:
    """
    Builds a Mistral Tool/Function schema from a Python callable by reusing
    google-genai's own function-schema introspection
    (parse_function_declaration_json_schema) purely as an offline JSON-schema
    generator — no Gemini API call happens anywhere in this file. Verified
    to produce a standard JSON schema (type/properties/required) that
    Mistral's Function.parameters accepts directly.
    """
    decl = parse_function_declaration_json_schema(fn, None)
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=decl.description or "",
            parameters=decl.parameters_json_schema or {"type": "object", "properties": {}},
        ),
    )


def build_agent(
    vectorstore_dir: str | Path,
    financials_db_path: str | Path,
    mistral_api_key: str | None = None,
    embedding_function: EmbeddingFunction | None = None,
) -> AgentSession:
    """Builds a fresh conversation session wired with the tool set for one session's stores."""
    mistral_api_key = mistral_api_key or os.environ.get("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise ValueError("Set MISTRAL_API_KEY (or pass mistral_api_key=) before calling build_agent()")

    client = Mistral(api_key=mistral_api_key)
    tools = make_agent_tools(vectorstore_dir, financials_db_path, embedding_function=embedding_function)
    tool_schemas = [_build_tool_schema(name, fn) for name, fn in tools.items()]

    return AgentSession(
        client=client,
        tools=tools,
        tool_schemas=tool_schemas,
        messages=[SystemMessage(content=SYSTEM_INSTRUCTION)],
    )


def _execute_tool_call(tool_call, tools: dict) -> str:
    """
    Executes one tool call and returns a JSON string result — always
    returns something serializable, even on failure, so a bad tool call
    becomes information the model can react to (per tools.py's own "error"
    dict pattern) rather than crashing the whole session.
    """
    name = tool_call.function.name
    raw_args = tool_call.function.arguments
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"error": f"Could not parse arguments for {name}: {e}"})

    fn = tools.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})

    try:
        result = fn(**args)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"{name} raised {type(e).__name__}: {e}"})


def _collect_tool_results(messages: list) -> list[dict]:
    """
    Extracts and JSON-parses every ToolMessage's content from a slice of
    conversation history — used by ask() to gather this turn's tool
    results for verify_answer(). Anything that fails to parse (shouldn't
    happen, since _execute_tool_call always returns valid JSON, but never
    trust that blindly) is skipped rather than raised.
    """
    results = []
    for m in messages:
        if isinstance(m, ToolMessage):
            try:
                results.append(json.loads(m.content))
            except (json.JSONDecodeError, TypeError):
                pass
    return results


def _call_mistral_with_retry(client: Mistral, messages: list, tool_schemas: list):
    """
    Retries on 429, 5xx, and the "too many tokens" 400 — identical
    conditions and backoff strategy to ingest/extract.py's
    _call_mistral_with_retry. See that file's docstring for the full
    evidence behind the "too many tokens is transient, retry in place"
    reasoning, and for the "hard 0 req/min limit fails fast" reasoning.
    """
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return client.chat.complete(
                model=MODEL_NAME, messages=messages, tools=tool_schemas, tool_choice="auto",
            )
        except BaseException as e:
            if not isinstance(e, Exception):
                raise
            status = _get_http_status(e)
            if status is None:
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
                    f"on your account. Check https://admin.mistral.ai/plateforme/limits."
                ) from e

            if status == 429 or (500 <= status < 600) or too_many_tokens:
                if retry_after is not None:
                    wait = float(retry_after)
                    wait_source = "server Retry-After"
                elif too_many_tokens:
                    wait = 20 * (attempt + 1)
                    wait_source = "assumed per-minute token budget"
                elif status == 429:
                    wait = 60
                    wait_source = "per-minute rate limit, no Retry-After given"
                else:
                    wait = min(2 ** attempt, 60)
                    wait_source = "exponential backoff (server error)"

                print(f"  Mistral call failed ({status}, {type(e).__name__}). "
                      f"limit_per_min={limit_per_min} remaining_per_min={remaining_per_min}. "
                      f"Waiting {wait:.0f}s ({wait_source}), attempt {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(wait)
                continue
            raise

    raise RuntimeError(f"Mistral call failed after {MAX_RETRIES} retries") from last_error


def ask(session: AgentSession, message: str) -> str:
    """
    Sends one message and runs the tool-calling loop until the model
    returns a final text answer (no more tool calls) or
    MAX_TOOL_ITERATIONS is reached.

    Rolls session.messages back to its pre-call length on any failure
    before it propagates — a failure partway through the tool loop could
    otherwise leave a dangling, incomplete turn in history that the next
    ask() call would build on top of, confusing the model. This was a real
    bug discovered the hard way in the earlier Gemini version; built in
    here from the start instead.

    Before returning, the final answer is checked against every tool
    result from this turn via verifier.verify_answer() — built after a
    real run showed open-mistral-nemo can retrieve a correct number via a
    tool call and then garble it when writing the prose answer (232139.94
    became "23,213.99" in one real case). A verification failure doesn't
    block the answer — there's no dedicated reasoning model available on
    the free tier to ask for a reliable self-correction, and a blocked
    answer serves the user less than a flagged one — but it's appended as
    a visible warning so a wrong number is never presented with silent
    confidence.

    Each tool call is printed as it happens (name, arguments, and whether
    it errored) — this was invisible before, which made an agent that gets
    stuck calling tools in a loop (a real, observed failure on a
    multi-company comparison question) completely opaque to debug.
    """
    snapshot_len = len(session.messages)
    session.messages.append(UserMessage(content=message))

    try:
        for iteration in range(MAX_TOOL_ITERATIONS):
            response = _call_mistral_with_retry(session.client, session.messages, session.tool_schemas)
            choice = response.choices[0]
            assistant_message = choice.message

            session.messages.append(AssistantMessage(
                content=assistant_message.content or "",
                tool_calls=assistant_message.tool_calls,
            ))

            if not assistant_message.tool_calls:
                final_text = assistant_message.content or ""
                tool_results = _collect_tool_results(session.messages[snapshot_len:])
                verification = verify_answer(final_text, tool_results)
                if not verification.verified:
                    unmatched_str = ", ".join(f"{n:,.2f}" for n in verification.unmatched_numbers)
                    print(f"  [verifier] unmatched figures in the answer: {unmatched_str}")
                    final_text += (
                        f"\n\n⚠️ Note: this answer mentions figures ({unmatched_str}) that don't "
                        f"match any value returned by the tools used to answer it — please "
                        f"double-check these directly before relying on them."
                    )
                return final_text

            for tool_call in assistant_message.tool_calls:
                print(f"  [tool call {iteration + 1}] {tool_call.function.name}({tool_call.function.arguments})")
                result_json = _execute_tool_call(tool_call, session.tools)
                is_error = '"error"' in result_json
                print(f"  [tool result] {'ERROR: ' if is_error else ''}{result_json[:200]}"
                      f"{'...' if len(result_json) > 200 else ''}")
                session.messages.append(ToolMessage(
                    tool_call_id=tool_call.id, name=tool_call.function.name, content=result_json,
                ))

        raise RuntimeError(f"Exceeded {MAX_TOOL_ITERATIONS} tool-calling iterations without a final answer")
    except Exception:
        del session.messages[snapshot_len:]
        raise


def _run_smoke_test() -> None:
    """
    Verifies tool-schema generation against the real tool set, the
    tool-execution wrapper's error handling, the full multi-round
    tool-calling loop (mocked client, since this sandbox cannot reach
    Mistral's API), retry/error-handling logic, and the history-rollback
    behavior. The actual round trip to Mistral is untested here — needs a
    real MISTRAL_API_KEY tested on your end, same as every other live-API
    step in this project.
    """
    import tempfile
    from unittest.mock import MagicMock, patch

    import httpx
    from mistralai.client.errors import SDKError
    from mistralai.client.models.functioncall import FunctionCall
    from mistralai.client.models.toolcall import ToolCall
    from stores.vectorstore import _FakeEmbeddingFunction

    # --- _build_tool_schema: real schema generation against a real tool ---
    with tempfile.TemporaryDirectory() as tmp:
        tools = make_agent_tools(Path(tmp) / "chroma", Path(tmp) / "db.sqlite",
                                  embedding_function=_FakeEmbeddingFunction())
        schema = _build_tool_schema("compute_trend", tools["compute_trend"])
        assert schema.type == "function"
        assert schema.function.name == "compute_trend"
        assert "company" in schema.function.parameters["properties"]
        assert "company" in schema.function.parameters["required"]
        assert "start_year" not in schema.function.parameters["required"]
    print("_build_tool_schema: produces a valid Mistral Tool/Function schema from a real tool")

    # --- build_agent: constructs cleanly with all 6 tools, system message present ---
    with tempfile.TemporaryDirectory() as tmp:
        session = build_agent(
            vectorstore_dir=Path(tmp) / "chroma", financials_db_path=Path(tmp) / "financials.sqlite",
            mistral_api_key="fake-key-for-smoke-test", embedding_function=_FakeEmbeddingFunction(),
        )
        assert len(session.tool_schemas) == 6
        assert len(session.messages) == 1
        assert isinstance(session.messages[0], SystemMessage)
    print("build_agent: constructs a session wired with all 6 tools and a system message")

    # --- _execute_tool_call: real tool, JSON-string args, malformed args, unknown tool, tool that raises ---
    with tempfile.TemporaryDirectory() as tmp:
        from stores.financials_db import FinancialFact, upsert_facts_bulk
        db_path = Path(tmp) / "financials.sqlite"
        upsert_facts_bulk(db_path, [FinancialFact("Tata Steel", 2024, "revenue", 218577.0, "INR crore")])
        tools = make_agent_tools(Path(tmp) / "chroma", db_path, embedding_function=_FakeEmbeddingFunction())

        def fake_tool_call(name, arguments):
            return ToolCall(id="call_1", type="function", function=FunctionCall(name=name, arguments=arguments))

        result = _execute_tool_call(fake_tool_call("list_available_metrics", '{"company": "Tata Steel"}'), tools)
        assert json.loads(result) == ["revenue"]

        result = _execute_tool_call(fake_tool_call("get_financial_series", '{"company": "Tata Steel", "metric": "revenue"}'), tools)
        assert json.loads(result)["series"][0]["value"] == 218577.0

        bad_json_result = _execute_tool_call(fake_tool_call("list_available_metrics", "{not valid json"), tools)
        assert "error" in json.loads(bad_json_result)

        unknown_tool_result = _execute_tool_call(fake_tool_call("nonexistent_tool", "{}"), tools)
        assert "error" in json.loads(unknown_tool_result)

        # a tool that raises internally (wrong arg name entirely) must not crash the loop
        raising_result = _execute_tool_call(fake_tool_call("get_financial_series", '{"totally_wrong_arg": 1}'), tools)
        assert "error" in json.loads(raising_result)
    print("_execute_tool_call: handles real calls, malformed JSON, unknown tools, and raising tools "
          "without crashing — always returns a serializable result")

    # --- ask(): full multi-round tool-calling loop against a mocked client ---
    def make_tool_call_response(tool_name, tool_args_json, call_id="call_1"):
        tc = ToolCall(id=call_id, type="function", function=FunctionCall(name=tool_name, arguments=tool_args_json))
        response = MagicMock()
        response.choices[0].message.content = None
        response.choices[0].message.tool_calls = [tc]
        return response

    def make_final_response(text):
        response = MagicMock()
        response.choices[0].message.content = text
        response.choices[0].message.tool_calls = None
        return response

    with tempfile.TemporaryDirectory() as tmp:
        from stores.financials_db import FinancialFact, upsert_facts_bulk
        db_path = Path(tmp) / "financials.sqlite"
        upsert_facts_bulk(db_path, [
            FinancialFact("Tata Steel", 2021, "revenue", 156294.0, "INR crore"),
            FinancialFact("Tata Steel", 2024, "revenue", 218577.0, "INR crore"),
        ])
        session = build_agent(Path(tmp) / "chroma", db_path, mistral_api_key="fake-key",
                               embedding_function=_FakeEmbeddingFunction())

        final_text = "Tata Steel's revenue grew at roughly an 11.8% CAGR from FY2021 to FY2024."
        session.client.chat.complete = MagicMock(side_effect=[
            make_tool_call_response("list_companies", "{}"),
            make_tool_call_response("compute_trend", '{"company": "Tata Steel", "metric": "revenue"}'),
            make_final_response(final_text),
        ])

        result = ask(session, "Analyze Tata Steel's revenue trend")

        assert result == final_text
        assert session.client.chat.complete.call_count == 3
        # system + user + (assistant+tool)x2 + final assistant = 7 messages
        assert len(session.messages) == 7, f"expected 7 messages in history, got {len(session.messages)}"
        assert isinstance(session.messages[1], UserMessage)
        assert isinstance(session.messages[2], AssistantMessage)
        assert isinstance(session.messages[3], ToolMessage)
        assert isinstance(session.messages[-1], AssistantMessage)
        assert session.messages[-1].content == final_text
    print("ask(): full multi-round tool-calling loop (2 tool calls then final answer) works end to end, "
          "history correctly interleaves assistant/tool messages")

    # --- ask(): a WRONG number in the final answer gets the verifier's warning appended ---
    # This is the actual integration point that matters — verifier.py has its own dedicated
    # tests, but this proves ask() actually wires it in and appends the warning to what's returned.
    with tempfile.TemporaryDirectory() as tmp:
        from stores.financials_db import FinancialFact, upsert_facts_bulk
        db_path2 = Path(tmp) / "financials.sqlite"
        upsert_facts_bulk(db_path2, [FinancialFact("Tata Steel", 2024, "revenue", 232139.94, "INR crore")])
        session_wrong = build_agent(Path(tmp) / "chroma", db_path2, mistral_api_key="fake-key",
                                     embedding_function=_FakeEmbeddingFunction())

        garbled_final_text = "Tata Steel's revenue was approximately ₹23,213.99 crores."
        session_wrong.client.chat.complete = MagicMock(side_effect=[
            make_tool_call_response("get_financial_series", '{"company": "Tata Steel", "metric": "revenue"}'),
            make_final_response(garbled_final_text),
        ])

        wrong_result = ask(session_wrong, "What was Tata Steel's revenue?")
        assert garbled_final_text in wrong_result, "the original (wrong) answer text should still be returned, not hidden"
        assert "⚠️ Note" in wrong_result, "a garbled number must get the verifier's warning appended"
        assert "23,213.99" in wrong_result
    print("ask(): a garbled number in the final answer gets the verifier's warning appended — "
          "reproduces and catches the exact real Tata Steel bug end to end")

    # --- ask(): a follow-up question in the same session builds on existing history ---
    with tempfile.TemporaryDirectory() as tmp:
        session2 = build_agent(Path(tmp) / "chroma", Path(tmp) / "db.sqlite", mistral_api_key="fake-key",
                                embedding_function=_FakeEmbeddingFunction())
        session2.client.chat.complete = MagicMock(side_effect=[
            make_final_response("First answer."),
            make_final_response("Second answer, building on the first."),
        ])
        first = ask(session2, "First question")
        second = ask(session2, "Second question")
        assert first == "First answer." and second == "Second answer, building on the first."
        # system + (user+assistant) + (user+assistant) = 5 messages, nothing lost between calls
        assert len(session2.messages) == 5
    print("ask(): a second question in the same session correctly builds on prior history")

    # --- ask(): retries on 429, respects headers, same as extract.py ---
    fake_429_response = httpx.Response(
        429, headers={"retry-after": "2", "x-ratelimit-limit-req-minute": "60",
                      "x-ratelimit-remaining-req-minute": "0"},
    )
    rate_limit_error = SDKError("Rate limit exceeded", raw_response=fake_429_response, body="{}")

    with tempfile.TemporaryDirectory() as tmp:
        session3 = build_agent(Path(tmp) / "chroma", Path(tmp) / "db.sqlite", mistral_api_key="fake-key",
                                embedding_function=_FakeEmbeddingFunction())
        session3.client.chat.complete = MagicMock(side_effect=[rate_limit_error, make_final_response("OK.")])

        with patch("time.sleep") as mock_sleep:
            result3 = ask(session3, "some question")

        assert result3 == "OK."
        mock_sleep.assert_called_once_with(2.0)
    print("ask(): retries on 429 respecting Retry-After, same behavior as extract.py")

    # --- ask(): a hard 0 req/minute limit fails fast ---
    zero_quota_response = httpx.Response(
        429, headers={"x-ratelimit-limit-req-minute": "0", "x-ratelimit-remaining-req-minute": "0"},
    )
    zero_quota_error = SDKError("Rate limit exceeded", raw_response=zero_quota_response, body="{}")

    with tempfile.TemporaryDirectory() as tmp:
        session4 = build_agent(Path(tmp) / "chroma", Path(tmp) / "db.sqlite", mistral_api_key="fake-key",
                                embedding_function=_FakeEmbeddingFunction())
        session4.client.chat.complete = MagicMock(side_effect=zero_quota_error)

        with patch("time.sleep") as mock_sleep_zero:
            try:
                ask(session4, "some question")
                raise AssertionError("expected RuntimeError for a hard 0 req/minute limit")
            except RuntimeError as e:
                assert "0 requests/minute" in str(e)
        mock_sleep_zero.assert_not_called()
    print("ask(): a hard 0 req/minute limit fails immediately, no wasted retries")

    # --- ask(): a failure partway through the tool loop rolls history back cleanly ---
    with tempfile.TemporaryDirectory() as tmp:
        session5 = build_agent(Path(tmp) / "chroma", Path(tmp) / "db.sqlite", mistral_api_key="fake-key",
                                embedding_function=_FakeEmbeddingFunction())
        pre_call_len = len(session5.messages)
        session5.client.chat.complete = MagicMock(side_effect=[
            make_tool_call_response("list_companies", "{}"),  # succeeds, appends 2 messages
            RuntimeError("Mistral call failed after 5 retries"),  # then the loop's next call fails outright
        ])
        try:
            ask(session5, "some question")
            raise AssertionError("expected the RuntimeError to propagate")
        except RuntimeError:
            pass
        assert len(session5.messages) == pre_call_len, \
            "a failed ask() call must roll history back to exactly its pre-call state, not leave a dangling turn"
    print("ask(): a failure partway through the tool loop rolls session.messages back to its pre-call state")

    # --- ask(): exceeding MAX_TOOL_ITERATIONS raises and rolls back, doesn't loop forever ---
    with tempfile.TemporaryDirectory() as tmp:
        session6 = build_agent(Path(tmp) / "chroma", Path(tmp) / "db.sqlite", mistral_api_key="fake-key",
                                embedding_function=_FakeEmbeddingFunction())
        pre_call_len6 = len(session6.messages)
        # the model never stops calling tools
        session6.client.chat.complete = MagicMock(
            side_effect=[make_tool_call_response("list_companies", "{}") for _ in range(MAX_TOOL_ITERATIONS + 2)]
        )
        try:
            ask(session6, "some question")
            raise AssertionError("expected a RuntimeError for exceeding MAX_TOOL_ITERATIONS")
        except RuntimeError as e:
            assert "iterations" in str(e)
        assert len(session6.messages) == pre_call_len6
        assert session6.client.chat.complete.call_count == MAX_TOOL_ITERATIONS
    print(f"ask(): a model that never stops calling tools is bounded at {MAX_TOOL_ITERATIONS} "
          f"iterations, then fails and rolls back cleanly")

    print("\nAll orchestrator.py smoke tests passed (real Mistral call untested — needs a live MISTRAL_API_KEY).")


if __name__ == "__main__":
    _run_smoke_test()