"""
agent/orchestrator_langgraph.py

Same capabilities as agent/orchestrator.py (tool-calling loop, structured
financial lookups, code execution, web search, visible planning, answer
verification, self-critique) but built on LangChain + LangGraph instead of
a hand-rolled loop against the raw Mistral SDK.

Reuses agent/tools.py's tool functions and agent/verifier.py's
verify_answer completely unchanged via StructuredTool.from_function — only
the orchestration layer (the loop itself, message handling, retries) is
different. Both files exist deliberately, side by side: understanding the
raw mechanics (orchestrator.py) is what let real Mistral-specific bugs
(the too-many-tokens retry-in-place discovery, the isinstance(e, SDKError)
mismatch) get found and fixed with precision in the first place. This file
demonstrates the same capability set on the framework most job postings
actually name.

Real, verified evidence this file's retry logic is built on, not assumed:
ChatMistralAI does NOT wrap the official mistralai SDK — it makes its own
raw httpx calls (confirmed by reading langchain_mistralai/chat_models.py
directly) and raises httpx.HTTPStatusError on any 4xx/5xx. Its own built-in
retry decorator (_create_retry_decorator in that same file) only retries
httpx.RequestError/httpx.StreamError — network-level failures — NOT
httpx.HTTPStatusError. That means a 429 from Mistral's free tier is NOT
retried by the framework at all without the wrapper below: confirmed by
reading the source, not assumed from documentation.

Graph shape:
    START -> agent -> [tools -> agent]* -> verify -> self_critique -> END
"tools" only runs when the model's last message has tool_calls; otherwise
"agent" routes straight to "verify". A MemorySaver checkpointer gives each
session a durable, thread_id-scoped message history — the same problem
app.py's original conversation-memory bug had, solved here by the
framework's own persistence primitive instead of a hand-rolled list.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Annotated, TypedDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import httpx
from chromadb.api.types import EmbeddingFunction
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent.tools import make_agent_tools
from agent.verifier import verify_answer

# Same free-tier finding as orchestrator.py / extract.py: Mistral's
# reasoning-tier models return a hard 0 requests/minute limit on the free
# "Experiment" tier — open-mistral-nemo is the one confirmed to have real
# free-tier access.
MODEL_NAME = "open-mistral-nemo"
MAX_TOOL_ITERATIONS = 10

SYSTEM_INSTRUCTION = """You are a financial analyst agent that answers questions about companies \
using their annual reports.

You have tools to look up ingested companies, their available financial metrics, raw financial \
data, computed trends (YoY growth, CAGR), qualitative report text (risk factors, MD&A, segment \
commentary), a Python sandbox for custom calculations, and web search for information outside \
the ingested reports.

Before your first tool call in response to a new question, briefly state your plan as your \
message content (1-3 short bullet points: which tools you expect to call and why) — state it \
even though you're also calling a tool in the same response. This is for the user's visibility \
into your reasoning, not a separate step — don't wait for a reply before acting on it.

How thorough to be depends on the question:
- A narrow, specific question (a single number, a single fact) — be economical: call only the \
  tools needed to answer exactly that, nothing more.
- A broad, open-ended question ("how is the company doing", "give me an investment thesis", \
  "should I invest") — be thorough, like a real analyst would: proactively gather multiple \
  angles (the relevant financial trend(s), qualitative risk factors, and a peer comparison if \
  one would add real value) before answering, rather than stopping at the first \
  minimally-sufficient answer. A shallow one-tool answer to a genuinely broad question is a \
  worse answer, not a more efficient one.

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
- Use run_python() for a calculation none of the other tools directly give you (a rolling \
  average, an outlier-excluded average, a custom ratio) — reference numbers from `scratchpad` \
  inside it rather than retyping them, since retyping is exactly how transcription errors \
  happen. Don't use it to recompute something compute_trend()/compare_companies() already gives \
  you directly.
- Use web_search() only for things genuinely outside the ingested reports (current stock price, \
  industry benchmark, recent news) — never as a substitute for a company's own reported figures, \
  and never to fill in a number an ingested-report tool already returned or could return.
- When you use vector_search() or web_search() results in your answer, cite the source (company \
  and page number for a report; the URL for a web result).
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


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    call_counter: int
    original_question: str


# --- duck-typed error helpers for httpx.HTTPStatusError --------------------
# A genuinely different (and more standard) shape than orchestrator.py's
# duck-typing around the raw mistralai SDK's exception — see this file's
# top docstring for the evidence behind why this wrapper exists at all.

def _get_http_status(e: BaseException) -> int | None:
    response = getattr(e, "response", None)
    return getattr(response, "status_code", None)


def _get_http_headers(e: BaseException) -> dict:
    response = getattr(e, "response", None)
    return dict(getattr(response, "headers", None) or {})


def _get_http_body_text(e: BaseException) -> str | None:
    response = getattr(e, "response", None)
    if response is None:
        return None
    try:
        return response.text
    except Exception:
        return None


def _is_too_many_tokens_error(e: BaseException) -> bool:
    """Same detection as orchestrator.py / extract.py — see extract.py's
    docstring for the full evidence behind treating this as retryable."""
    if _get_http_status(e) != 400:
        return False
    body_text = _get_http_body_text(e)
    try:
        body = json.loads(body_text) if body_text else {}
    except (json.JSONDecodeError, TypeError):
        return False
    return body.get("code") == "3210" or body.get("type") == "invalid_request_prompt"


def _invoke_with_retry(model, messages: list, max_retries: int = 5):
    """
    Retries on 429, 5xx, and the "too many tokens" 400 — identical
    conditions and backoff strategy to orchestrator.py's
    _call_mistral_with_retry, adapted to httpx.HTTPStatusError's shape
    instead of the raw mistralai SDK's. model.invoke(messages) is
    LangChain's call surface — this wrapper is required precisely because
    ChatMistralAI's own built-in retry does not cover this exception type
    (see the module docstring).
    """
    last_error: Exception | None = None
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            response = model.invoke(messages)
            print(f"  [mistral_call] attempt={attempt} took={time.time() - t0:.2f}s")
            return response
        except httpx.HTTPStatusError as e:
            status = _get_http_status(e)
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

            if status == 429 or (status is not None and 500 <= status < 600) or too_many_tokens:
                if retry_after is not None:
                    wait = float(retry_after)
                elif too_many_tokens:
                    wait = 20 * (attempt + 1)
                elif status == 429:
                    wait = 60
                else:
                    wait = min(2 ** attempt, 60)
                print(f"  Mistral call failed ({status}). limit_per_min={limit_per_min} "
                      f"remaining_per_min={remaining_per_min}. Waiting {wait:.0f}s, "
                      f"attempt {attempt + 1}/{max_retries}...")
                time.sleep(wait)
                continue
            raise

    raise RuntimeError(f"Mistral call failed after {max_retries} retries") from last_error


def build_graph(
    vectorstore_dir: str | Path,
    financials_db_path: str | Path,
    mistral_api_key: str | None = None,
    embedding_function: EmbeddingFunction | None = None,
    model=None,
    model_with_tools=None,
):
    """
    Builds and compiles the LangGraph app. Returns (compiled_graph, scratchpad) —
    scratchpad is the same dict run_python() reads from, exposed here so a
    caller (or a test) can inspect what's accumulated across a conversation.

    model / model_with_tools are injectable (same pattern as
    embedding_function elsewhere in this project) so the graph's own
    orchestration logic can be smoke-tested offline with a
    FakeMessagesListChatModel instead of a real ChatMistralAI — real
    ChatMistralAI's bind_tools() has no fake-model equivalent to test
    against, so tests inject a pre-scripted stand-in for both roles
    directly rather than trying to fake bind_tools() itself. Leave both
    None for real production use.

    Call the returned graph with:
        graph.invoke(
            {"messages": [HumanMessage(question)], "call_counter": 0, "original_question": question},
            config={"configurable": {"thread_id": "some-session-id"}},
        )
    Or just use ask(graph, thread_id, question) below, which also handles
    injecting the system prompt correctly on a thread's first turn only.
    The thread_id is what MemorySaver uses to keep separate conversations'
    message histories apart — reuse the same id across calls for a session
    to build on prior turns; use a fresh one to start a new conversation.
    Note that `scratchpad` (for run_python) is intentionally scoped to this
    whole build_graph() call, not per-thread — matches orchestrator.py's
    AgentSession.scratchpad, and is correct as long as one build_graph()
    call maps to one real user session, same as build_agent() there.
    """
    mistral_api_key = mistral_api_key or os.environ.get("MISTRAL_API_KEY")
    if model is None and not mistral_api_key:
        raise ValueError("Set MISTRAL_API_KEY (or pass mistral_api_key=) before calling build_graph()")

    scratchpad: dict = {}
    _global_call_id = {"n": 0}  # session-scoped (closure-captured), NOT per-turn — gives
                                 # scratchpad keys that stay unique across an entire session's
                                 # worth of questions, independent of the per-turn iteration
                                 # count used for the MAX_TOOL_ITERATIONS check below.
    raw_tools = make_agent_tools(vectorstore_dir, financials_db_path,
                                  embedding_function=embedding_function, scratchpad=scratchpad)
    lc_tools = [StructuredTool.from_function(func=fn, name=name) for name, fn in raw_tools.items()]
    tools_by_name = {t.name: t for t in lc_tools}

    # max_retries=0 here deliberately: _invoke_with_retry above is the real
    # retry logic (tuned to Mistral's actual observed behavior), not this
    # constructor's built-in one, which doesn't cover HTTP status errors at all.
    if model is None:
        model = ChatMistralAI(model=MODEL_NAME, api_key=mistral_api_key, max_retries=0, timeout=25)
    if model_with_tools is None:
        model_with_tools = model.bind_tools(lc_tools)

    def agent_node(state: AgentState) -> dict:
        response = _invoke_with_retry(model_with_tools, state["messages"])
        if response.content and response.tool_calls:
            print(f"  [plan] {response.content}")
        return {"messages": [response]}

    def tools_node(state: AgentState) -> dict:
        """turn_iteration (state["call_counter"]) bounds THIS question's tool
        loop, resetting each new ask() call — matching orchestrator.py's
        MAX_TOOL_ITERATIONS semantics. _global_call_id keeps incrementing for
        the whole session instead, purely so scratchpad keys stay unique
        across different questions in the same conversation."""
        last_message = state["messages"][-1]
        turn_iteration = state["call_counter"]
        tool_messages = []

        for tool_call in last_message.tool_calls:
            turn_iteration += 1
            _global_call_id["n"] += 1
            name = tool_call["name"]
            args = tool_call["args"]
            print(f"  [tool call {turn_iteration}] {name}({args})")

            tool = tools_by_name.get(name)
            if tool is None:
                result = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    result = tool.invoke(args)
                except Exception as e:
                    result = {"error": f"{name} raised {type(e).__name__}: {e}"}

            scratchpad[f"call_{_global_call_id['n']}_{name}"] = result
            is_error = isinstance(result, dict) and "error" in result
            result_str = json.dumps(result) if not isinstance(result, str) else result
            print(f"  [tool result] {'ERROR: ' if is_error else ''}{result_str[:200]}"
                  f"{'...' if len(result_str) > 200 else ''}")

            tool_messages.append(ToolMessage(content=result_str, tool_call_id=tool_call["id"], name=name))

        return {"messages": tool_messages, "call_counter": turn_iteration}

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            if state["call_counter"] >= MAX_TOOL_ITERATIONS:
                raise RuntimeError(f"Exceeded {MAX_TOOL_ITERATIONS} tool-calling iterations without a final answer")
            return "tools"
        return "verify"

    def verify_node(state: AgentState) -> dict:
        """Grounds the draft answer against every tool result gathered this
        SESSION (not just this turn — matches orchestrator.py, where a
        follow-up question's answer can legitimately reference numbers
        fetched earlier in the same conversation). Reuses verify_answer
        unchanged. Replaces the last message in place (same id) rather than
        appending a duplicate — required for LangGraph's add_messages
        reducer to update instead of duplicate; verified in this file's
        own tests."""
        final_message = state["messages"][-1]
        final_text = final_message.content or ""
        tool_results = list(scratchpad.values())
        verification = verify_answer(final_text, tool_results)
        if verification.verified:
            return {}
        unmatched_str = ", ".join(f"{n:,.2f}" for n in verification.unmatched_numbers)
        print(f"  [verifier] unmatched figures in the answer: {unmatched_str}")
        final_text += (
            f"\n\n⚠️ Note: this answer mentions figures ({unmatched_str}) that don't "
            f"match any value returned by the tools used to answer it — please "
            f"double-check these directly before relying on them."
        )
        return {"messages": [AIMessage(content=final_text, id=final_message.id)]}

    def self_critique_node(state: AgentState) -> dict:
        """One extra, lightweight call: does the (possibly verifier-amended)
        draft fully address the original question? Same reasoning as
        orchestrator.py's _check_completeness — a fresh, focused pass is
        more reliable than asking the model to grade its own completeness
        in the same turn it answered in."""
        final_message = state["messages"][-1]
        final_text = final_message.content or ""
        critique_prompt = (
            f"Original question: {state['original_question']}\n\n"
            f"Draft answer: {final_text}\n\n"
            f"Does this answer fully and directly address every part of the original question? "
            f"Respond with exactly \"COMPLETE\" if yes. If something is missing or only partially "
            f"addressed, respond with ONE short sentence naming what's missing — nothing else."
        )
        response = _invoke_with_retry(model, [HumanMessage(content=critique_prompt)])
        text = (response.content or "").strip()
        if text.upper().startswith("COMPLETE"):
            return {}
        print(f"  [self-check] incomplete: {text}")
        final_text += f"\n\n📝 Self-check: {text}"
        return {"messages": [AIMessage(content=final_text, id=final_message.id)]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("verify", verify_node)
    graph.add_node("self_critique", self_critique_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "verify": "verify"})
    graph.add_edge("tools", "agent")
    graph.add_edge("verify", "self_critique")
    graph.add_edge("self_critique", END)

    compiled = graph.compile(checkpointer=MemorySaver())
    return compiled, scratchpad


def ask(compiled_graph, thread_id: str, question: str) -> str:
    """Convenience wrapper matching orchestrator.py's ask(session, message)
    call shape — sends one question on the given thread (session) and
    returns the final answer text. Prior turns on the same thread_id are
    automatically included via the checkpointer; no manual history
    management needed.

    Only injects SYSTEM_INSTRUCTION on a thread's first turn — checked via
    get_state() rather than assumed, so a second ask() call on the same
    thread_id doesn't keep re-appending a duplicate system message onto an
    already-persisted history."""
    config = {"configurable": {"thread_id": thread_id}}
    current_state = compiled_graph.get_state(config)
    is_first_turn = not current_state.values.get("messages")

    new_messages = [HumanMessage(content=question)]
    if is_first_turn:
        new_messages = [SystemMessage(content=SYSTEM_INSTRUCTION), *new_messages]

    result = compiled_graph.invoke(
        {"messages": new_messages, "call_counter": 0, "original_question": question},
        config=config,
    )
    return result["messages"][-1].content


def get_turn_trace(compiled_graph, thread_id: str, messages_before: int) -> list[str]:
    """
    Reconstructs a human-readable trace of what happened during the most
    recent ask() call on this thread — every visible plan and every tool
    call/result — by reading the graph's persisted state after the fact
    and slicing off everything added since messages_before (the message
    count on this thread right before that ask() call).

    Deliberately built this way rather than threading a callback through
    the graph's nodes: LangGraph's checkpointed state is already the
    source of truth for what happened, so re-deriving a display trace from
    it is simpler and less error-prone than plumbing UI-specific hooks
    into the node functions themselves. Call len(get_state(...)) before
    ask() and pass it here after — see app_langgraph.py for the pattern.
    """
    state = compiled_graph.get_state({"configurable": {"thread_id": thread_id}})
    new_messages = state.values.get("messages", [])[messages_before:]

    trace = []
    for m in new_messages:
        if isinstance(m, AIMessage) and m.content and m.tool_calls:
            trace.append(f"[plan] {m.content}")
        elif isinstance(m, ToolMessage):
            trace.append(f"{m.name}(...) -> {str(m.content)[:200]}{'...' if len(str(m.content)) > 200 else ''}")
    return trace


def _run_smoke_test() -> None:
    """
    Verifies the graph's own orchestration logic (routing, scratchpad
    accumulation, message replacement vs. duplication, thread isolation,
    the MAX_TOOL_ITERATIONS bound) and the retry wrapper, all offline.

    model/model_with_tools are injected as FakeMessagesListChatModel
    instances rather than a real ChatMistralAI, since the base fake chat
    model has no bind_tools() implementation to fake realistically — see
    build_graph()'s docstring. The retry wrapper is tested separately,
    directly against real httpx.HTTPStatusError instances (confirmed this
    is genuinely what ChatMistralAI raises by reading its source — see
    this file's module docstring), since that's the one piece a
    FakeMessagesListChatModel can't exercise.

    The actual round trip to Mistral (a real ChatMistralAI, a real
    bind_tools() call, a real network response) is untested here — needs a
    live MISTRAL_API_KEY, same as every other live-API step in this
    project.
    """
    import tempfile

    import httpx as _httpx
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    from stores.financials_db import FinancialFact, upsert_facts_bulk
    from stores.vectorstore import _FakeEmbeddingFunction

    def tool_call_msg(name, args, content="", call_id=None):
        return AIMessage(
            content=content,
            tool_calls=[{"name": name, "args": args, "id": call_id or f"call_{name}", "type": "tool_call"}],
        )

    # --- basic flow: visible plan + tool call + final answer, verify clean, self-critique COMPLETE ---
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "financials.sqlite"
        upsert_facts_bulk(db_path, [
            FinancialFact("Tata Steel", 2021, "revenue", 156294.0, "INR crore"),
            FinancialFact("Tata Steel", 2024, "revenue", 218577.0, "INR crore"),
        ])
        final_text = "Tata Steel's revenue grew at roughly an 11.8% CAGR from FY2021 to FY2024."
        fake_agent = FakeMessagesListChatModel(responses=[
            tool_call_msg("compute_trend", {"company": "Tata Steel", "metric": "revenue"},
                          content="Plan: fetch the revenue trend for Tata Steel."),
            AIMessage(content=final_text),
        ])
        fake_critique = FakeMessagesListChatModel(responses=[AIMessage(content="COMPLETE")])
        graph, scratchpad = build_graph(Path(tmp) / "chroma", db_path, embedding_function=_FakeEmbeddingFunction(),
                                         model=fake_critique, model_with_tools=fake_agent)
        result = ask(graph, "smoke-1", "Analyze Tata Steel's revenue trend")
        assert result == final_text, f"unexpected: {result}"
        assert "call_1_compute_trend" in scratchpad

        trace = get_turn_trace(graph, "smoke-1", messages_before=2)  # system + human = 2 before this turn
        assert any("Plan: fetch the revenue trend" in t for t in trace), f"trace missing the plan: {trace}"
        assert any("compute_trend" in t for t in trace), f"trace missing the tool call: {trace}"
    print("basic flow: plan + tool call + final answer + clean verify + COMPLETE self-critique works end to end")
    print(f"  get_turn_trace reconstructed: {trace}")

    # --- follow-up question on the same thread: no duplicate system message, history correct ---
    with tempfile.TemporaryDirectory() as tmp:
        fake_agent = FakeMessagesListChatModel(responses=[
            AIMessage(content="First answer."), AIMessage(content="Second answer, building on the first."),
        ])
        fake_critique = FakeMessagesListChatModel(responses=[AIMessage(content="COMPLETE"), AIMessage(content="COMPLETE")])
        graph, _ = build_graph(Path(tmp) / "chroma", Path(tmp) / "db.sqlite", embedding_function=_FakeEmbeddingFunction(),
                                model=fake_critique, model_with_tools=fake_agent)
        first = ask(graph, "smoke-2", "First question")
        second = ask(graph, "smoke-2", "Second question")
        assert first == "First answer." and second == "Second answer, building on the first."
        state = graph.get_state({"configurable": {"thread_id": "smoke-2"}})
        messages = state.values["messages"]
        system_count = sum(1 for m in messages if isinstance(m, SystemMessage))
        assert system_count == 1, f"expected exactly 1 system message across 2 turns, got {system_count}"
        assert len(messages) == 5, f"expected system+2x(human+ai)=5 messages, got {len(messages)}"
    print("follow-up question on the same thread: system prompt injected exactly once, history correct")

    # --- verifier catches a garbled number, REPLACES the message (doesn't duplicate it) ---
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "financials.sqlite"
        upsert_facts_bulk(db_path, [FinancialFact("Tata Steel", 2024, "revenue", 232139.94, "INR crore")])
        garbled = "Tata Steel's revenue was approximately ₹23,213.99 crores."
        fake_agent = FakeMessagesListChatModel(responses=[
            tool_call_msg("get_financial_series", {"company": "Tata Steel", "metric": "revenue"}),
            AIMessage(content=garbled),
        ])
        fake_critique = FakeMessagesListChatModel(responses=[AIMessage(content="COMPLETE")])
        graph, _ = build_graph(Path(tmp) / "chroma", db_path, embedding_function=_FakeEmbeddingFunction(),
                                model=fake_critique, model_with_tools=fake_agent)
        result = ask(graph, "smoke-3", "What was Tata Steel's revenue?")
        assert garbled in result and "⚠️ Note" in result and "23,213.99" in result
        state = graph.get_state({"configurable": {"thread_id": "smoke-3"}})
        final_ai_messages = [m for m in state.values["messages"] if isinstance(m, AIMessage) and not m.tool_calls]
        assert len(final_ai_messages) == 1, \
            f"verifier must replace the final message in place, not duplicate it, got {len(final_ai_messages)}"
    print("verifier: reproduces and catches the real Tata Steel garbled-number bug, replaces message in place")

    # --- self-critique catches an incomplete answer, also replaces (not duplicates) ---
    with tempfile.TemporaryDirectory() as tmp:
        fake_agent = FakeMessagesListChatModel(responses=[AIMessage(content="Tata Steel's revenue grew.")])
        fake_critique = FakeMessagesListChatModel(responses=[
            AIMessage(content="The question also asked about L&T, which wasn't addressed.")
        ])
        graph, _ = build_graph(Path(tmp) / "chroma", Path(tmp) / "db.sqlite", embedding_function=_FakeEmbeddingFunction(),
                                model=fake_critique, model_with_tools=fake_agent)
        result = ask(graph, "smoke-4", "Compare Tata Steel and L&T's revenue growth")
        assert "📝 Self-check:" in result and "L&T" in result
        state = graph.get_state({"configurable": {"thread_id": "smoke-4"}})
        final_ai_messages = [m for m in state.values["messages"] if isinstance(m, AIMessage)]
        assert len(final_ai_messages) == 1, f"self-critique must replace, not duplicate, got {len(final_ai_messages)}"
    print("self-critique: catches an incomplete answer and appends a note, replaces message in place")

    # --- run_python reads scratchpad data written by an earlier tool call in the SAME turn ---
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "financials.sqlite"
        upsert_facts_bulk(db_path, [
            FinancialFact("Tata Steel", 2021, "revenue", 156294.0, "INR crore"),
            FinancialFact("Tata Steel", 2024, "revenue", 218577.0, "INR crore"),
        ])
        fake_agent = FakeMessagesListChatModel(responses=[
            tool_call_msg("compute_trend", {"company": "Tata Steel", "metric": "revenue"}, call_id="c1"),
            tool_call_msg("run_python",
                          {"code": "vals=[p['value'] for p in scratchpad['call_1_compute_trend']['series']]\n"
                                   "result=sum(vals)/len(vals)"}, call_id="c2"),
            AIMessage(content="The average is approximately 187,435.5 crore."),
        ])
        fake_critique = FakeMessagesListChatModel(responses=[AIMessage(content="COMPLETE")])
        graph, scratchpad = build_graph(Path(tmp) / "chroma", db_path, embedding_function=_FakeEmbeddingFunction(),
                                         model=fake_critique, model_with_tools=fake_agent)
        ask(graph, "smoke-5", "What's the average revenue across the data we have?")
        assert "call_1_compute_trend" in scratchpad and "call_2_run_python" in scratchpad
        assert abs(scratchpad["call_2_run_python"]["result"] - 187435.5) < 0.01
    print(f"run_python: correctly computed over scratchpad data from an earlier tool call this turn "
          f"-> {scratchpad['call_2_run_python']['result']}")

    # --- MAX_TOOL_ITERATIONS bounds a model that never stops calling tools ---
    with tempfile.TemporaryDirectory() as tmp:
        responses = [tool_call_msg("list_companies", {}, call_id=f"c{i}") for i in range(MAX_TOOL_ITERATIONS + 2)]
        fake_agent = FakeMessagesListChatModel(responses=responses)
        graph, _ = build_graph(Path(tmp) / "chroma", Path(tmp) / "db.sqlite", embedding_function=_FakeEmbeddingFunction(),
                                model=FakeMessagesListChatModel(responses=[]), model_with_tools=fake_agent)
        try:
            ask(graph, "smoke-6", "some question")
            raise AssertionError("expected RuntimeError for exceeding MAX_TOOL_ITERATIONS")
        except RuntimeError as e:
            assert "iterations" in str(e)
    print(f"a model that never stops calling tools is bounded at {MAX_TOOL_ITERATIONS} iterations, then raises")

    # --- retry wrapper against REAL httpx.HTTPStatusError (confirmed this is what
    # ChatMistralAI actually raises — see the module docstring) ---
    def make_http_error(status, headers=None, body=""):
        request = _httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
        response = _httpx.Response(status, request=request, headers=headers or {}, content=body.encode())
        return _httpx.HTTPStatusError(f"Error response {status}", request=request, response=response)

    call_count = {"n": 0}

    class FlakyModel:
        def invoke(self, messages):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise make_http_error(429, headers={"retry-after": "0.01", "x-ratelimit-limit-req-minute": "60"})
            return "SUCCESS"

    orig_sleep = time.sleep
    time.sleep = lambda s: None
    try:
        result = _invoke_with_retry(FlakyModel(), [])
        assert result == "SUCCESS" and call_count["n"] == 2
        print("retry wrapper: 429 with Retry-After retried correctly against real httpx.HTTPStatusError")

        class ZeroQuotaModel:
            def invoke(self, messages):
                raise make_http_error(429, headers={"x-ratelimit-limit-req-minute": "0"})

        try:
            _invoke_with_retry(ZeroQuotaModel(), [])
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "0 requests/minute" in str(e)
        print("retry wrapper: hard 0 req/min limit fails immediately, no wasted retries")

        call_count2 = {"n": 0}

        class TooManyTokensModel:
            def invoke(self, messages):
                call_count2["n"] += 1
                if call_count2["n"] < 2:
                    body = ('{"object":"error","message":"Too many tokens overall, split into more '
                             'batches.","type":"invalid_request_prompt","code":"3210"}')
                    raise make_http_error(400, body=body)
                return "SUCCESS_AFTER_SPLIT_RETRY"

        result2 = _invoke_with_retry(TooManyTokensModel(), [])
        assert result2 == "SUCCESS_AFTER_SPLIT_RETRY" and call_count2["n"] == 2
        print("retry wrapper: 'too many tokens' 400 retried in place, same as extract.py/orchestrator.py")

        class BadRequestModel:
            def invoke(self, messages):
                raise make_http_error(400, body='{"message": "invalid model name"}')

        try:
            _invoke_with_retry(BadRequestModel(), [])
            raise AssertionError("expected httpx.HTTPStatusError to propagate")
        except _httpx.HTTPStatusError:
            pass
        print("retry wrapper: a genuine unrelated 400 raises immediately, is not treated as retryable")
    finally:
        time.sleep = orig_sleep

    # --- real ChatMistralAI.bind_tools() accepts our full tool set without error
    # (schema-conversion check only — no network call happens here) ---
    from stores.vectorstore import _FakeEmbeddingFunction as _FEF
    with tempfile.TemporaryDirectory() as tmp:
        raw_tools = make_agent_tools(Path(tmp) / "chroma", Path(tmp) / "db.sqlite",
                                      embedding_function=_FEF(), scratchpad={})
        lc_tools = [StructuredTool.from_function(func=fn, name=name) for name, fn in raw_tools.items()]
        real_model = ChatMistralAI(model=MODEL_NAME, api_key="fake-key-schema-check-only", max_retries=0)
        real_model.bind_tools(lc_tools)
    print(f"ChatMistralAI.bind_tools() accepts the full real {len(lc_tools)}-tool set without error "
          f"(schema-conversion check only, no network call)")

    print("\nAll orchestrator_langgraph.py smoke tests passed "
          "(real Mistral call untested — needs a live MISTRAL_API_KEY).")


if __name__ == "__main__":
    _run_smoke_test()