"""
backend/main.py

A thin HTTP wrapper around agent/orchestrator_langgraph.py — this is what
lets the agent be called from a separate frontend process (or any other
HTTP client) instead of being imported in-process, the way app_langgraph.py
does it.

Design notes:
- Non-streaming: POST /chat waits for the full answer and returns it along
  with the turn's trace (plan + tool calls) in one response. The current
  Streamlit app shows these live because everything runs in one process;
  splitting into a separate backend means either accepting that (what this
  does) or adding SSE/WebSocket streaming (more correct to the existing UX,
  meaningfully more code — not built here, flagged as a clear next step).
- Graphs are cached per API key, not rebuilt per request, but also not
  built once globally — different visitors can supply different Mistral
  keys (matching app_langgraph.py's sidebar override), and each gets its
  own cached graph rather than forcing a single shared one.
- No CORS middleware: the frontend calls this server-to-server (Streamlit's
  Python backend calling this API via `requests`), not from JavaScript in
  a visitor's browser, so cross-origin restrictions don't apply here.

Run locally:
    uvicorn backend.main:app --reload --port 8000
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.orchestrator_langgraph import ask, build_graph, get_turn_trace
from stores.financials_db import list_companies

FINANCIALS_DB_PATH = Path(os.environ.get("FINANCIALS_DB_PATH", _PROJECT_ROOT / "data" / "financials.sqlite"))
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", _PROJECT_ROOT / "data" / "chroma_db"))

# Keyed by API key so different visitors' own keys (see ChatRequest.mistral_api_key)
# each get their own graph, without rebuilding on every single request.
_graph_cache: dict[str, tuple] = {}


def _get_or_build_graph(api_key: str):
    if api_key not in _graph_cache:
        _graph_cache[api_key] = build_graph(CHROMA_DIR, FINANCIALS_DB_PATH, mistral_api_key=api_key)
    return _graph_cache[api_key]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Best-effort warmup: if a server-side default key is configured, build
    # its graph now instead of on the first request. Not required — if this
    # fails (bad key, no key at all), requests still work as long as the
    # caller supplies their own key per-request.
    default_key = os.environ.get("MISTRAL_API_KEY", "")
    if default_key:
        try:
            _get_or_build_graph(default_key)
        except Exception as e:
            print(f"[startup] Warmup build with the server default key failed (non-fatal): {e}")
    yield


app = FastAPI(title="Financial Analyst Agent API", lifespan=lifespan)


class ChatRequest(BaseModel):
    thread_id: str
    question: str
    mistral_api_key: str | None = None


class ChatResponse(BaseModel):
    answer: str
    trace: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/companies")
def companies():
    if not FINANCIALS_DB_PATH.exists():
        return []
    return list_companies(FINANCIALS_DB_PATH)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    api_key = request.mistral_api_key or os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="No Mistral API key available — pass mistral_api_key in the request, "
                   "or set MISTRAL_API_KEY on the server.",
        )
    if not FINANCIALS_DB_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="No ingested data found. Run ingest.build_index before asking questions.",
        )

    try:
        graph, _scratchpad = _get_or_build_graph(api_key)
    except Exception as e:
        # Broadened from `except ValueError` — that only caught the missing-key
        # case build_graph() itself raises deliberately. Anything else (a
        # chromadb open failure, a bad data path, whatever's actually going on
        # inside Docker right now) was escaping uncaught and surfacing as an
        # opaque, undiagnosable 500. Now it comes back as a real message.
        raise HTTPException(status_code=500, detail=f"Failed to build the agent graph: {type(e).__name__}: {e}") from e

    try:
        state_before = graph.get_state({"configurable": {"thread_id": request.thread_id}})
        messages_before = len(state_before.values.get("messages", []))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read graph state: {type(e).__name__}: {e}") from e

    try:
        answer = ask(graph, request.thread_id, request.question)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e

    try:
        trace = get_turn_trace(graph, request.thread_id, messages_before)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build trace: {type(e).__name__}: {e}") from e

    return ChatResponse(answer=answer, trace=trace)