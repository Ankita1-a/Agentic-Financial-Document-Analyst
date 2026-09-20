"""
app_langgraph.py

The actual UI for agent/orchestrator_langgraph.py — this is what makes it
something a person (or an interviewer) can actually open and use, rather
than a module only exercised by its own smoke tests.

Different shape from app.py/app2.py on purpose: those let you upload one
PDF per session and ask questions about just that document. This app
queries the PERSISTENT, pre-ingested stores (data/financials.sqlite +
data/chroma_db) built by ingest/build_index.py — there's no upload step,
because the whole point of the ingest pipeline is that ingestion happens
once, ahead of time, for as many companies/years as you've run it on.

Run:
    streamlit run app_langgraph.py
"""

import os
import sys
import uuid
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from agent.orchestrator_langgraph import MODEL_NAME, ask, build_graph, get_turn_trace
from stores.financials_db import list_companies

FINANCIALS_DB_PATH = _PROJECT_ROOT / "data" / "financials.sqlite"
CHROMA_DIR = _PROJECT_ROOT / "data" / "chroma_db"

st.set_page_config(page_title="Financial Analyst Agent (LangGraph)", page_icon="📊", layout="wide")
st.title("📊 Financial Analyst Agent — LangGraph edition")
st.caption(
    "Same capabilities as the RAG apps (planning, verification, self-critique, code execution, "
    "web search) — this one runs on LangChain + LangGraph against pre-ingested company data "
    "instead of a per-session PDF upload."
)

with st.sidebar:
    st.header("Setup")

    default_key = ""
    try:
        default_key = st.secrets.get("MISTRAL_API_KEY", "")
    except Exception:
        pass
    if not default_key:
        default_key = os.environ.get("MISTRAL_API_KEY", "")

    api_key_input = st.text_input(
        "Mistral API key",
        value=default_key,
        type="password",
        help="Free at console.mistral.ai. Only kept in this session's memory.",
    )

    st.markdown("---")
    if not FINANCIALS_DB_PATH.exists():
        st.warning(
            "No ingested data found yet. Run this first:\n\n"
            "`python -m ingest.build_index manifest.json data/financials.sqlite data/chroma_db`"
        )
        ingested_companies = []
    else:
        ingested_companies = list_companies(FINANCIALS_DB_PATH)
        if ingested_companies:
            st.markdown("**Ingested companies:**")
            for c in ingested_companies:
                st.markdown(f"- {c}")
        else:
            st.info("financials.sqlite exists but has no companies in it yet.")

    st.markdown("---")
    st.caption(
        f"Model: `{MODEL_NAME}` (Mistral free tier). Graph: LangGraph StateGraph with a "
        f"MemorySaver checkpointer — this session's conversation persists across questions "
        f"via a thread_id, not a hand-rolled message list."
    )

if not api_key_input:
    st.info("Enter your free Mistral API key in the sidebar to get started.")
    st.stop()

if not FINANCIALS_DB_PATH.exists() or not ingested_companies:
    st.info("Ingest at least one company's report before asking questions — see the sidebar.")
    st.stop()

# Build the graph once per session (not once per question) — mirrors
# app.py/app2.py's "(re)process only when needed" pattern, just keyed on
# the API key instead of an uploaded filename, since there's no per-session
# document here to key on.
if st.session_state.get("built_with_key") != api_key_input:
    with st.spinner("Building the agent graph (loading the embedding model can take a few seconds)..."):
        graph, scratchpad = build_graph(CHROMA_DIR, FINANCIALS_DB_PATH, mistral_api_key=api_key_input)
        st.session_state.graph = graph
        st.session_state.scratchpad = scratchpad
        st.session_state.built_with_key = api_key_input
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.display_messages = []

# Display chat history
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("🔧 Plan & tool calls"):
                for t in msg["trace"]:
                    st.code(t, language=None)

# Chat input
if question := st.chat_input("Ask about the ingested companies..."):
    st.session_state.display_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=True)
        graph = st.session_state.graph
        thread_id = st.session_state.thread_id

        state_before = graph.get_state({"configurable": {"thread_id": thread_id}})
        messages_before = len(state_before.values.get("messages", []))

        try:
            answer = ask(graph, thread_id, question)
            status.update(label="Done", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Error", state="error", expanded=True)
            answer = f"⚠️ Something went wrong answering that: {e}"

        trace = get_turn_trace(graph, thread_id, messages_before) if "Something went wrong" not in answer else []

        st.markdown(answer)
        if trace:
            with st.expander("🔧 Plan & tool calls"):
                for t in trace:
                    st.code(t, language=None)

    st.session_state.display_messages.append({"role": "assistant", "content": answer, "trace": trace})
