"""
frontend/app.py

A thin client for backend/main.py — genuinely decoupled from the agent now:
this file has no import of agent/, stores/, or ingest/ at all, just
`requests` calls to the backend's HTTP API. This is the actual point of
splitting into backend+frontend — this file could be replaced by a CLI, a
mobile app, or a different UI entirely without touching the agent at all.

Run locally (with the backend already running separately):
    streamlit run frontend/app.py
"""

import os
import uuid

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Financial Analyst Agent", page_icon="📊", layout="wide")
st.title("📊 Financial Analyst Agent")
st.caption(
    "Planning, verification, self-critique, code execution, web search — "
    f"served by a separate FastAPI backend at `{BACKEND_URL}`."
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
        help="Free at console.mistral.ai. Sent with each request to the backend, never stored there.",
    )

    st.markdown("---")
    try:
        resp = requests.get(f"{BACKEND_URL}/companies", timeout=10)
        resp.raise_for_status()
        ingested_companies = resp.json()
        if ingested_companies:
            st.markdown("**Ingested companies:**")
            for c in ingested_companies:
                st.markdown(f"- {c}")
        else:
            st.info("No companies ingested yet.")
    except requests.RequestException as e:
        st.error(f"Can't reach the backend at {BACKEND_URL}: {e}")
        ingested_companies = None  # distinct from [] — means "couldn't check", not "checked, empty"

    st.markdown("---")
    st.caption("Model: Mistral free tier, via the backend's LangGraph agent.")

if not api_key_input:
    st.info("Enter your free Mistral API key in the sidebar to get started.")
    st.stop()

if ingested_companies is None:
    st.stop()  # error already shown above
if not ingested_companies:
    st.info("Ingest at least one company's report on the backend before asking questions.")
    st.stop()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.display_messages = []

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("🔧 Plan & tool calls"):
                for t in msg["trace"]:
                    st.code(t, language=None)

if question := st.chat_input("Ask about the ingested companies..."):
    st.session_state.display_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Non-streaming: the backend does the full plan -> tools -> verify ->
        # self-critique loop before responding, so this waits for one
        # complete HTTP response rather than showing live progress. See
        # backend/main.py's docstring for the streaming trade-off.
        with st.spinner("Thinking... (this can take a while — the backend is planning, calling "
                         "tools, verifying, and self-critiquing before responding)"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={
                        "thread_id": st.session_state.thread_id,
                        "question": question,
                        "mistral_api_key": api_key_input,
                    },
                    timeout=180,
                )
                resp.raise_for_status()
                body = resp.json()
                answer = body["answer"]
                trace = body["trace"]
            except requests.RequestException as e:
                detail = ""
                try:
                    detail = e.response.json().get("detail", "") if e.response is not None else ""
                except Exception:
                    pass
                answer = f"⚠️ Something went wrong answering that: {detail or e}"
                trace = []

        st.markdown(answer)
        if trace:
            with st.expander("🔧 Plan & tool calls"):
                for t in trace:
                    st.code(t, language=None)

    st.session_state.display_messages.append({"role": "assistant", "content": answer, "trace": trace})