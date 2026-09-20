"""
tests/test_app.py

Verifies app.py using Streamlit's own official offline testing framework
(streamlit.testing.v1.AppTest), which runs the real script headlessly and
lets us inspect what it rendered — no browser, no live network needed.

Three scenarios:
1. No API key set at all -> the sidebar asks for one, nothing crashes.
2. API key set, no data ingested yet -> the "no data" message shows.
3. API key set, real (fixture) data present -> a full chat exchange,
   including a tool call and the verifier's pass-through, actually
   renders correctly end to end.

The Mistral network call is mocked at the lowest level (Mistral.chat.complete
itself) rather than by patching app.py's imported names, since AppTest
loads the script in its own way and patching the actual SDK call site is
robust regardless of exactly how that loading works.

Run with:
    python3 tests/test_app.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from streamlit.testing.v1 import AppTest

from mistralai.client.models.functioncall import FunctionCall
from mistralai.client.models.toolcall import ToolCall
from stores.financials_db import FinancialFact, upsert_facts_bulk
from stores.vectorstore import TextChunk, add_chunks, get_collection

APP_PATH = str(_PROJECT_ROOT / "app.py")
FINANCIALS_DB_PATH = _PROJECT_ROOT / "data" / "financials.sqlite"
CHROMA_DIR = _PROJECT_ROOT / "data" / "chroma_db"


def _make_final_response(text: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = text
    response.choices[0].message.tool_calls = None
    return response


def _make_tool_call_response(tool_name: str, args_json: str) -> MagicMock:
    tc = ToolCall(id="call_1", type="function", function=FunctionCall(name=tool_name, arguments=args_json))
    response = MagicMock()
    response.choices[0].message.content = None
    response.choices[0].message.tool_calls = [tc]
    return response


def test_no_api_key() -> None:
    os.environ.pop("MISTRAL_API_KEY", None)
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"app crashed with no API key set: {at.exception}"
    assert len(at.sidebar.text_input) == 1, "should show exactly one API key input when none is set"
    print("test_no_api_key: sidebar asks for a key, no crash")


def test_api_key_but_no_data() -> None:
    if FINANCIALS_DB_PATH.exists():
        FINANCIALS_DB_PATH.unlink()

    os.environ["MISTRAL_API_KEY"] = "fake-key-for-app-test"
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"app crashed with a key but no data: {at.exception}"
    assert len(at.chat_input) == 0, "chat should not be offered before any data is ingested"
    print("test_api_key_but_no_data: no chat input shown yet, no crash")


def test_full_chat_flow_with_real_data() -> None:
    if FINANCIALS_DB_PATH.exists():
        FINANCIALS_DB_PATH.unlink()
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)

    upsert_facts_bulk(FINANCIALS_DB_PATH, [
        FinancialFact("Tata Steel", 2021, "revenue", 156294.0, "INR crore"),
        FinancialFact("Tata Steel", 2024, "revenue", 218577.0, "INR crore"),
    ])

    os.environ["MISTRAL_API_KEY"] = "fake-key-for-app-test"

    # Patch the real embeddings call site, not app.py's imported names — needed for
    # BOTH seeding the fixture chunk below and app.py's later real code path to use
    # the SAME embedding function identity ("mistral"). Chroma persists which
    # embedding function a collection was created with and rejects a mismatch, so
    # seeding with _FakeEmbeddingFunction while app.py opens the same collection
    # with a real MistralEmbeddingFunction would fail with a real Chroma error, not
    # just a test artifact — confirmed by hitting exactly that error first.
    final_text = "Tata Steel's revenue grew at roughly an 11.8% CAGR from FY2021 to FY2024."

    def fake_embeddings_create(model, inputs):
        response = MagicMock()
        response.data = [MagicMock(embedding=[0.0] * 8) for _ in inputs]
        return response

    with patch("mistralai.client.chat.Chat.complete") as mock_complete, \
         patch("mistralai.client.embeddings.Embeddings.create", side_effect=fake_embeddings_create):
        mock_complete.side_effect = [
            _make_tool_call_response("compute_trend", '{"company": "Tata Steel", "metric": "revenue"}'),
            _make_final_response(final_text),
        ]

        from stores.vectorstore import MistralEmbeddingFunction
        collection = get_collection(CHROMA_DIR, embedding_function=MistralEmbeddingFunction(api_key="fake-key-for-app-test"))
        add_chunks(collection, [
            TextChunk("Tata Steel", 2024, page=12, section="risk_factors",
                      text="Raw material price volatility is a key risk for Tata Steel."),
        ])

        at = AppTest.from_file(APP_PATH, default_timeout=15)
        at.run()
        assert not at.exception, f"app crashed on initial load with real data: {at.exception}"
        assert len(at.chat_input) == 1, "chat input should appear once data exists"

        sidebar_text = " ".join(m.value for m in at.sidebar.markdown) if at.sidebar.markdown else ""
        assert "Tata Steel" in sidebar_text or any(
            "Tata Steel" in str(w) for w in at.sidebar
        ), "the ingested company should be listed in the sidebar"

        at.chat_input[0].set_value("Analyze Tata Steel's revenue trend").run()
        assert not at.exception, f"app crashed handling a chat message: {at.exception}"

        rendered_messages = [m.markdown[0].value for m in at.chat_message if m.markdown]
        assert any("Analyze Tata Steel's revenue trend" in m for m in rendered_messages), \
            "the user's own message should render in the chat"
        assert any(final_text in m for m in rendered_messages), \
            "the agent's final answer should render in the chat"

        assert mock_complete.call_count == 2, "expected exactly the 2 mocked calls (1 tool call, 1 final answer)"

    print("test_full_chat_flow_with_real_data: sidebar shows the ingested company, a full "
          "tool-call-then-answer exchange renders correctly end to end")


if __name__ == "__main__":
    test_no_api_key()
    test_api_key_but_no_data()
    test_full_chat_flow_with_real_data()
    print("\nAll app.py tests passed.")