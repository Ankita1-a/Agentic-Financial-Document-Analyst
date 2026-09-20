"""
scripts/chat_langgraph.py

Fastest way to actually talk to agent/orchestrator_langgraph.py with a real
MISTRAL_API_KEY against your real ingested data — no UI to build first, no
Streamlit round-trip while you're just checking whether the model, the
tools, and the graph behave the way the smoke tests say they should.

Usage:
    export MISTRAL_API_KEY=your-key-here
    python scripts/chat_langgraph.py

Type a question, see the answer (with its visible plan / tool calls printed
above it, same as every other diagnostic script in this project). Type
'quit' to exit. Every question in one run shares the same thread_id, so
follow-up questions build on prior ones, same as a real chat session.
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.orchestrator_langgraph import ask, build_graph

FINANCIALS_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "financials.sqlite"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_db"


def main() -> None:
    if not FINANCIALS_DB_PATH.exists():
        print(f"No data at {FINANCIALS_DB_PATH} yet — run ingest/build_index.py first:")
        print("    python -m ingest.build_index manifest.json data/financials.sqlite data/chroma_db")
        sys.exit(1)

    print("Building the graph (loading the embedding model can take a few seconds on first run)...")
    try:
        graph, scratchpad = build_graph(CHROMA_DIR, FINANCIALS_DB_PATH)
    except ValueError as e:
        print(f"⚠️  {e}")
        sys.exit(1)
    thread_id = str(uuid.uuid4())
    print(f"Ready. Session thread_id: {thread_id}")
    print("Type a question, or 'quit' to exit. Follow-up questions build on prior ones in this run.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            break

        try:
            answer = ask(graph, thread_id, question)
        except Exception as e:
            print(f"\n⚠️  {type(e).__name__}: {e}\n")
            continue

        print(f"\nAgent: {answer}\n")

    print(f"\nSession scratchpad had {len(scratchpad)} tool result(s) accumulated: {list(scratchpad.keys())}")


if __name__ == "__main__":
    main()
