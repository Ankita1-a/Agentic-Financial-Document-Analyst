"""
Financial Document Analyst — Streamlit app.

Upload a 10-K / annual report PDF and ask questions about it. Answers are
grounded via RAG (retrieval over the document) and a safe calculator tool,
using Groq's free-tier API for the LLM and local embeddings + an in-memory
Chroma index for retrieval — no paid services required.

Run locally:   streamlit run app.py
Deploy free:   Streamlit Community Cloud or Hugging Face Spaces (see DEPLOY.md)
"""

import ast
import json
import operator
import os
import re
import sys
import time
import types

import chromadb
import pdfplumber
import requests
import streamlit as st
from groq import APIConnectionError, Groq, RateLimitError
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# ---------------------------------------------------------------------------
# Workaround for a PIL.ImageText import bug seen on some hosted environments
# (harmless no-op if the environment's Pillow install is fine).
# ---------------------------------------------------------------------------
def _ensure_pil_imagetext_importable():
    try:
        from PIL import ImageText  # noqa: F401
        return
    except ImportError:
        import PIL
        stub = types.ModuleType("PIL.ImageText")
        sys.modules["PIL.ImageText"] = stub
        PIL.ImageText = stub


_ensure_pil_imagetext_importable()

st.set_page_config(page_title="Financial Document Analyst", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PREFERRED_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "llama-3.3-70b-versatile",
]

SYSTEM_PROMPT = (
    "You are a financial document analyst. Answer questions using ONLY information "
    "retrieved via the search_document tool — never rely on prior knowledge of the "
    "company. Always cite the page number(s) your answer is based on. For ANY "
    "arithmetic (growth rates, ratios, sums), use the calculate tool rather than "
    "computing it yourself. If the document doesn't contain the answer, say so "
    "plainly instead of guessing.\n\n"
    "Be economical with tool calls — each one costs tokens against a tight rate "
    "limit. Formulate one clear, specific search query per distinct fact you need. "
    "Do not re-search repeatedly with only minor rewordings of the same query, and "
    "never use a page number as search text — search for the concept or figure "
    "itself. If a search doesn't return what you need, try at most one "
    "substantially different phrasing, then answer with the best information "
    "available rather than continuing to search."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_document",
            "description": (
                "Search the financial document for relevant passages or tables. "
                "Use this whenever you need facts, figures, or statements from the "
                "report — never answer from memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for in the document"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a basic arithmetic expression. Use this for ANY math — "
                "growth rates, ratios, sums, percentages — instead of computing it yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '(150-120)/120*100'"}
                },
                "required": ["expression"],
            },
        },
    },
]

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


# ---------------------------------------------------------------------------
# Core logic (pure functions — ported directly from the Colab notebook)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def load_reranker():
    # A small, free, locally-run cross-encoder. Cross-encoders score a
    # (query, passage) pair jointly rather than comparing independent
    # embeddings, which makes them much more precise than raw vector
    # similarity — the tradeoff is they're too slow to run over an entire
    # document, so we only use one to re-score a short candidate list that
    # hybrid search has already narrowed down.
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def parse_pdf(file_obj):
    """Returns a list of dicts: {"page": int, "type": "text"/"table", "content": str}"""
    records = []
    with pdfplumber.open(file_obj) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                records.append({"page": i, "type": "text", "content": text.strip()})
            for table in page.extract_tables():
                cleaned_rows = [[cell.strip() if cell else "" for cell in row] for row in table]
                table_str = "\n".join(" | ".join(row) for row in cleaned_rows)
                if table_str.strip():
                    records.append({"page": i, "type": "table", "content": table_str})
    return records


def chunk_text(text, page, chunk_words=250, overlap_words=40):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_words
        chunk = " ".join(words[start:end])
        chunks.append({"page": page, "type": "text", "content": chunk})
        start += chunk_words - overlap_words
    return chunks


def build_chunks(records):
    all_chunks = []
    for r in records:
        if r["type"] == "text":
            all_chunks.extend(chunk_text(r["content"], r["page"]))
        else:  # tables stay whole
            all_chunks.append(r)
    return all_chunks


def build_document_context(records, max_pages=5, max_chars=2000):
    """Concatenate text from the first few pages as always-available context.

    Basic document-identity questions (company name, report title, fiscal
    year) shouldn't have to depend on retrieval luck. The canonical fact is
    usually stated once, in the front matter — while the rest of a long
    report just uses short-hand references ("the Company", "IHCL") and may
    also contain many other similarly-named entities (subsidiaries, stock
    exchanges, auditors) that can out-compete the real answer in a vague
    semantic search. Always including the front matter sidesteps that
    failure mode entirely for this class of question.
    """
    front_matter = [r["content"] for r in records if r["type"] == "text" and r["page"] <= max_pages]
    combined = "\n\n".join(front_matter)
    return combined[:max_chars]


def build_collection(chunks, embed_model):
    """Fresh in-memory (ephemeral) Chroma client per session — never shared
    across users and never persisted to disk, since each session's uploaded
    document is only ever relevant to that session."""
    client = chromadb.EphemeralClient()
    collection = client.create_collection("financial_doc")
    texts = [c["content"] for c in chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=False, batch_size=32)
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=[e.tolist() for e in embeddings],
        documents=texts,
        metadatas=[{"page": c["page"], "type": c["type"]} for c in chunks],
    )
    return collection


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25_index(chunks):
    """A keyword index alongside the semantic one. Embedding similarity is
    good at matching *meaning* but genuinely weak at matching a specific
    term precisely in a large document — 'capex' can get diluted among
    thousands of chunks even when several of them literally contain the
    word. BM25 guarantees exact-term matches aren't lost to that dilution."""
    tokenized = [_tokenize(c["content"]) for c in chunks]
    return BM25Okapi(tokenized)


def reciprocal_rank_fusion(rankings, rrf_k=60):
    """Merges multiple ranked lists of chunk indices into one combined
    ranking, without needing the two methods' raw scores to be on the same
    scale (cosine similarity and BM25 scores aren't comparable directly)."""
    scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)


def search_document(chunks, collection, bm25, embed_model, reranker, query,
                     k=4, candidate_pool=10, max_chars_per_chunk=1200, max_chars_for_reranking=800):
    t0 = time.time()
    n_candidates = min(candidate_pool, len(chunks))

    # Semantic candidates (meaning-based — good for paraphrases/concepts)
    query_embedding = embed_model.encode([query])[0].tolist()
    t_embed = time.time()
    vec_results = collection.query(query_embeddings=[query_embedding], n_results=n_candidates)
    vec_ids = [int(i) for i in vec_results["ids"][0]]
    t_chroma = time.time()

    # Keyword candidates (exact-term-based — good for specific terms/figures)
    bm25_scores = bm25.get_scores(_tokenize(query))
    bm25_ids = sorted(range(len(chunks)), key=lambda i: bm25_scores[i], reverse=True)[:n_candidates]
    t_bm25 = time.time()

    fused_ids = reciprocal_rank_fusion([vec_ids, bm25_ids])[:candidate_pool]
    if not fused_ids:
        return "No relevant passages found in the document."

    # Rerank the fused candidates with a cross-encoder for final precision.
    # Truncated separately (and more aggressively) than the final display
    # text: cross-encoders have a ~512-token input limit regardless, so
    # feeding them a multi-KB table chunk in full just burns CPU time on
    # text the model was going to truncate and ignore anyway — this was the
    # actual cause of slow responses on large documents with big tables.
    if len(fused_ids) > k:
        pairs = [(query, chunks[i]["content"][:max_chars_for_reranking]) for i in fused_ids]
        rerank_scores = reranker.predict(pairs)
        order = sorted(range(len(fused_ids)), key=lambda i: rerank_scores[i], reverse=True)
        top_ids = [fused_ids[i] for i in order[:k]]
    else:
        top_ids = fused_ids  # nothing to prioritize among — skip reranking entirely
    t_rerank = time.time()

    print(
        f"[search_document] chunks={len(chunks)} query={query!r} | "
        f"embed_query={t_embed - t0:.2f}s chroma={t_chroma - t_embed:.2f}s "
        f"bm25={t_bm25 - t_chroma:.2f}s rerank={t_rerank - t_bm25:.2f}s | "
        f"TOTAL={t_rerank - t0:.2f}s",
        flush=True,
    )

    formatted = []
    for i in top_ids:
        c = chunks[i]
        content = c["content"] if len(c["content"]) <= max_chars_per_chunk else c["content"][:max_chars_per_chunk] + " …[truncated]"
        formatted.append(f"[Page {c['page']} | {c['type']}]\n{content}")
    return "\n\n---\n\n".join(formatted)


def safe_calculate(expression: str):
    """Evaluate a basic arithmetic expression. No names, no function calls,
    no attribute access — just numbers and +-*/**()."""

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported or unsafe expression: {ast.dump(node)}")

    tree = ast.parse(expression, mode="eval")
    return _eval(tree.body)


def pick_available_model(api_key, preferred=PREFERRED_MODELS):
    resp = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15,
    )
    resp.raise_for_status()
    available_ids = {m["id"] for m in resp.json()["data"]}
    for model_id in preferred:
        if model_id in available_ids:
            return model_id
    raise RuntimeError(
        f"None of PREFERRED_MODELS are currently live on your account. "
        f"Available: {sorted(available_ids)}"
    )


def groq_completion_with_retry(client, max_retries=3, **kwargs):
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            response = client.chat.completions.create(**kwargs)
            print(f"[groq_call] attempt={attempt} took={time.time() - t0:.2f}s", flush=True)
            return response
        except RateLimitError as e:
            print(f"[groq_call] attempt={attempt} RateLimitError after {time.time() - t0:.2f}s", flush=True)
            if attempt == max_retries:
                raise
            wait_s = 15.0
            try:
                header_val = e.response.headers.get("retry-after")
                if header_val is not None:
                    wait_s = float(header_val)
            except (TypeError, ValueError, AttributeError):
                pass
            time.sleep(wait_s + 1)
        except APIConnectionError as e:
            # Covers both a true timeout and a general connection failure —
            # surfaces a clear message instead of a raw traceback or an
            # opaque, indefinite wait.
            print(f"[groq_call] attempt={attempt} APIConnectionError after {time.time() - t0:.2f}s: {e}", flush=True)
            if attempt == max_retries:
                raise RuntimeError(
                    "Lost connection to Groq after several attempts (each capped "
                    "at 25s). Check your internet connection or Groq's status at "
                    "status.groq.com, then try again."
                ) from e
            time.sleep(3)


def ask(question, groq_client, model, chunks, collection, bm25, embed_model, reranker,
        document_context="", max_turns=5, trace=None, on_tool_call=None):
    system_content = SYSTEM_PROMPT
    if document_context:
        system_content += (
            "\n\nFor reference, here is text from the first few pages of the document "
            "(useful for questions about the company/entity name, report title, or "
            "fiscal year without needing to search — but still verify numeric claims "
            "via search_document):\n---\n" + document_context + "\n---"
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]
    for _turn in range(max_turns):
        response = groq_completion_with_retry(
            groq_client, model=model, messages=messages, tools=TOOLS, tool_choice="auto",
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            return msg.content

        messages.append(msg)
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            if trace is not None:
                trace.append(f"{name}({args})")
            if on_tool_call is not None:
                on_tool_call(name, args)

            if name == "search_document":
                result = search_document(chunks, collection, bm25, embed_model, reranker, args["query"])
            elif name == "calculate":
                try:
                    result = str(safe_calculate(args["expression"]))
                except Exception as e:
                    result = f"Error: {e}"
            else:
                result = f"Unknown tool: {name}"

            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    # Ran out of turns while the model was still searching — force a real
    # answer from whatever's already been gathered, instead of giving up.
    if trace is not None:
        trace.append("⚠️ hit max_turns — forcing a final answer from gathered context")
    messages.append({
        "role": "user",
        "content": (
            "You've used all your allowed searches. Based ONLY on the information "
            "already gathered above, give your best final answer now. If something "
            "genuinely wasn't found, say so plainly instead of guessing."
        ),
    })
    final_response = groq_completion_with_retry(groq_client, model=model, messages=messages, tool_choice="none")
    return final_response.choices[0].message.content


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📊 Financial Document Analyst")
st.caption("Upload a 10-K / annual report PDF and ask questions — free-tier RAG + tool-calling agent.")

with st.sidebar:
    st.header("Setup")

    # Secret lookup order: Streamlit Cloud secrets -> environment variable
    # (Hugging Face Spaces / local) -> manual input as a last resort.
    default_key = ""
    try:
        default_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass
    if not default_key:
        default_key = os.environ.get("GROQ_API_KEY", "")

    api_key_input = st.text_input(
        "Groq API key",
        value=default_key,
        type="password",
        help="Free at console.groq.com. Only kept in this session's memory.",
    )
    uploaded_file = st.file_uploader("Upload a PDF (10-K / annual report)", type=["pdf"])
    st.markdown("---")
    st.caption(
        "Runs on Groq's free tier + local embeddings + an in-memory vector "
        "index. Nothing is persisted to disk or shared between sessions."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = None

if not api_key_input:
    st.info("Enter your free Groq API key in the sidebar to get started.")
    st.stop()

if uploaded_file is None:
    st.info("Upload a PDF in the sidebar to begin.")
    st.stop()

# (Re)process only when a genuinely new file is uploaded — Streamlit reruns
# this whole script on every interaction, so without this check we'd
# re-parse and re-embed the same document on every single chat message.
if st.session_state.processed_filename != uploaded_file.name:
    with st.spinner("Parsing PDF and building the search index — this can take a minute for large reports..."):
        embed_model = load_embed_model()
        reranker = load_reranker()
        records = parse_pdf(uploaded_file)
        chunks = build_chunks(records)
        collection = build_collection(chunks, embed_model)
        bm25 = build_bm25_index(chunks)
        document_context = build_document_context(records)
        groq_client = Groq(api_key=api_key_input, timeout=25.0, max_retries=0)
        model = pick_available_model(api_key_input)

        st.session_state.chunks = chunks
        st.session_state.embed_model = embed_model
        st.session_state.reranker = reranker
        st.session_state.collection = collection
        st.session_state.bm25 = bm25
        st.session_state.document_context = document_context
        st.session_state.groq_client = groq_client
        st.session_state.model = model
        st.session_state.processed_filename = uploaded_file.name
        st.session_state.messages = []

    st.success(f"Indexed {len(chunks)} chunks from **{uploaded_file.name}**. Using model: `{st.session_state.model}`")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("🔧 Tool calls"):
                for t in msg["trace"]:
                    st.code(t, language=None)

# Chat input
if question := st.chat_input("Ask about the document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        trace = []
        status = st.status("Thinking...", expanded=True)

        def _on_tool_call(name, args, _status=status):
            _status.write(f"🔧 {name}({args})")

        try:
            answer = ask(
                question,
                st.session_state.groq_client,
                st.session_state.model,
                st.session_state.chunks,
                st.session_state.collection,
                st.session_state.bm25,
                st.session_state.embed_model,
                st.session_state.reranker,
                document_context=st.session_state.document_context,
                trace=trace,
                on_tool_call=_on_tool_call,
            )
            status.update(label="Done", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Error", state="error", expanded=True)
            answer = f"⚠️ Something went wrong answering that: {e}"

        st.markdown(answer)
        if trace:
            with st.expander("🔧 Tool calls"):
                for t in trace:
                    st.code(t, language=None)

    st.session_state.messages.append({"role": "assistant", "content": answer, "trace": trace})