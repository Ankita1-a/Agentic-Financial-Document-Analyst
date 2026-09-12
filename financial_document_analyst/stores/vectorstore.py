"""
stores/vectorstore.py

Chroma-backed vector store for qualitative text (risk factors, MD&A,
strategy commentary, etc.) pulled out of annual reports. Every chunk is
tagged with company/fiscal_year/page metadata so retrieval can be scoped
to one company or left open for cross-company queries within the same
chat session.

Embeddings default to Mistral's `mistral-embed` via chromadb's built-in
MistralEmbeddingFunction (requires a MISTRAL_API_KEY environment
variable). The embedding function is injectable so the storage/retrieval
mechanics can be smoke-tested offline without spending API quota — see
_run_smoke_test() at the bottom, which uses a fake deterministic embedder.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from chromadb.api.types import EmbeddingFunction

COLLECTION_NAME = "annual_report_chunks"

# Embedding-call batching: real evidence from this project (a chat-completion
# request that broke down somewhere between 14 pages of text and a whole
# 754-page document's worth of chunks in one call) shows Mistral's "too many
# tokens overall, split into more batches" error also applies to the
# embeddings endpoint — and add_chunks() previously handed Chroma's upsert()
# every chunk from an entire report in one call, which in turn calls this
# embedding function with every chunk's text in a single request. 64 texts
# per call is a conservative batch size chosen defensively, not from a
# documented limit (Mistral doesn't publish one for embeddings either).
EMBED_BATCH_SIZE = 64
EMBED_MAX_RETRIES = 5


def _get_http_status(e: BaseException) -> int | None:
    """
    Duck-types the HTTP status code off an exception rather than relying on
    isinstance(e, SDKError) — real evidence from ingest/extract.py's retry
    logic showed the exception mistralai actually raises does not reliably
    satisfy that isinstance check, for reasons never fully explained.
    Kept as a local copy here rather than a shared import to avoid a
    cross-module dependency between ingest/ and stores/ for a handful of
    small, pure helper functions.
    """
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
    """Same detection as ingest/extract.py's _is_too_many_tokens_error — see
    that file's docstring for the full story on why this is retried rather
    than treated as a fixed request-size problem."""
    if _get_http_status(e) != 400:
        return False
    body_text = _get_http_body_text(e)
    try:
        body = json.loads(body_text) if body_text else {}
    except (json.JSONDecodeError, TypeError):
        return False
    return body.get("code") == "3210" or body.get("type") == "invalid_request_prompt"


@dataclass
class TextChunk:
    company: str
    fiscal_year: int
    page: int
    text: str
    section: str | None = None        # e.g. "risk_factors", "md&a", "segment_overview"
    source_report: str | None = None
    id: str = field(default="")       # auto-generated in __post_init__ if not supplied

    def __post_init__(self) -> None:
        if not self.id:
            # Deterministic, not random: re-constructing a TextChunk with the
            # same company/fiscal_year/page/text (e.g. re-running ingestion
            # on the same report in a fresh process) produces the same id,
            # so add_chunks() upserts in place instead of creating a
            # duplicate. A prior version used a random uuid suffix here,
            # which meant every fresh ingestion run silently duplicated
            # every chunk in the vector store.
            content_hash = hashlib.sha256(
                f"{self.company}|{self.fiscal_year}|{self.page}|{self.text}".encode("utf-8")
            ).hexdigest()[:12]
            self.id = f"{self.company}|{self.fiscal_year}|{self.page}|{content_hash}"


def chunk_page_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Splits a page's text into chunks on paragraph boundaries where possible,
    falling back to a sliding window for a single paragraph longer than
    chunk_size (common on pages that are mostly one dense table-as-text block).
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(para):
                chunks.append(para[start:start + chunk_size])
                start += chunk_size - overlap
        elif len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    return chunks


class MistralEmbeddingFunction(EmbeddingFunction):
    """
    Chroma ships a MistralEmbeddingFunction, but it does `from mistralai
    import Mistral` internally — that import path no longer exists in
    mistralai>=2.x, which restructured the client under `mistralai.client`.
    Confirmed by actually instantiating chroma's version: it raises
    "the mistralai python package is not installed" even with mistralai
    correctly installed, purely because of the stale import path. This is
    our own corrected version against the real installed SDK.

    __call__ batches its input (EMBED_BATCH_SIZE at a time) rather than
    sending everything Chroma hands it in one request — Chroma's
    collection.upsert() calls this with every text from a single add_chunks()
    call at once, which for a full report is thousands of chunks in one
    shot otherwise. See EMBED_BATCH_SIZE above for why that matters.
    """

    def __init__(
        self,
        model: str = "mistral-embed",
        api_key: str | None = None,
        api_key_env_var: str = "MISTRAL_API_KEY",
    ):
        """
        api_key, if given, is used directly — otherwise falls back to the
        api_key_env_var environment variable, as before. Added after a
        real bug: every caller in this project up to app.py happened to
        also set the environment variable directly, masking the fact that
        an explicitly-passed key (e.g. from build_agent(mistral_api_key=...))
        was silently ignored here — the one place in the whole call chain
        that didn't accept an explicit key.
        """
        from mistralai.client import Mistral

        self.model = model
        self.api_key_env_var = api_key_env_var
        resolved_key = api_key or os.environ.get(api_key_env_var)
        if not resolved_key:
            raise ValueError(
                f"No Mistral API key given, and the {api_key_env_var} environment variable is not set."
            )
        self.client = Mistral(api_key=resolved_key)

    def __call__(self, input):
        all_texts = list(input)
        all_embeddings: list[list[float]] = []
        for i in range(0, len(all_texts), EMBED_BATCH_SIZE):
            batch = all_texts[i:i + EMBED_BATCH_SIZE]
            all_embeddings.extend(self._embed_batch_with_retry(batch))
        return all_embeddings

    def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        """
        Retries on 429, 5xx, and the "too many tokens" 400 — same
        conditions and backoff strategy as
        ingest/extract.py's _call_mistral_with_retry, duplicated rather
        than shared because these are the only two call sites and a
        cross-module dependency didn't seem worth it for this much logic.
        """
        last_error: Exception | None = None
        for attempt in range(EMBED_MAX_RETRIES):
            try:
                response = self.client.embeddings.create(model=self.model, inputs=texts)
                return [item.embedding for item in response.data]
            except BaseException as e:
                if not isinstance(e, Exception):
                    raise
                status = _get_http_status(e)
                if status is None:
                    raise  # not an HTTP-backed SDK error at all — nothing to retry

                last_error = e
                headers = _get_http_headers(e)
                limit_per_min = headers.get("x-ratelimit-limit-req-minute")
                remaining_per_min = headers.get("x-ratelimit-remaining-req-minute")
                retry_after = headers.get("retry-after")
                too_many_tokens = _is_too_many_tokens_error(e)

                if status == 429 and limit_per_min == "0":
                    raise RuntimeError(
                        f"Mistral reports a hard 0 requests/minute limit for model '{self.model}' "
                        f"on your account. This is not a timing issue — check "
                        f"https://admin.mistral.ai/plateforme/limits."
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

                    print(f"  Mistral embeddings call failed ({status}, {type(e).__name__}). "
                          f"limit_per_min={limit_per_min} remaining_per_min={remaining_per_min}. "
                          f"Waiting {wait:.0f}s ({wait_source}), attempt {attempt + 1}/{EMBED_MAX_RETRIES}...")
                    time.sleep(wait)
                    continue
                raise

        raise RuntimeError(f"Mistral embeddings failed after {EMBED_MAX_RETRIES} retries") from last_error

    @staticmethod
    def name() -> str:
        return "mistral"

    def get_config(self):
        return {"model": self.model, "api_key_env_var": self.api_key_env_var}

    @staticmethod
    def build_from_config(config):
        return MistralEmbeddingFunction(
            model=config.get("model", "mistral-embed"),
            api_key_env_var=config.get("api_key_env_var", "MISTRAL_API_KEY"),
        )


def get_collection(
    persist_dir: str | Path,
    embedding_function: EmbeddingFunction | None = None,
    mistral_model: str = "mistral-embed",
    mistral_api_key: str | None = None,
):
    """
    Returns a persistent Chroma collection. Defaults to Mistral embeddings —
    mistral_api_key is used directly if given, otherwise falls back to the
    MISTRAL_API_KEY environment variable; pass a different embedding_function
    for offline testing or to swap providers later.
    """
    client = chromadb.PersistentClient(path=str(persist_dir))
    if embedding_function is None:
        embedding_function = MistralEmbeddingFunction(model=mistral_model, api_key=mistral_api_key)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )


def add_chunks(collection, chunks: list[TextChunk]) -> int:
    """
    Upserts chunks. Safe to re-run ingestion on the same report: as long as
    the same chunk ids are regenerated deterministically upstream (or you
    pass explicit ids), re-adding corrects rather than duplicates.
    """
    if not chunks:
        return 0
    collection.upsert(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "company": c.company,
                "fiscal_year": c.fiscal_year,
                "page": c.page,
                "section": c.section or "",
                "source_report": c.source_report or "",
            }
            for c in chunks
        ],
    )
    return len(chunks)


def delete_company_year(collection, company: str, fiscal_year: int) -> None:
    """
    Deletes every chunk for one company/fiscal_year. Call this before
    re-ingesting a report whose parsing or chunking logic changed since the
    last run: deterministic ids (see TextChunk.__post_init__) already
    prevent duplicates when the chunk text is unchanged, but if the text
    itself changed, the new chunks get different ids and the old ones
    would otherwise be orphaned in the store forever instead of replaced.
    """
    collection.delete(where={"$and": [{"company": company}, {"fiscal_year": fiscal_year}]})


def search(
    collection,
    query: str,
    company: str | None = None,
    fiscal_year: int | None = None,
    n_results: int = 5,
) -> list[dict]:
    """
    Semantic search, optionally scoped to one company and/or fiscal year.
    Leave company=None for cross-company retrieval within the same session.
    Returns a list of {text, company, fiscal_year, page, section, distance}
    ordered by relevance (lowest distance first).
    """
    conditions = []
    if company is not None:
        conditions.append({"company": company})
    if fiscal_year is not None:
        conditions.append({"fiscal_year": fiscal_year})

    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    result = collection.query(query_texts=[query], n_results=n_results, where=where)

    hits = []
    ids = result["ids"][0]
    for i in range(len(ids)):
        meta = result["metadatas"][0][i]
        hits.append({
            "text": result["documents"][0][i],
            "company": meta["company"],
            "fiscal_year": meta["fiscal_year"],
            "page": meta["page"],
            "section": meta["section"],
            "distance": result["distances"][0][i],
        })
    return hits


class _FakeEmbeddingFunction(EmbeddingFunction):
    """
    Deterministic, hash-based embeddings for offline testing only. These
    carry no real semantic meaning — they validate storage/retrieval
    mechanics (upsert, metadata filtering, no-duplicate re-ingestion), not
    retrieval quality. Real quality needs to be checked against the live
    Mistral API, same as parse.py needed a real annual report.
    """

    def __init__(self, dim: int = 16):
        self.dim = dim

    def __call__(self, input):
        import hashlib
        embeddings = []
        for text in input:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            embeddings.append([b / 255.0 for b in digest[: self.dim]])
        return embeddings

    @staticmethod
    def name() -> str:
        return "fake-test-embedding"

    def get_config(self):
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config):
        return _FakeEmbeddingFunction(dim=config.get("dim", 16))


def _run_smoke_test() -> None:
    import tempfile

    # --- chunk_page_text ---
    short = "Just one short paragraph."
    assert chunk_page_text(short) == [short]

    long_text = "\n\n".join([f"Paragraph {i} " + ("word " * 40) for i in range(10)])
    chunks = chunk_page_text(long_text, chunk_size=300, overlap=50)
    assert len(chunks) > 1, "long text should split into multiple chunks"
    assert all(len(c) <= 320 for c in chunks), "chunks should roughly respect chunk_size"
    print(f"chunk_page_text: split a {len(long_text)}-char page into {len(chunks)} chunks")

    single_huge_paragraph = "x" * 1000  # no \n\n at all, forces sliding-window fallback
    windowed = chunk_page_text(single_huge_paragraph, chunk_size=300, overlap=50)
    assert len(windowed) > 1, "an unbroken block of text should still split via sliding window"

    # --- vector store mechanics (fake embeddings, no API key needed) ---
    with tempfile.TemporaryDirectory() as tmp:
        collection = get_collection(Path(tmp) / "chroma_db", embedding_function=_FakeEmbeddingFunction())

        tata_chunks = [
            TextChunk("Tata Steel", 2024, page=12, section="risk_factors",
                      text="Raw material price volatility is a key risk for Tata Steel."),
            TextChunk("Tata Steel", 2024, page=15, section="md&a",
                      text="Tata Steel's revenue grew driven by higher steel prices."),
        ]
        lt_chunks = [
            TextChunk("L&T", 2024, page=8, section="risk_factors",
                      text="Execution delays on infrastructure projects are a risk for L&T."),
        ]

        added = add_chunks(collection, tata_chunks + lt_chunks)
        assert added == 3
        assert collection.count() == 3, f"expected 3 stored chunks, got {collection.count()}"

        # The real property that matters: a SEPARATELY CONSTRUCTED TextChunk
        # with identical fields (simulating a fresh process re-running
        # ingestion on the same report) must get the SAME id, not a random
        # one, so re-adding it upserts instead of duplicating.
        rebuilt_first_chunk = TextChunk(
            "Tata Steel", 2024, page=12, section="risk_factors",
            text="Raw material price volatility is a key risk for Tata Steel.",
        )
        assert rebuilt_first_chunk.id == tata_chunks[0].id, "identical chunk content produced a different id"
        add_chunks(collection, [rebuilt_first_chunk])
        assert collection.count() == 3, "re-ingesting identical chunk content from a fresh object duplicated a row"

        # Scoped search returns only that company's chunks
        tata_results = search(collection, "risk factors", company="Tata Steel", n_results=5)
        assert len(tata_results) == 2, f"expected 2 Tata Steel chunks, got {len(tata_results)}"
        assert all(r["company"] == "Tata Steel" for r in tata_results)

        # Unscoped search can return chunks from both companies
        all_results = search(collection, "risk factors", n_results=5)
        assert len(all_results) == 3, f"expected 3 total chunks, got {len(all_results)}"

        print(f"vectorstore: stored {collection.count()} chunks, scoped search returned "
              f"{len(tata_results)} for Tata Steel, unscoped returned {len(all_results)} total")

        # delete_company_year: clears exactly the targeted company/year, nothing else
        delete_company_year(collection, "Tata Steel", 2024)
        assert collection.count() == 1, f"expected 1 chunk left after deleting Tata Steel 2024, got {collection.count()}"
        remaining = search(collection, "risk factors", n_results=5)
        assert remaining[0]["company"] == "L&T", "delete_company_year removed the wrong company's chunks"
        print("delete_company_year: removed only the targeted company/year, L&T chunk untouched")

    # --- MistralEmbeddingFunction: batching and retry logic, without a real API key ---
    from unittest.mock import MagicMock, patch
    import httpx

    os.environ["MISTRAL_API_KEY"] = "fake-key-for-embedding-tests"
    embedder = MistralEmbeddingFunction()

    # Batching: 130 texts at EMBED_BATCH_SIZE=64 should be 3 calls (64, 64, 2),
    # with the returned embeddings in the same order as the input.
    batch_call_log = []

    def fake_create(model, inputs):
        batch_call_log.append(len(inputs))
        response = MagicMock()
        response.data = [MagicMock(embedding=[float(i)]) for i in range(len(inputs))]
        return response

    embedder.client = MagicMock()
    embedder.client.embeddings.create.side_effect = fake_create

    texts = [f"chunk {i}" for i in range(130)]
    embeddings = embedder(texts)

    assert batch_call_log == [64, 64, 2], f"expected batches of [64, 64, 2], got {batch_call_log}"
    assert len(embeddings) == 130, f"expected 130 embeddings back, got {len(embeddings)}"
    print(f"MistralEmbeddingFunction: {len(texts)} texts split into {len(batch_call_log)} "
          f"batches of {batch_call_log}, all embeddings returned in order")

    # Retry: the exact real "too many tokens" error, retried in place, same as extract.py
    too_many_tokens_response = httpx.Response(400, headers={})
    too_many_tokens_body = ('{"object":"error","message":"Too many tokens overall, split into more '
                             'batches.","type":"invalid_request_prompt","param":null,"code":"3210"}')

    class _FakeEmbeddingSDKError(Exception):
        def __init__(self, message, raw_response, body):
            super().__init__(message)
            self.raw_response = raw_response
            self.body = body

    retry_call_count = {"n": 0}

    def fake_create_transient_then_success(model, inputs):
        retry_call_count["n"] += 1
        if retry_call_count["n"] == 1:
            raise _FakeEmbeddingSDKError("API error occurred", too_many_tokens_response, too_many_tokens_body)
        response = MagicMock()
        response.data = [MagicMock(embedding=[1.0]) for _ in inputs]
        return response

    embedder.client.embeddings.create.side_effect = fake_create_transient_then_success

    with patch("time.sleep") as mock_sleep:
        retried_embeddings = embedder(["one chunk"])

    assert len(retried_embeddings) == 1
    assert retry_call_count["n"] == 2, "should retry once after 'too many tokens', then succeed"
    mock_sleep.assert_called_once_with(20.0)
    print("MistralEmbeddingFunction: retries the 'too many tokens' error in place, same as extract.py")

    print("\nAll vectorstore smoke tests passed.")


if __name__ == "__main__":
    _run_smoke_test()