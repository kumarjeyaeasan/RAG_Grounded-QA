"""MP3 · Grounded Q&A — A Cited Document Assistant
==================================================

Extends my MP2 Mini-RAG (v4: 600/150 chunking, hybrid BM25 + dense retrieval with
weighted RRF and a same-story restriction, TOP_K=7) into a *trustworthy*
assistant. The four MP3 functions are implemented:

    format_context     (citations)   validate_citations (citations)
    should_abstain     (abstention)  evaluate           (evaluation)

  1. CITATIONS  — the answer cites each claim inline with [n], and every [n]
                  is validated against an actually-retrieved chunk.
  2. ABSTENTION — when retrieval finds no supporting chunk (top dense cosine
                  below RETRIEVAL_THRESHOLD), the assistant says
                  "I don't know based on the provided documents" without ever
                  calling the LLM.
  3. EVALUATION — a small harness measures retrieval hit rate, grounded rate,
                  abstention accuracy and hallucination rate over a question
                  set, and prints a table (the graded artifact).

Cost guard (the API key is a ~$1 Vocareum budget): every embedding and chat
completion is cached under .cache/, so re-running eval/sweep/inspect is free.
Each live command ends with an API-usage + estimated-cost line.

Run (from learner/, with .venv active — .env is loaded automatically):
    python mp3_grounded_qa.py selftest           # OFFLINE checks of the new logic (no keys, no network)
    python mp3_grounded_qa.py ingest             # build the Qdrant collection (once)
    python mp3_grounded_qa.py inspect            # retrieval scores per eval question (embeddings only, no chat)
    python mp3_grounded_qa.py ask                # interactive Q&A with citations + abstention
    python mp3_grounded_qa.py eval --out mp3_eval.txt   # the evaluation table (graded artifact)
    python mp3_grounded_qa.py sweep --out mp3_sweep.txt # Stretch: threshold trade-off table

Common options: --threshold 0.35   --no-cache   --dense-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

# ─── Configuration ──────────────────────────────────────────────────────

CORPUS_DIR         = Path(__file__).parent / "corpus"
DATA_DIR           = Path(__file__).parent / "data"
CACHE_DIR          = Path(__file__).parent / ".cache"
COLLECTION_NAME    = "mp3_sherlock"          # separate from MP2's 'sherlock_mini_rag'
EMBEDDING_MODEL    = "text-embedding-3-small"
EMBEDDING_DIM      = 1536
CHAT_MODEL         = "gpt-4o-mini"

# --- carried from MP2 v4 (see MP2_Reflection: each value was tuned against a diagnosed failure) ---
TARGET_CHUNK_SIZE  = 600    # characters. 600 keeps the 527-char Speckled Band "trained the snake" paragraph intact
CHUNK_OVERLAP      = 150    # characters. 60 orphaned "John Horner, a plumber" across a split
TOP_K              = 7      # chunks sent to the LLM
DENSE_CANDIDATES   = 12     # dense pool before fusion (q1's answer chunk ranked 10th)
BM25_CANDIDATES    = 12     # lexical pool before fusion
RRF_K              = 60     # standard Reciprocal Rank Fusion damping constant
DENSE_WEIGHT       = 1.5    # dense outweighs BM25 in fusion (equal weights lost q2's answer chunk)
BM25_WEIGHT        = 1.0

# --- MP3 ---
# Abstain if the top *dense cosine* (before fusion / story restriction) is below this.
# Tune via .env or --threshold; `inspect` and `sweep` show where the gap is.
# 0.30 (starter default) only gated the off-topic question (0.157); in-universe
# near-misses scored 0.449 (Toby) / 0.468 (Moriarty) and passed to the LLM. The lowest
# answerable question scored 0.540, so 0.50 puts the gate in front of all three
# should-abstain questions with a 0.04 margin; 0.55+ starts refusing q2/q6 (see mp3_sweep.txt).
RETRIEVAL_THRESHOLD = float(os.environ.get("RETRIEVAL_THRESHOLD", "0.50"))

ABSTAIN_MESSAGE = "I don't know based on the provided documents."

# Approximate list prices (USD per 1M tokens) — only used for the cost estimate line.
PRICE_EMBED_PER_M    = 0.02
PRICE_CHAT_IN_PER_M  = 0.15
PRICE_CHAT_OUT_PER_M = 0.60

# ─── Clients (lazy, so `selftest` never touches the network) ────────────
# MP2 fix carried forward: the openai v1 SDK only auto-reads OPENAI_BASE_URL, not
# OPENAI_API_BASE, so without an explicit base_url the Vocareum key would be sent
# to api.openai.com. python-dotenv loads .env on any OS (no `source .env`), and
# tolerates both `export VAR=...` and plain `VAR=...` lines.

_oai = None
_qdrant = None


def _use_windows_trust_store() -> None:
    """Make httpx (used by both openai and qdrant-client) trust the Windows root store too.

    httpx verifies TLS against certifi's bundle only. Antivirus HTTPS scanning (here: AVG
    Web/Mail Shield) re-signs every certificate with its own root, which Windows trusts but
    certifi doesn't → CERTIFICATE_VERIFY_FAILED. Building certifi + Windows ROOT certs into
    one bundle and pointing SSL_CERT_FILE at it keeps full verification on (nothing is
    disabled). Skipped off Windows, or if SSL_CERT_FILE is already set."""
    if sys.platform != "win32" or os.environ.get("SSL_CERT_FILE"):
        return
    import ssl
    import certifi
    pems = [Path(certifi.where()).read_text(encoding="utf-8")]
    for cert, encoding, trust in ssl.enum_certificates("ROOT"):
        # trust is True (any purpose) or a set of OIDs; 1.3.6.1.5.5.7.3.1 = server auth
        if encoding == "x509_asn" and (trust is True or "1.3.6.1.5.5.7.3.1" in trust):
            pems.append(ssl.DER_cert_to_PEM_cert(cert))
    bundle = CACHE_DIR / "ca-bundle.pem"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_text("\n".join(pems), encoding="utf-8")
    os.environ["SSL_CERT_FILE"] = str(bundle)


def _clients():
    """Build the OpenAI + Qdrant clients on first use; exit with the real cause on failure."""
    global _oai, _qdrant
    if _oai is not None and _qdrant is not None:
        return _oai, _qdrant
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent / ".env")
        _use_windows_trust_store()
        from openai import OpenAI
        from qdrant_client import QdrantClient
        base_url = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
        _oai = OpenAI(base_url=base_url) if base_url else OpenAI()
        # The brief pins qdrant-client <1.16; Qdrant Cloud now runs 1.19, so the client warns
        # on every start. The REST calls used here (query_points, upsert, ...) work fine.
        import warnings
        warnings.filterwarnings("ignore", message="Qdrant client version")
        _qdrant = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"),
                               timeout=int(os.environ.get("QDRANT_TIMEOUT", "30")))
    except Exception as e:
        sys.exit(f"Could not set up OpenAI/Qdrant clients ({type(e).__name__}: {e}).\n"
                 "Check learner/.env has OPENAI_API_KEY, OPENAI_API_BASE and QDRANT_URL. "
                 "(The `selftest` command needs neither.)")
    return _oai, _qdrant


# ─── Cost guard: local cache + usage counter ────────────────────────────

USE_CACHE = True
_usage = {"embed_calls": 0, "embed_tokens": 0, "chat_calls": 0,
          "chat_in": 0, "chat_out": 0, "cache_hits": 0}


def _cache_path(kind: str, payload: Any) -> Path:
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return CACHE_DIR / kind / f"{key}.json"


def _cache_get(kind: str, payload: Any) -> Any:
    if not USE_CACHE:
        return None
    p = _cache_path(kind, payload)
    if p.exists():
        _usage["cache_hits"] += 1
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _cache_put(kind: str, payload: Any, value: Any) -> None:
    p = _cache_path(kind, payload)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value), encoding="utf-8")


def usage_line() -> str:
    u = _usage
    cost = (u["embed_tokens"] * PRICE_EMBED_PER_M + u["chat_in"] * PRICE_CHAT_IN_PER_M
            + u["chat_out"] * PRICE_CHAT_OUT_PER_M) / 1_000_000
    return (f"  API usage this run: {u['embed_calls']} embed call(s) / {u['embed_tokens']} tok · "
            f"{u['chat_calls']} chat call(s) / {u['chat_in']} in + {u['chat_out']} out tok · "
            f"{u['cache_hits']} cache hit(s) · est. ${cost:.5f}")


def chat(messages: list[dict[str, str]]) -> str:
    """Temperature-0 chat completion, cached on (model, messages)."""
    payload = {"model": CHAT_MODEL, "temperature": 0, "messages": messages}
    cached = _cache_get("chat", payload)
    if cached is not None:
        return cached
    oai, _ = _clients()
    resp = oai.chat.completions.create(model=CHAT_MODEL, temperature=0, messages=messages)
    _usage["chat_calls"] += 1
    if resp.usage:
        _usage["chat_in"] += resp.usage.prompt_tokens
        _usage["chat_out"] += resp.usage.completion_tokens
    text = (resp.choices[0].message.content or "").strip()
    _cache_put("chat", payload, text)
    return text


# ─── Pipeline (carried from MP2 v4) ─────────────────────────────────────

def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[dict[str, Any]]:
    """Read every .txt file in the corpus directory (pure — no network).
    First non-empty line is the title."""
    docs = []
    for path in sorted(Path(corpus_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        title = next((ln.strip() for ln in text.splitlines() if ln.strip()), path.stem)
        docs.append({"source": path.name, "title": title, "text": text})
    if not docs:
        raise FileNotFoundError(f"No .txt files found in {corpus_dir}")
    return docs


_HEADING_RE = re.compile(r"^[^.!?\n]{1,60}$")   # short line, no terminal punctuation -> section heading


def _looks_like_heading(paragraph: str) -> bool:
    return bool(_HEADING_RE.match(paragraph.strip()))


def chunk_document(doc: dict[str, Any], chunk_size: int = TARGET_CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> list[dict[str, Any]]:
    """MP2 v4 chunker: paragraph packing up to chunk_size, heading-tracked
    `section` labels, sentence-boundary fallback for long paragraphs, and a
    character overlap carried from the previous chunk (pure — no network).

    Returns [{source, title, section, chunk_index, text}]."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc["text"]) if p.strip()]
    if paragraphs and paragraphs[0] == doc["title"]:
        paragraphs = paragraphs[1:]

    raw_chunks: list[tuple[str, str]] = []
    current_section = doc["title"]
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            raw_chunks.append((current_section, buf.strip()))
        buf = ""

    for para in paragraphs:
        if _looks_like_heading(para):
            flush()
            current_section = para
            continue
        if len(buf) + len(para) + 1 <= chunk_size:
            buf = f"{buf}\n{para}".strip() if buf else para
            continue
        flush()
        if len(para) <= chunk_size:
            buf = para
        else:
            for sent in re.split(r"(?<=[.!?])\s+", para):
                if len(buf) + len(sent) + 1 <= chunk_size:
                    buf = f"{buf} {sent}".strip() if buf else sent
                else:
                    flush()
                    buf = sent
    flush()

    chunks = []
    for idx, (section, text) in enumerate(raw_chunks):
        if idx > 0 and overlap > 0:
            text = f"{raw_chunks[idx - 1][1][-overlap:]} {text}"
        chunks.append({"source": doc["source"], "title": doc["title"], "section": section,
                       "chunk_index": idx, "text": text})
    return chunks


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Batch-embed, skipping any text already in the local cache."""
    out: list[list[float] | None] = [None] * len(texts)
    missing = []
    for i, t in enumerate(texts):
        cached = _cache_get("embed", {"model": EMBEDDING_MODEL, "input": t})
        if cached is not None:
            out[i] = cached
        else:
            missing.append(i)
    if missing:
        oai, _ = _clients()
        for start in range(0, len(missing), batch_size):
            idxs = missing[start:start + batch_size]
            resp = oai.embeddings.create(model=EMBEDDING_MODEL, input=[texts[i] for i in idxs])
            _usage["embed_calls"] += 1
            if resp.usage:
                _usage["embed_tokens"] += resp.usage.total_tokens
            for i, d in zip(idxs, resp.data):
                out[i] = d.embedding
                _cache_put("embed", {"model": EMBEDDING_MODEL, "input": texts[i]}, d.embedding)
    return out  # type: ignore[return-value]


def setup_collection() -> None:
    """Delete (if present) and create the collection (MP2's non-deprecated path)."""
    from qdrant_client.models import Distance, VectorParams
    _, qdrant = _clients()
    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)
    qdrant.create_collection(collection_name=COLLECTION_NAME,
                             vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))


def ingest_chunks(chunks: list[dict[str, Any]]) -> int:
    from qdrant_client.models import PointStruct
    _, qdrant = _clients()
    vectors = embed_texts([c["text"] for c in chunks])
    points = [PointStruct(id=i, vector=v, payload=c) for i, (c, v) in enumerate(zip(chunks, vectors))]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    return len(points)


def retrieve(query: str, k: int = DENSE_CANDIDATES) -> list[dict[str, Any]]:
    """Dense retrieval: embed the query, cosine-search Qdrant (query_points)."""
    _, qdrant = _clients()
    qvec = embed_texts([query])[0]
    hits = qdrant.query_points(collection_name=COLLECTION_NAME, query=qvec, limit=k).points
    return [{**h.payload, "score": round(float(h.score), 3), "via": "dense"} for h in hits]


# --- hybrid (BM25 + dense) — carried from MP2 v4 --------------------------
# The BM25 index is built lazily from local files (no API calls). It re-chunks the
# corpus with the same chunk_document() used for ingest, so chunks line up 1:1
# with Qdrant (matched on source + chunk_index).

_bm25_index = None
_bm25_chunks: list[dict[str, Any]] | None = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _build_bm25_index() -> None:
    global _bm25_index, _bm25_chunks
    if _bm25_index is not None:
        return
    from rank_bm25 import BM25Okapi
    _bm25_chunks = [c for doc in load_corpus() for c in chunk_document(doc)]
    _bm25_index = BM25Okapi([_tokenize(c["text"]) for c in _bm25_chunks])


def bm25_retrieve(query: str, k: int = BM25_CANDIDATES) -> list[dict[str, Any]]:
    """Lexical retrieval — local, free."""
    _build_bm25_index()
    scores = _bm25_index.get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{**_bm25_chunks[i], "bm25_score": float(scores[i])} for i in ranked]


def hybrid_retrieve(query: str, k: int = TOP_K,
                    dense_hits: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Weighted RRF over dense + BM25, both restricted to the story of the top
    dense hit (MP2 v4). `dense_hits` can be passed in so answer() reuses the
    dense search it already ran for the abstention gate — no second embedding.

    Every returned chunk keeps its dense cosine `score`; a chunk that only BM25
    surfaced has score=None (its cosine was outside the dense pool)."""
    if dense_hits is None:
        dense_hits = retrieve(query, k=DENSE_CANDIDATES)
    lexical_all = bm25_retrieve(query, k=BM25_CANDIDATES)

    top_source = dense_hits[0]["source"] if dense_hits else None
    dense = [h for h in dense_hits if h["source"] == top_source] if top_source else dense_hits
    lexical = [h for h in lexical_all if h["source"] == top_source] if top_source else lexical_all

    fused: dict[tuple, float] = {}
    pool: dict[tuple, dict[str, Any]] = {}
    for rank, h in enumerate(dense):
        key = (h["source"], h["chunk_index"])
        fused[key] = fused.get(key, 0.0) + DENSE_WEIGHT / (RRF_K + rank + 1)
        pool[key] = dict(h)
    for rank, h in enumerate(lexical):
        key = (h["source"], h["chunk_index"])
        fused[key] = fused.get(key, 0.0) + BM25_WEIGHT / (RRF_K + rank + 1)
        if key in pool:
            pool[key]["via"] = "both"
        else:
            pool[key] = {**h, "score": None, "via": "bm25"}

    ranked = sorted(fused, key=lambda key: fused[key], reverse=True)[:k]
    return [pool[key] for key in ranked]


# ─── NEW 1 · Citations ──────────────────────────────────────────────────

def format_context(chunks: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Number the retrieved chunks [1..n] for the prompt and return a parallel
    source map the answer's [n] markers can be validated against."""
    blocks, sources = [], []
    for n, c in enumerate(chunks, start=1):
        blocks.append(f'[{n}] (from "{c["title"]}" — {c["section"]})\n{c["text"]}')
        sources.append({"n": n, "source": c["source"], "title": c["title"],
                        "section": c["section"], "score": c.get("score"),
                        "via": c.get("via", "dense")})
    return "\n\n".join(blocks), sources


# Matches [3] and grouped forms like [1, 2] / [1,2,3] (models sometimes merge them).
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def validate_citations(answer_text: str, n_sources: int) -> dict[str, Any]:
    """Parse [n] markers from the answer and check each maps to a real source.
    A fabricated citation (a number with no matching source) is the tell we
    care about. Lists are de-duplicated, in order of first appearance."""
    used: list[int] = []
    invalid: list[int] = []
    for group in _CITATION_RE.findall(answer_text):
        for num in (int(x) for x in group.split(",")):
            bucket = used if 1 <= num <= n_sources else invalid
            if num not in bucket:
                bucket.append(num)
    return {"used": used, "invalid": invalid, "has_citation": bool(used)}


# ─── NEW 2 · Abstention ─────────────────────────────────────────────────

def should_abstain(chunks: list[dict[str, Any]], threshold: float = RETRIEVAL_THRESHOLD) -> bool:
    """Abstain when there is no sufficiently-relevant chunk to ground an answer:
    no chunks, or the best cosine score is below `threshold`. Chunks without a
    cosine score (BM25-only) can't vouch for relevance, so they're ignored."""
    scores = [c["score"] for c in chunks if c.get("score") is not None]
    return not scores or max(scores) < threshold


SYSTEM_PROMPT = """You answer questions about a small collection of Sherlock Holmes stories.
Use ONLY the numbered excerpts provided below. After each claim, cite the excerpt it
came from using its number in square brackets, e.g. [1]. Only cite numbers that appear
in the excerpts. If the excerpts do not contain the answer, reply with exactly:
"I don't know based on the provided documents." Do not use any outside knowledge.

Be careful with character identity: when the excerpts mention more than one person in a
similar role (e.g. more than one hotel employee, suspect, or servant), re-check which
specific name matches which fact — such as who was accused versus who gave evidence, or
who is a disguise versus a real identity — before attributing an action to a name. Do
not swap or merge two different characters."""

_PROSE_ABSTAIN_RE = re.compile(r"^\W*i (?:don't|do not) know\b", re.IGNORECASE)


def _is_prose_abstention(text: str) -> bool:
    """The model refused in its own words (exact sentence, or opening with 'I don't know')."""
    t = text.replace("’", "'").strip()
    return ABSTAIN_MESSAGE.lower() in t.lower() or bool(_PROSE_ABSTAIN_RE.match(t))


def _abstain_result(question: str, retrieved_sources, top_score, t0, reason: str) -> dict[str, Any]:
    return {"question": question, "answer": ABSTAIN_MESSAGE, "citations": [],
            "retrieved_sources": retrieved_sources, "top_score": top_score,
            "abstained": True, "abstain_reason": reason,
            "citation_check": {"used": [], "invalid": [], "has_citation": False},
            "latency_ms": int((time.time() - t0) * 1000)}


def answer(question: str, k: int = TOP_K, threshold: float = RETRIEVAL_THRESHOLD,
           use_hybrid: bool = True) -> dict[str, Any]:
    """Dense search (gate) → hybrid fusion → (abstain?) → cited generation → validate.

    The gate uses the top dense cosine across the whole dense pool, *before* the
    same-story restriction — the honest "is anything in the corpus relevant?"
    signal. RRF fused scores are rank-based, so they can't be thresholded."""
    t0 = time.time()
    dense = retrieve(question, k=max(DENSE_CANDIDATES, k))
    chunks = hybrid_retrieve(question, k=k, dense_hits=dense) if use_hybrid else dense[:k]
    top_score = dense[0]["score"] if dense else None
    retrieved_sources = [{"source": c["source"], "score": c.get("score"), "via": c.get("via")}
                         for c in chunks]

    if should_abstain(dense, threshold):
        return _abstain_result(question, retrieved_sources, top_score, t0, "score gate")

    context, sources = format_context(chunks)
    text = chat([{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {question}"}])

    # A model can ignore instructions and abstain in prose — respect that.
    if _is_prose_abstention(text):
        return _abstain_result(question, retrieved_sources, top_score, t0, "model")

    check = validate_citations(text, len(sources))
    cited = [s for s in sources if s["n"] in check["used"]]
    return {"question": question, "answer": text, "citations": cited,
            "retrieved_sources": retrieved_sources, "top_score": top_score,
            "abstained": False, "abstain_reason": None,
            "citation_check": check, "latency_ms": int((time.time() - t0) * 1000)}


# ─── NEW 3 · Evaluation harness ─────────────────────────────────────────

def _is_should_abstain(q: dict[str, Any]) -> bool:
    return bool(q.get("should_abstain")) or q.get("expected_source") in (None, "", "ABSTAIN")


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 2) if den else None


def _facts_matched(q: dict[str, Any], answer_text: str) -> str | None:
    """MP2's literal-substring fact check, reported as 'hit/total' (diagnostic only)."""
    facts = q.get("expected_facts") or []
    if not facts:
        return None
    low = answer_text.lower()
    return f"{sum(f.lower() in low for f in facts)}/{len(facts)}"


def evaluate(questions: list[dict[str, Any]],
             answer_fn: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    """Score an assistant over a question set. Each question is either
    answerable (has `expected_source`) or a should-abstain question
    (`should_abstain: true` or `expected_source: null`).

    Metrics (demonstrating grounded-response and hallucination KPIs):
      - retrieval hit rate    : answerable Qs whose expected source is in top-K
      - grounded & cited rate  : answerable Qs answered, citing the right source, no bad [n]
      - abstention accuracy    : should-abstain Qs the assistant correctly refused
      - hallucination rate     : should-abstain Qs the assistant answered anyway
    Extra (diagnostic): false-abstention rate — answerable Qs that were refused.
    """
    rows = []
    n_ans = n_abs = hits = grounded_n = abst_ok = hallucinated = false_abst = 0

    for q in questions:
        r = answer_fn(q["question"])
        check = r["citation_check"]
        cited_sources = sorted({c["source"] for c in r["citations"]})
        row = {"id": q.get("id", q["question"][:30]), "abstained": r["abstained"],
               "top_score": r.get("top_score"), "facts": _facts_matched(q, r["answer"]),
               "cited": cited_sources, "invalid": check["invalid"],
               "abstain_reason": r.get("abstain_reason")}

        if _is_should_abstain(q):
            n_abs += 1
            ok = bool(r["abstained"])
            abst_ok += ok
            hallucinated += not ok
            row.update(type="abstain", hit=None, grounded=None, ok=ok)
        else:
            n_ans += 1
            expected = q["expected_source"]
            hit = expected in {s["source"] for s in r["retrieved_sources"]}
            grounded = (not r["abstained"] and check["has_citation"] and not check["invalid"]
                        and expected in cited_sources)
            hits += hit
            grounded_n += grounded
            false_abst += bool(r["abstained"])
            row.update(type="answer", hit=hit, grounded=grounded, ok=grounded)
        rows.append(row)

    return {"rows": rows, "summary": {
        "n_answerable": n_ans, "n_abstain": n_abs,
        "retrieval_hit_rate": _rate(hits, n_ans),
        "grounded_rate": _rate(grounded_n, n_ans),
        "abstention_accuracy": _rate(abst_ok, n_abs),
        "hallucination_rate": _rate(hallucinated, n_abs),
        "false_abstention_rate": _rate(false_abst, n_ans),
    }}


def format_eval_table(result: dict[str, Any]) -> str:
    """Render the evaluation as the submittable table (+ a diagnostics block)."""
    def cell(v):
        return {True: "✓", False: "✗", None: "—"}.get(v, str(v))

    def rate(v):
        return "n/a" if v is None else f"{v:.2f}"

    lines = ["", "  id                              type     hit   abstained  grounded  ok",
             "  " + "-" * 74]
    for r in result["rows"]:
        lines.append(f"  {r['id']:<31} {r['type']:<8} {cell(r['hit']):<5} "
                     f"{cell(r['abstained']):<10} {cell(r['grounded']):<9} {cell(r['ok'])}")
    s = result["summary"]
    lines += [
        "  " + "-" * 74, "",
        f"  Retrieval hit rate    : {rate(s['retrieval_hit_rate'])}   "
        f"({s['n_answerable']} answerable questions)",
        f"  Grounded & cited rate : {rate(s['grounded_rate'])}   (KPI: grounded response rate)",
        f"  Abstention accuracy   : {rate(s['abstention_accuracy'])}   "
        f"({s['n_abstain']} should-abstain questions)",
        f"  Hallucination rate    : {rate(s['hallucination_rate'])}   (KPI: hallucination rate)",
        "",
    ]

    # Diagnostics — not part of the graded table, but it's what justifies the threshold.
    if any("top_score" in r for r in result["rows"]):
        lines += ["  Diagnostics", "  " + "-" * 74,
                  "  id                              top cos  facts  cited / why abstained",
                  "  " + "-" * 74]
        for r in result["rows"]:
            ts = "—" if r.get("top_score") is None else f"{r['top_score']:.3f}"
            if r["abstained"]:
                detail = f"abstained ({r.get('abstain_reason') or '?'})"
            else:
                detail = ", ".join(r.get("cited") or []) or "(no valid citation)"
                if r.get("invalid"):
                    detail += f"  ⚠ fabricated {r['invalid']}"
            lines.append(f"  {r['id']:<31} {ts:<8} {r.get('facts') or '—':<6} {detail}")
        if "false_abstention_rate" in s:
            lines += ["  " + "-" * 74,
                      f"  False-abstention rate : {rate(s['false_abstention_rate'])}   "
                      "(answerable questions wrongly refused)"]
        lines.append("")
    return "\n".join(lines)


# ─── Commands ───────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _is_template(q: dict[str, Any]) -> bool:
    return str(q.get("question", "")).strip().lower().startswith(
        ("replace this", "your second", "your unanswerable"))


def load_questions() -> list[dict[str, Any]]:
    """Predefined questions + the learner's own (un-authored templates skipped)."""
    questions = _load_jsonl(DATA_DIR / "predefined_questions.jsonl")
    learner = DATA_DIR / "learner_questions.jsonl"
    if learner.exists():
        questions += [q for q in _load_jsonl(learner) if not _is_template(q)]
    return questions


def _config_line(threshold: float, use_hybrid: bool) -> str:
    mode = (f"hybrid BM25+dense RRF ({DENSE_WEIGHT}/{BM25_WEIGHT}), same-story"
            if use_hybrid else "dense only")
    return (f"  Config: threshold={threshold:.2f} (top dense cosine) · top_k={TOP_K} · {mode} · "
            f"chunks {TARGET_CHUNK_SIZE}/{CHUNK_OVERLAP} · {CHAT_MODEL} + {EMBEDDING_MODEL}")


def _emit(text: str, out: str | None) -> None:
    print(text)
    if out:
        Path(out).write_text(text + "\n", encoding="utf-8")
        print(f"  → written to {out} (UTF-8)")


def cmd_ingest(args) -> None:
    print("→ Loading corpus…")
    all_chunks: list[dict[str, Any]] = []
    for doc in load_corpus(CORPUS_DIR):
        cs = chunk_document(doc)
        all_chunks.extend(cs)
        print(f"  {doc['source']}: {len(cs)} chunks")
    print(f"→ {len(all_chunks)} chunks · setting up '{COLLECTION_NAME}' · ingesting…")
    setup_collection()
    n = ingest_chunks(all_chunks)
    print(f"✓ Ingested {n} chunks. Try: python mp3_grounded_qa.py inspect")


def _print_result(r: dict[str, Any]) -> None:
    print(f"\n{r['answer']}\n")
    ts = "—" if r.get("top_score") is None else f"{r['top_score']:.3f}"
    if r["abstained"]:
        print(f"  (abstained — {r.get('abstain_reason')}; top dense cosine {ts})")
    else:
        print(f"  Sources (top dense cosine {ts}):")
        for c in r["citations"]:
            sc = c["via"] if c.get("score") is None else f"score {c['score']}"
            print(f"    [{c['n']}] {c['title']} — {c['section']}  ({sc})")
        if r["citation_check"]["invalid"]:
            print(f"  ⚠ fabricated citation(s): {r['citation_check']['invalid']}")
    print(f"  Latency: {r['latency_ms']}ms\n")


def cmd_ask(args) -> None:
    print("Grounded Q&A over the Sherlock corpus. Empty line to exit.\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not q:
            break
        _print_result(answer(q, threshold=args.threshold, use_hybrid=not args.dense_only))


def cmd_inspect(args) -> None:
    """Retrieval only — embeddings, no chat calls. Shows where the abstention gap is."""
    questions = load_questions()
    print(f"\n  Retrieval inspection — {len(questions)} questions · gate at {args.threshold:.2f}\n")
    for q in questions:
        dense = retrieve(q["question"], k=DENSE_CANDIDATES)
        top = dense[0]["score"] if dense else None
        kind = "ABSTAIN" if _is_should_abstain(q) else f"expect {q['expected_source']}"
        gate = "abstain" if should_abstain(dense, args.threshold) else "answer"
        print(f"  {q['id']}  [{kind}]  top cos {top}  → gate: {gate}")
        for h in dense[:3]:
            print(f"      dense {h['score']:.3f}  {h['source']:<26} {h['section']}")
        if not args.dense_only:
            fused = hybrid_retrieve(q["question"], k=TOP_K, dense_hits=dense)
            print("      hybrid top-k: " + ", ".join(
                f"{c['source'][:2]}#{c['chunk_index']}({c['via']})" for c in fused))
        print()


def cmd_eval(args) -> None:
    questions = load_questions()
    result = evaluate(questions, lambda q: answer(q, threshold=args.threshold,
                                                  use_hybrid=not args.dense_only))
    _emit(_config_line(args.threshold, not args.dense_only) + "\n" + format_eval_table(result), args.out)


def _apply_gate(r: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Re-decide the score gate for an ungated answer (no API call)."""
    if r["abstained"] or r.get("top_score") is None or r["top_score"] >= threshold:
        return r
    return {**r, "answer": ABSTAIN_MESSAGE, "citations": [], "abstained": True,
            "abstain_reason": "score gate",
            "citation_check": {"used": [], "invalid": [], "has_citation": False}}


def cmd_sweep(args) -> None:
    """Stretch: sweep the threshold. Each question is answered ONCE with the gate
    off (cached), then every threshold is simulated offline from those results."""
    questions = load_questions()
    ungated = {q["question"]: answer(q["question"], threshold=float("-inf"),
                                     use_hybrid=not args.dense_only) for q in questions}
    thresholds = [float(t) for t in args.thresholds.split(",")]
    lines = [_config_line(args.threshold, not args.dense_only).replace(
                 f"threshold={args.threshold:.2f}", "threshold=SWEPT"), "",
             "  thresh  hit rate  grounded  abst.acc  halluc.  false-abst",
             "  " + "-" * 58]
    for t in thresholds:
        s = evaluate(questions, lambda q: _apply_gate(ungated[q], t))["summary"]
        f = lambda v: "n/a" if v is None else f"{v:.2f}"
        lines.append(f"  {t:<6.2f}  {f(s['retrieval_hit_rate']):<8}  {f(s['grounded_rate']):<8}  "
                     f"{f(s['abstention_accuracy']):<8}  {f(s['hallucination_rate']):<7}  "
                     f"{f(s['false_abstention_rate'])}")
    lines += ["", "  Top dense cosine per question (what the gate compares against):"]
    for q in questions:
        r = ungated[q["question"]]
        kind = "abstain" if _is_should_abstain(q) else "answer "
        note = "  (LLM refused on its own)" if r["abstained"] else ""
        lines.append(f"    {kind}  {r['top_score']:.3f}  {q['id']}{note}")
    _emit("\n".join(lines) + "\n", args.out)


def cmd_selftest(args) -> None:
    """Offline checks of the MP3 logic — no OpenAI, no Qdrant, no keys, no network."""
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'✓' if cond else '✗'} {name}")
        ok = ok and cond

    docs = load_corpus(CORPUS_DIR)
    check(f"load_corpus reads the corpus ({len(docs)} docs)", len(docs) >= 1)
    if docs:
        cs = chunk_document(docs[0])
        check(f"chunk_document splits doc 0 into chunks ({len(cs)})",
              len(cs) >= 1 and all({"source", "title", "section", "text"} <= c.keys() for c in cs))

    fake = [{"source": "a.txt", "title": "A", "section": "S1", "text": "alpha", "score": 0.9},
            {"source": "b.txt", "title": "B", "section": "S2", "text": "beta", "score": 0.5}]
    ctx, srcs = format_context(fake)
    check("format_context numbers chunks [1]/[2]", "[1]" in ctx and "[2]" in ctx and len(srcs) == 2)

    v_ok = validate_citations("The adder [1] came via the ventilator [2].", 2)
    v_bad = validate_citations("An outside fact [3].", 2)
    check("validate_citations accepts real markers", v_ok["used"] == [1, 2] and not v_ok["invalid"])
    check("validate_citations flags a fabricated marker", v_bad["invalid"] == [3])

    check("should_abstain: low score → abstain", should_abstain([{"score": 0.1}], 0.3) is True)
    check("should_abstain: good score → answer", should_abstain([{"score": 0.8}], 0.3) is False)
    check("should_abstain: no chunks → abstain", should_abstain([], 0.3) is True)

    # evaluate() with a deterministic mock answer_fn (no network)
    qs = [
        {"id": "ans_hit", "question": "q1", "expected_source": "01.txt"},
        {"id": "ans_miss", "question": "q2", "expected_source": "02.txt"},
        {"id": "abstain_ok", "question": "q3", "should_abstain": True},
    ]
    def mock(q):
        if q == "q1":  # correct source retrieved + cited
            return {"answer": "A [1]", "abstained": False, "retrieved_sources": [{"source": "01.txt", "score": 0.9}],
                    "citations": [{"source": "01.txt"}], "citation_check": {"used": [1], "invalid": [], "has_citation": True}}
        if q == "q2":  # wrong source retrieved → miss + not grounded
            return {"answer": "B [1]", "abstained": False, "retrieved_sources": [{"source": "99.txt", "score": 0.9}],
                    "citations": [{"source": "99.txt"}], "citation_check": {"used": [1], "invalid": [], "has_citation": True}}
        return {"answer": ABSTAIN_MESSAGE, "abstained": True, "retrieved_sources": [{"source": "x", "score": 0.1}],
                "citations": [], "citation_check": {"used": [], "invalid": [], "has_citation": False}}
    res = evaluate(qs, mock)
    s = res["summary"]
    check("evaluate: hit rate 1/2 answerable", s["retrieval_hit_rate"] == 0.5)
    check("evaluate: grounded rate 1/2 answerable", s["grounded_rate"] == 0.5)
    check("evaluate: abstention accuracy 1/1", s["abstention_accuracy"] == 1.0)
    check("evaluate: hallucination rate 0/1", s["hallucination_rate"] == 0.0)
    check("format_eval_table renders a table", "Retrieval hit rate" in format_eval_table(res))

    # --- MP3-extension checks (still offline) ---
    v_grp = validate_citations("Both [1, 3] and again [1] plus [0].", 2)
    check("validate_citations: grouped [1, 3] + [0] → used [1], invalid [3, 0]",
          v_grp["used"] == [1] and v_grp["invalid"] == [3, 0])
    check("should_abstain ignores BM25-only chunks (score None)",
          should_abstain([{"score": None}], 0.3) is True
          and should_abstain([{"score": None}, {"score": 0.6}], 0.3) is False)
    check("prose abstention detected ('I don’t know …')",
          _is_prose_abstention("I don’t know — the excerpts don't say.")
          and not _is_prose_abstention("Holmes did not know at first that [1]..."))
    empty = evaluate([], mock)["summary"]
    check("evaluate: empty question set → rates n/a, no ZeroDivisionError",
          empty["retrieval_hit_rate"] is None and "n/a" in format_eval_table({"rows": [], "summary": empty}))
    gated = _apply_gate({"answer": "x [1]", "abstained": False, "top_score": 0.25, "citations": [{}],
                         "citation_check": {"used": [1], "invalid": [], "has_citation": True}}, 0.3)
    check("sweep gate simulation abstains below threshold", gated["abstained"] and not gated["citations"])
    all_chunks = [c for d in docs for c in chunk_document(d)]
    check(f"chunker parity with MP2 v4 (72 chunks; got {len(all_chunks)})", len(all_chunks) == 72)
    _build_bm25_index()
    top = bm25_retrieve("Victor Hatherley hydraulic press counterfeit", k=1)[0]
    check(f"BM25 index builds locally (top hit {top['source']})", top["source"] == "04_engineers_thumb.txt")
    check("no API calls were made", _usage["embed_calls"] == 0 and _usage["chat_calls"] == 0)

    print("\n✓ selftest passed" if ok else "\n✗ selftest FAILED")
    sys.exit(0 if ok else 1)


def main() -> None:
    try:  # Windows consoles default to cp1252, which can't print ✓ ✗ → —
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="MP3 Grounded Q&A over the Sherlock corpus",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["selftest", "ingest", "inspect", "ask", "eval", "sweep"])
    parser.add_argument("--threshold", type=float, default=RETRIEVAL_THRESHOLD,
                        help=f"abstention gate on top dense cosine (default {RETRIEVAL_THRESHOLD})")
    parser.add_argument("--dense-only", action="store_true", help="skip BM25 fusion (A/B vs. hybrid)")
    parser.add_argument("--no-cache", action="store_true", help="force fresh API calls")
    parser.add_argument("--out", help="also write the output to this file (UTF-8)")
    parser.add_argument("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50",
                        help="comma-separated thresholds for `sweep`")
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    args = parser.parse_args()

    global USE_CACHE
    USE_CACHE = not args.no_cache
    cmds = {"selftest": cmd_selftest, "ingest": cmd_ingest, "inspect": cmd_inspect,
            "ask": cmd_ask, "eval": cmd_eval, "sweep": cmd_sweep}
    try:
        cmds[args.command](args)
    finally:
        if args.command != "selftest" and (any(_usage.values())):
            print(usage_line())


if __name__ == "__main__":
    main()
