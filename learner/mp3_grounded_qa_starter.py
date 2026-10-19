"""MP3 · Grounded Q&A — Starter
================================

The load → chunk → embed → store → retrieve → answer pipeline is PROVIDED and
working (it's the MP2 loop). Your job is the three things that make it
*trustworthy*. Fill in the FOUR functions marked TODO:

    format_context     (citations)   validate_citations (citations)
    should_abstain     (abstention)  evaluate           (evaluation)

Everything else — the pipeline, the prompt, `answer()`, the table renderer, the
CLI — is done for you. Once your four functions are in, `answer()` and `eval`
will work end to end.

Check your work offline (no API key needed):
    python mp3_grounded_qa.py selftest      # goes green once your 4 functions are correct

Under the hood this is the MP2 Mini-RAG pipeline extended into a *trustworthy*
assistant:

  1. CITATIONS  — the answer cites each claim inline with [n], and every [n]
                  is validated against an actually-retrieved chunk.
  2. ABSTENTION — when retrieval finds no supporting chunk, the assistant says
                  "I don't know based on the provided documents" instead of
                  guessing.
  3. EVALUATION — a small harness measures retrieval hit rate, grounded rate,
                  and abstention accuracy over a question set, and prints a
                  table (this is the graded artifact).

The load → chunk → embed → store → retrieve loop is carried from MP2. The three
capabilities above are the MP3 additions.

Run:
    pip install -r requirements.txt
    source .env
    python mp3_grounded_qa.py ingest      # build the collection (once)
    python mp3_grounded_qa.py ask         # interactive Q&A with citations + abstention
    python mp3_grounded_qa.py eval        # the retrieval-quality evaluation table
    python mp3_grounded_qa.py selftest    # OFFLINE checks of the new logic (no keys)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

# ─── Configuration ──────────────────────────────────────────────────────

CORPUS_DIR         = Path(__file__).parent / "corpus"
DATA_DIR           = Path(__file__).parent / "data"
COLLECTION_NAME    = "mp3_sherlock"
EMBEDDING_MODEL    = "text-embedding-3-small"
EMBEDDING_DIM      = 1536
CHAT_MODEL         = "gpt-4o-mini"
TARGET_CHUNK_SIZE  = 500    # characters
CHUNK_OVERLAP      = 80     # characters
RETRIEVAL_THRESHOLD = float(os.environ.get("RETRIEVAL_THRESHOLD", "0.30"))  # abstain if top chunk scores below this (cosine); tune via .env

ABSTAIN_MESSAGE = "I don't know based on the provided documents."

# ─── Clients (guarded so the pure logic + selftest import with no creds) ──
# The pipeline commands (ingest/ask/eval) need real credentials; the pure
# citation/abstention/eval logic and `selftest` do not.
try:
    from openai import OpenAI
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    oai = OpenAI()
    qdrant = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"),
                          timeout=int(os.environ.get("QDRANT_TIMEOUT", "30")))
except Exception:
    oai = qdrant = None


def _require_clients() -> None:
    if oai is None or qdrant is None:
        sys.exit("This command needs OPENAI_API_KEY and QDRANT_URL set. "
                 "Run `source .env` first. (The `selftest` command needs neither.)")


# ─── Pipeline (carried from MP2) ────────────────────────────────────────

def load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    """Read every .txt file in the corpus directory (pure — no network)."""
    docs = []
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        title = next((ln.strip() for ln in text.splitlines() if ln.strip()), path.stem)
        docs.append({"source": path.name, "title": title, "text": text})
    return docs


def _is_header(line: str) -> bool:
    """Heuristic: short lines with no terminal punctuation are section headers."""
    s = line.strip()
    return 0 < len(s) < 60 and not s.endswith((".", "!", "?", ":", ",", ";"))


def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Paragraph-aware chunking with heuristic section-header tracking (pure)."""
    chunks: list[dict[str, Any]] = []
    section = doc["title"]
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append({"source": doc["source"], "title": doc["title"],
                           "section": section, "text": buf.strip()})
        buf = ""

    for para in doc["text"].split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if _is_header(para):
            flush()
            section = para
            continue
        if len(buf) + len(para) + 1 > TARGET_CHUNK_SIZE:
            flush()
            # carry a little overlap from the previous chunk
            if chunks and CHUNK_OVERLAP:
                buf = chunks[-1]["text"][-CHUNK_OVERLAP:] + " "
        buf += para + " "
    flush()
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = oai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def setup_collection() -> None:
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


def ingest_chunks(chunks: list[dict[str, Any]]) -> None:
    vectors = embed_texts([c["text"] for c in chunks])
    points = [PointStruct(id=str(uuid.uuid4()), vector=v, payload=c)
              for c, v in zip(chunks, vectors)]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    qvec = embed_texts([query])[0]
    hits = qdrant.search(collection_name=COLLECTION_NAME, query_vector=qvec, limit=k)
    out = []
    for h in hits:
        chunk = dict(h.payload)
        chunk["score"] = round(float(h.score), 3)
        out.append(chunk)
    return out


# ─── NEW 1 · Citations ──────────────────────────────────────────────────

def format_context(chunks: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Number the retrieved chunks [1..n] for the prompt and return a parallel
    source map the answer's [n] markers can be validated against."""
    # TODO (citations): build two things and return them as a tuple.
    #   1. a context string where each chunk is prefixed with its number, e.g.
    #        [1] (from "<title>" — <section>)
    #        <chunk text>
    #      (blank line between chunks)
    #   2. a `sources` list of dicts, one per chunk, each with keys:
    #        n, source, title, section, score
    #      (parallel to the numbering — [1] is sources[0], etc.)
    raise NotImplementedError("Implement format_context")


def validate_citations(answer_text: str, n_sources: int) -> dict[str, Any]:
    """Parse [n] markers from the answer and check each maps to a real source.
    A fabricated citation (a number with no matching source) is the tell we
    care about."""
    # TODO (citations): find every [n] marker in answer_text (regex on \[(\d+)\]),
    # then split the numbers into:
    #   used    : the valid ones (1 <= n <= n_sources)
    #   invalid : the out-of-range ones (fabricated citations)
    # Return {"used": [...], "invalid": [...], "has_citation": bool(used)}.
    raise NotImplementedError("Implement validate_citations")


# ─── NEW 2 · Abstention ─────────────────────────────────────────────────

def should_abstain(chunks: list[dict[str, Any]], threshold: float = RETRIEVAL_THRESHOLD) -> bool:
    """Abstain when there is no sufficiently-relevant chunk to ground an answer."""
    # TODO (abstention): return True (abstain) when there are no chunks, OR when
    # the highest chunk score is below `threshold`. Otherwise return False.
    raise NotImplementedError("Implement should_abstain")


SYSTEM_PROMPT = """You answer questions about a small collection of Sherlock Holmes stories.
Use ONLY the numbered excerpts provided below. After each claim, cite the excerpt it
came from using its number in square brackets, e.g. [1]. If the excerpts do not contain
the answer, reply with exactly: "I don't know based on the provided documents." Do not
use any outside knowledge."""


def answer(question: str, k: int = 3, threshold: float = RETRIEVAL_THRESHOLD) -> dict[str, Any]:
    """Retrieve → (abstain?) → cite-formatted generation → validate citations."""
    t0 = time.time()
    chunks = retrieve(question, k=k)
    retrieved_sources = [{"source": c["source"], "score": c.get("score", 0.0)} for c in chunks]

    if should_abstain(chunks, threshold):
        return {"question": question, "answer": ABSTAIN_MESSAGE, "citations": [],
                "retrieved_sources": retrieved_sources, "abstained": True,
                "citation_check": {"used": [], "invalid": [], "has_citation": False},
                "latency_ms": int((time.time() - t0) * 1000)}

    context, sources = format_context(chunks)
    resp = oai.chat.completions.create(
        model=CHAT_MODEL, temperature=0,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {question}"}],
    )
    text = resp.choices[0].message.content.strip()

    # A model can ignore instructions and abstain in prose — respect that.
    if ABSTAIN_MESSAGE.lower() in text.lower():
        return {"question": question, "answer": ABSTAIN_MESSAGE, "citations": [],
                "retrieved_sources": retrieved_sources, "abstained": True,
                "citation_check": {"used": [], "invalid": [], "has_citation": False},
                "latency_ms": int((time.time() - t0) * 1000)}

    check = validate_citations(text, len(sources))
    cited = [s for s in sources if s["n"] in check["used"]]
    return {"question": question, "answer": text, "citations": cited,
            "retrieved_sources": retrieved_sources, "abstained": False,
            "citation_check": check, "latency_ms": int((time.time() - t0) * 1000)}


# ─── NEW 3 · Evaluation harness ─────────────────────────────────────────

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
    """
    rows = []
    # TODO (evaluation): score the assistant over `questions`.
    #
    # For each question q, call r = answer_fn(q["question"]).
    #   A question is "should-abstain" if:
    #       q.get("should_abstain")  OR  q.get("expected_source") in (None, "", "ABSTAIN")
    #   Otherwise it is "answerable" and has a real q["expected_source"].
    #
    # Over ANSWERABLE questions, track:
    #   hits      → q["expected_source"] appears in the sources of r["retrieved_sources"]
    #               (this is the retrieval hit rate)
    #   grounded  → (not r["abstained"]) AND r["citation_check"]["has_citation"]
    #               AND not r["citation_check"]["invalid"]
    #               AND q["expected_source"] is among {c["source"] for c in r["citations"]}
    #
    # Over SHOULD-ABSTAIN questions, track:
    #   abst_ok       → r["abstained"] is True   (abstention accuracy)
    #   hallucinated  → r["abstained"] is False  (hallucination rate)
    #
    # Append a per-question dict to `rows` (see format_eval_table for the keys it
    # renders: id, type, hit, abstained, grounded, ok) and return:
    #   {"rows": rows, "summary": {
    #       "n_answerable", "n_abstain",
    #       "retrieval_hit_rate", "grounded_rate",
    #       "abstention_accuracy", "hallucination_rate"}}
    raise NotImplementedError("Implement evaluate")


def format_eval_table(result: dict[str, Any]) -> str:
    """Render the evaluation as the submittable table."""
    def cell(v):
        return {True: "✓", False: "✗"}.get(v, str(v))

    lines = ["", "  id                              type     hit   abstained  grounded  ok",
             "  " + "-" * 74]
    for r in result["rows"]:
        lines.append(f"  {r['id']:<31} {r['type']:<8} {cell(r['hit']):<5} "
                     f"{cell(r['abstained']):<10} {cell(r['grounded']):<9} {cell(r['ok'])}")
    s = result["summary"]
    lines += [
        "  " + "-" * 74, "",
        f"  Retrieval hit rate    : {s['retrieval_hit_rate']}   "
        f"({s['n_answerable']} answerable questions)",
        f"  Grounded & cited rate : {s['grounded_rate']}   (KPI: grounded response rate)",
        f"  Abstention accuracy   : {s['abstention_accuracy']}   "
        f"({s['n_abstain']} should-abstain questions)",
        f"  Hallucination rate    : {s['hallucination_rate']}   (KPI: hallucination rate)",
        "",
    ]
    return "\n".join(lines)


# ─── Commands ───────────────────────────────────────────────────────────

def cmd_ingest() -> None:
    _require_clients()
    print("→ Loading corpus…")
    docs = load_corpus(CORPUS_DIR)
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        cs = chunk_document(doc)
        all_chunks.extend(cs)
        print(f"  {doc['source']}: {len(cs)} chunks")
    print(f"→ {len(all_chunks)} chunks · setting up collection · ingesting…")
    setup_collection()
    ingest_chunks(all_chunks)
    print("✓ Done. Try: python mp3_grounded_qa.py ask")


def _print_result(r: dict[str, Any]) -> None:
    print(f"\n{r['answer']}\n")
    if r["abstained"]:
        print("  (abstained — no supporting excerpt)")
    else:
        print("  Sources:")
        for c in r["citations"]:
            print(f"    [{c['n']}] {c['title']} — {c['section']}  (score {c['score']})")
        if r["citation_check"]["invalid"]:
            print(f"  ⚠ fabricated citation(s): {r['citation_check']['invalid']}")
    print(f"  Latency: {r['latency_ms']}ms\n")


def cmd_ask() -> None:
    _require_clients()
    print("Grounded Q&A over the Sherlock corpus. Empty line to exit.\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return
        if not q:
            return
        _print_result(answer(q))


def cmd_eval() -> None:
    _require_clients()
    path = DATA_DIR / "predefined_questions.jsonl"
    questions = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    learner = DATA_DIR / "learner_questions.jsonl"
    if learner.exists():
        extra = [json.loads(l) for l in learner.read_text().splitlines() if l.strip()]
        if extra and not str(extra[0]["question"]).startswith("Replace this"):
            questions += extra
    print(format_eval_table(evaluate(questions, answer)))


def cmd_selftest() -> None:
    """Offline checks of the MP3 logic — no OpenAI, no Qdrant, no keys."""
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

    print("\n✓ selftest passed" if ok else "\n✗ selftest FAILED")
    sys.exit(0 if ok else 1)


def main() -> None:
    cmds = {"ingest": cmd_ingest, "ask": cmd_ask, "eval": cmd_eval, "selftest": cmd_selftest}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__); sys.exit(0 if len(sys.argv) < 2 else 1)
    cmds[sys.argv[1]]()


if __name__ == "__main__":
    main()
