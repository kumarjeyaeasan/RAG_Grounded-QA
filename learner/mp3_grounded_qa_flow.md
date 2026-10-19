# mp3_grounded_qa.py: End-to-End Flow

This page shows how [mp3_grounded_qa.py](mp3_grounded_qa.py) runs, command by command. It mirrors `mp2_rag_flow.md`. Each part is tagged with where it comes from:
- **(MP2)** carried from MP2 v4
- **(MP3)** new for MP3
- **(ext)** an extension beyond the brief

---

## 0. Module load and client setup

| Line | What happens |
|---|---|
| [L74](mp3_grounded_qa.py#L74) | `RETRIEVAL_THRESHOLD = 0.50`. It can be overridden by the `RETRIEVAL_THRESHOLD` env var or `--threshold`. |
| [L116](mp3_grounded_qa.py#L116) | `_clients()` is **lazy**: the clients are built on first use, so `selftest` never touches the network. It does the following, in order: `load_dotenv()` **(MP2)**, `_use_windows_trust_store()` **(ext)**, `OpenAI(base_url=OPENAI_API_BASE)` **(MP2 Vocareum fix)**, `QdrantClient(...)`. |
| [L93](mp3_grounded_qa.py#L93) | `_use_windows_trust_store()` **(ext)**: on Windows it builds one bundle from certifi plus the Windows ROOT certificates and sets `SSL_CERT_FILE`. This is needed when antivirus HTTPS scanning (AVG) re-signs certificates. Verification stays **on**. |
| [L140-L190](mp3_grounded_qa.py#L140) | **(ext)** Cost guard. `_cache_get/_cache_put` store every embedding and every temperature-0 chat reply under `.cache/`, keyed by sha256(model + input). `usage_line()` prints the calls, tokens and estimated $ at the end of each live command. |

## `main` ([L817](mp3_grounded_qa.py#L817))

1. Forces UTF-8 stdout, because a Windows cp1252 console can't print `✓ ✗ →`.
2. Parses `command` plus the options: `--threshold`, `--dense-only`, `--no-cache`, `--out`, `--thresholds`.
3. Dispatches to the matching `cmd_*` function.

---

## Flow A: `ingest` **(MP2)**

```
cmd_ingest()
├── load_corpus()                    → [{source, title, text}] × 5
├── chunk_document(doc)              → 600/150 paragraph chunks + section labels → 72 chunks
├── setup_collection()               → delete if exists, create 'mp3_sherlock' (1536-d cosine)
└── ingest_chunks(chunks)
      ├── embed_texts(texts)         → cache-aware, 1 API call for 72 chunks
      └── qdrant.upsert(points)
```

## Flow B: `ask` → `answer(question)` ([L450](mp3_grounded_qa.py#L450))

```
answer(q, k=7, threshold=0.50)
├── dense = retrieve(q, k=12)                         (MP2)  1 embedding + Qdrant query_points
├── chunks = hybrid_retrieve(q, 7, dense_hits=dense)  (MP2)  BM25 (local) + weighted RRF,
│                                                            same-story restriction; reuses `dense`
│                                                            → no 2nd embedding
├── top_score = dense[0].score                        (MP3)  top cosine BEFORE story restriction
├── should_abstain(dense, threshold)?                 (MP3)  ── yes → ABSTAIN_MESSAGE, no chat call
├── format_context(chunks)                            (MP3)  "[n] (from "<title>" — <section>)\n<text>"
│                                                            + parallel sources[n, source, title,
│                                                              section, score, via]
├── chat(SYSTEM_PROMPT + excerpts + question)         (MP3 prompt + MP2 anti-conflation; cached)
├── _is_prose_abstention(text)?                       (MP3)  ── yes → abstain (reason "model")
├── validate_citations(text, n)                       (MP3)  [n] / [1, 3] → used / invalid
└── → {question, answer, citations, retrieved_sources, top_score,
       abstained, abstain_reason, citation_check, latency_ms}
```

**Why the gate uses the dense cosine:** RRF fused scores are rank-based (about 0.01–0.04, whatever the relevance), so they can't be thresholded. A chunk that only BM25 surfaced has `score=None`, and `should_abstain` ignores it.

## Flow C: `eval` → `evaluate(questions, answer_fn)` ([L502](mp3_grounded_qa.py#L502))

```
load_questions()          predefined (6) + learner (3), template rows skipped
for each q:
   r = answer_fn(q)
   should-abstain q  →  abst_ok += r.abstained ; hallucinated += not r.abstained
   answerable q      →  hit      = expected_source ∈ retrieved_sources
                        grounded = answered ∧ has_citation ∧ no invalid ∧ expected_source ∈ cited
→ {rows, summary{n_answerable, n_abstain, retrieval_hit_rate, grounded_rate,
                 abstention_accuracy, hallucination_rate, false_abstention_rate(ext)}}
format_eval_table()       graded table + diagnostics (top cos, MP2 facts n/m, cited/why abstained)
--out mp3_eval.txt        UTF-8 file (PowerShell's `>` would write UTF-16)
```

## Flow D: `inspect` **(ext)**, retrieval only

For each eval question it prints the top cosine, the top 3 dense hits, the hybrid top-k (`02#7(both)` etc.) and the gate decision. It uses **embeddings only, never chat**. This is how the threshold was chosen.

## Flow E: `sweep` **(ext, the brief's Stretch)**

```
ungated = answer(q, threshold=-inf) for all q     one chat call per question (cached)
for t in thresholds:
    evaluate(questions, λq: _apply_gate(ungated[q], t))     offline, $0
→ threshold × {hit, grounded, abst.acc, halluc., false-abst} table + top cosine per question
```

## Flow F: `selftest`, offline

- The starter's 13 checks, plus 8 more: grouped citations, BM25-only scores, prose abstention, an empty eval set, sweep gate simulation, **72-chunk parity with MP2**, the BM25 build, and "no API calls were made".
- No keys and no network.

---

## Summary

```
INGEST:  corpus → chunk (600/150) → embed → Qdrant 'mp3_sherlock'
ANSWER:  q → dense(12) ─┬─ top cosine < 0.50 ? → "I don't know based on the provided documents."
                        └─ hybrid RRF → top-7 → [1..7] context → gpt-4o-mini → validate [n] → cited answer
EVAL:    9 questions → 4 rates (+ false-abstention) → mp3_eval.txt
```
