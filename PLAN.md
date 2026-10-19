# MP3 Grounded Q&A: Build Plan

**Status (2026-10-19): phases 0–6 are done.** Total API spend is about $0.002.
- Ingest: 72 chunks.
- Inspect: picked a threshold of 0.50.
- Sweep: written to `mp3_sweep.txt`.
- Eval: 1.00 / 1.00 / 1.00 / 0.00, written to `mp3_eval.txt`.
- Docs: `mp3_reflection.md` and `mp3_grounded_qa_flow.md`.

Problems hit along the way:
- AVG HTTPS interception. Fixed with the Windows trust store in code, plus an AVG exception.
- The Vocareum budget was exhausted on every key. `.env` now uses the user's own OpenAI key; the Vocareum lines are commented out.

**Remaining (user):** a 2–3 min `ask` demo, and pushing to GitHub (`.env` is git-ignored).

Goal: carry the **MP2 Mini-RAG pipeline (v4, tuned)** forward into MP3 and add the
three MP3 capabilities: **citations, abstention, evaluation**. All new work goes in
`AI-RAG_MP3_GroundedQA/learner/`. MP2 stays read-only.

---

## 1. What I reviewed

| Source | Key takeaways |
|---|---|
| `AI-RAG_MP3_Brief.md` / `Week 12_Graded Mini Project.docx` | Same content. 4 TODO functions (`format_context`, `validate_citations`, `should_abstain`, `evaluate`), CLI `selftest / ingest / ask / eval`. Deliverables: `mp3_grounded_qa.py`, `mp3_eval.txt`, `data/learner_questions.jsonl`, `mp3_reflection.md`, demo. Optional stretch: threshold sweep. Constraints: OpenAI `gpt-4o-mini` + `text-embedding-3-small` + Qdrant only, no frameworks, **no new deps beyond MP2**. |
| `learner/mp3_grounded_qa_starter.py` | Working *generic* pipeline (dense-only, k=3, 500/80 chunking) plus the 4 TODOs. `selftest` checks the contracts offline. `format_eval_table` and `answer()` expect specific dict keys. |
| `MP2/learner/mp2_rag.py` (v4) | Tuned pipeline: `load_dotenv`, **Vocareum base-URL fix**, chunker 600/150 with sentence fallback + `chunk_index`, `query_points` (non-deprecated), **hybrid BM25 + dense weighted RRF with a same-story restriction**, TOP_K=7, anti-conflation prompt, `_facts_matched`. Got 5/5 in `mp2_validation.txt`. |
| `MP2_Reflection.docx` | Diagnosis-first tuning history (v1 to v4). Known limits: same-story heuristic, q2 "swamp adder" gap, cross-encoder rerank as the next step. |
| Data | Corpus is **identical** in MP2 and MP3. MP3 `predefined_questions.jsonl` is **new**: 4 answerable + 2 abstain. Note that it does **not** cover `03_blue_carbuncle`. MP3 `learner_questions.jsonl` is still a template. |
| Environment | MP3 `.env` = Vocareum proxy (`OPENAI_API_BASE=https://openai.vocareum.com`, about $1 budget), mixed `export`/plain lines. No venv yet. System Python 3.13 lacks `qdrant_client` and `rank_bm25`. |

## 2. Issues found in the MP3 starter (will be fixed in the build)

1. **Vocareum would be bypassed.** The starter's `OpenAI()` ignores `OPENAI_API_BASE` (the openai v1 SDK only reads `OPENAI_BASE_URL`), so it would call api.openai.com with a Vocareum key. MP2 already solved this, so I carry its fix forward.
2. **`.env` is never loaded.** The starter relies on `source .env`, which doesn't work on Windows/PowerShell. I carry MP2's `load_dotenv()` forward.
3. **Deprecated Qdrant calls.** `qdrant.search` and `recreate_collection` are deprecated. I swap in MP2's `query_points` and delete/create.
4. **Silent client failure.** The starter wraps client construction in `try/except Exception`, so a bad config shows up as a confusing "needs OPENAI_API_KEY" error. I'll keep the guard (selftest has to stay offline) but print the real cause.
5. **Windows console crash.** `✓ ✗ → —` raise `UnicodeEncodeError` on a cp1252 console (seen during this review). Fix: `sys.stdout.reconfigure(encoding="utf-8")`. Also, PowerShell 5.1 `>` writes UTF-16, so I add `eval --out mp3_eval.txt`, which writes UTF-8.
6. ~~"swamp adder" is not in the corpus~~ **CORRECTED:** it is in the corpus (`02_speckled_band.txt` line 33, chunks #11 and #12). My grep cut the line off before the phrase, and the second pattern needed 40 characters after "adder" when only about 30 follow. The real finding (see `mp3_reflection.md`): for q2's wording, those chunks rank 11th and 12th in dense retrieval and miss the top-7, so the answer leaves out the swamp adder and misreads chunk #13.

## 3. Architecture: how MP3 extends MP2

Single file `learner/mp3_grounded_qa.py` (graders run `selftest` on one file), organized as:

```
┌ Config        MP2 v4 values (CHUNK 600/150, TOP_K 7, DENSE/BM25 cands 12, RRF 60, weights 1.5/1.0)
│               + RETRIEVAL_THRESHOLD (env-tunable), ABSTAIN_MESSAGE
├ Clients       load_dotenv + Vocareum base_url (MP2), guarded so selftest needs no creds
├ Pipeline      ← CARRIED FROM MP2 v4 (attributed): load_corpus, chunk_document,
│                 embed_texts, setup_collection, ingest_chunks, retrieve (query_points),
│                 BM25 index, hybrid_retrieve
├ NEW 1 Citations   format_context, validate_citations
├ NEW 2 Abstention  should_abstain + answer()
├ NEW 3 Evaluation  evaluate, format_eval_table (+ extra diagnostics block)
├ Cost guard        local JSON cache + API-call/token counter
└ CLI           selftest | ingest | inspect | ask | eval | sweep
```

### Key design decision: the abstention gate vs. hybrid scores
Hybrid retrieval ranks by **RRF fused score**, which is rank-based and not comparable
to a cosine threshold. So:
- `answer()` runs the **dense retrieve once** (12 candidates). The **gate score = top
  dense cosine across all 12, before the same-story restriction.** That is the honest
  "is anything in the corpus relevant?" signal.
- `hybrid_retrieve` is refactored to **accept those precomputed dense hits**, which
  avoids a second embedding call per question.
- Every chunk passed to `format_context` keeps its dense cosine `score`. A BM25-only
  chunk (no cosine) shows as `bm25` in the sources list.
- `should_abstain(chunks, threshold)` keeps the starter's exact contract (top
  `score` < threshold, or no chunks), so `selftest` passes unchanged.

### Prompt
The starter's citation/abstain `SYSTEM_PROMPT`, plus MP2's anti-conflation paragraph
(it fixed a real John Horner / James Ryder mix-up in MP2). Excerpts are numbered `[1..7]`.

### Contracts kept exactly as the starter defines them
`format_context → (ctx, sources[n,source,title,section,score])`,
`validate_citations → {used, invalid, has_citation}` (ints, de-duplicated, in order of appearance),
`evaluate → {rows, summary{n_answerable, n_abstain, retrieval_hit_rate, grounded_rate, abstention_accuracy, hallucination_rate}}`.
Rates are rounded to 2 dp, and an empty denominator gives `None` / "n/a" instead of a ZeroDivisionError.

## 4. Extensions beyond the brief (small, no new deps, no extra API cost)

| Extension | Why |
|---|---|
| **`inspect` command**: retrieval only, prints the top dense cosine plus hybrid hits for every eval question. **Embeddings only, no chat calls.** | Lets me pick the threshold from data before spending any chat tokens. |
| **`sweep` command (Stretch)**: re-scores cached results across thresholds, e.g. 0.20 to 0.50 in steps of 0.05. **Zero extra API calls.** | Retrieval scores don't depend on the threshold, so every threshold can be simulated from one cached run. The table goes into the reflection. |
| **Diagnostics block under the main table**: per question, the top cosine, MP2-style facts matched (`n/m`), and the invalid-citation list. | Justifies the threshold choice and surfaces the "swamp adder" gap. The graded table format stays unchanged. |
| **Answer-side abstain detection** that also catches short "I don't know…" variants. | The starter's check needs the exact sentence. |

## 5. Cost guardrails (Vocareum about $1)

- **Local cache** at `learner/.cache/` (JSON, stdlib), keyed by sha256 of (model, input).
  Embeddings and temp-0 chat completions are deterministic enough that a re-run of
  `eval`/`sweep`/`inspect` costs **$0**. `--no-cache` forces fresh calls.
- **Call counter**: every command ends with `API calls: N embed / M chat, ~T tokens, ~$X`.
- **Retrieval-only paths** (`inspect`, `sweep`) never call chat.
- **Expected total spend** for the full build: ingest (1 batch call, ~8k tokens) + inspect
  (9 embeds) + eval (≤9 chat calls, ~2k tokens each) is **under $0.01**. Debugging re-runs
  are free via the cache.
- **No API call happens without your go-ahead at each checkpoint** (section 6).

## 6. Build phases (each ends at a checkpoint)

| # | Phase | API cost | Checkpoint |
|---|---|---|---|
| 0 | Create `learner/.venv`, `pip install -r requirements.txt` (add `rank-bm25`, which is an MP2 dep, so still compliant) | none | — |
| 1 | Write `mp3_grounded_qa.py`, `.gitignore` (**`.env`, `.venv`, `.cache`**, so the real keys never reach GitHub). Run `selftest` plus an offline parity check (the chunker gives 72 chunks like MP2, BM25 builds) | none | show selftest output |
| 2 | `ingest` into a new collection `mp3_sherlock` (keeps MP2's `sherlock_mini_rag` untouched) | 1 embed batch, about $0.0002 | **ask before running** |
| 3 | Author the 3 learner questions (check them against the corpus with grep), then `inspect` for all 9 and pick the threshold from the score gap | 9 embeds, about $0.00005 | show scores + proposed threshold |
| 4 | `eval --out mp3_eval.txt` | ≤9 chat calls, about $0.005 | **ask before running** |
| 5 | `sweep`, fully offline from cache | none | — |
| 6 | Docs: `mp3_grounded_qa_flow.md` (mirrors MP2's flow doc), README run notes, **draft** `mp3_reflection.md` using the real numbers | none | you review and edit the reflection |

## 7. Learner questions (proposal, final wording after a corpus check)
- **q5 (answerable, Blue Carbuncle, which predefined doesn't cover)**: who is wrongly arrested and what is his trade. Expected 03, facts `John Horner`, `plumber`.
- **q6 (answerable, multi-hop, Scandal in Bohemia)**: the two disguises and what each one reveals. Expected 05, facts `groom`, `clergyman`, `photograph`.
  *(Alternative if you'd rather the two stories differ from predefined: a Red-Headed League multi-hop on the encyclopaedia job plus the tunnel.)*
- **q7 (abstain, plausible near-miss)**: a Holmes character or event from a story *not* in the corpus, e.g. *"What was the name of the dog Holmes used to follow the creosote trail?"* (Toby, *The Sign of Four*). Chosen to sit close to the corpus vocabulary so it actually tests the gate.

## 8. Decisions (user, 2026-10-19)
1. **Retrieval: hybrid (MP2 v4).** The docx/brief doesn't require vector-only. Its only constraints are OpenAI + Qdrant, no frameworks, and no deps beyond MP2. `MP3_FAQ.md` says explicitly: *"If you did do MP2 and want to reuse your own pipeline, you're welcome to drop it in; just keep the four MP3 functions and the CLI."* `rank-bm25` was already a dependency of the user's MP2.
2. **Starter file:** leave `mp3_grounded_qa_starter.py` untouched. Work in a copy named `mp3_grounded_qa.py`.
3. **Reflection:** write `mp3_reflection.md` from the real run results.
4. **Learner questions:** section 7 proposals, checked against the corpus with grep (Horner/plumber and groom/clergyman are present, Toby/creosote is absent). Waiting for user confirmation.
