# MP3 · Grounded Q&A
## A Cited Document Assistant

> *Mini Project 3 · Week 12 deliverable · ~6–8 hours over 2–3 sittings.
> A Phase-2 showcase: take a working RAG pipeline and make it **trustworthy** —
> it cites its sources, admits when it doesn't know, and you measure how well it
> does. Demonstrates two of the eight production KPIs: grounded response rate and
> hallucination rate.*

---

## What you're building

You're extending **Mini-RAG (MP2)** into a *grounded, cited* assistant over the
same Sherlock Holmes corpus. The pipeline (load → chunk → embed → store →
retrieve → answer) is **provided and working**. You add the three things that
separate a demo from something you'd trust:

1. **Citations.** Every claim in the answer is tagged inline with `[1]`, `[2]`,
   and each marker maps to an actual retrieved chunk. You'll also *validate* the
   citations — catch the case where the model invents a `[7]` that doesn't exist.

2. **Abstention.** When retrieval finds nothing relevant, the assistant replies
   **"I don't know based on the provided documents"** instead of making something
   up. This is the single most important trust behaviour in RAG.

3. **A retrieval-quality check.** A small evaluation harness runs a question set
   and prints a **table**: retrieval hit rate, grounded response rate, abstention
   accuracy, and hallucination rate. This table is your main graded artifact.

You implement **four functions**; everything else is done for you.

---

## Why this exists

By Week 12 your capstone is a big, measured, defensible system (that's the M2
milestone). MP3 is the *small, portfolio-sized* version of the same idea, on a
fun corpus: prove you can make a RAG assistant that **cites, abstains, and is
measured**. When someone asks *"how do you stop a RAG system from hallucinating,
and how do you know it's working?"* — this project is your answer.

It also cements the two Phase-2 ideas that matter most in production: **grounding**
(don't answer beyond your sources) and **evaluation** (measure it, don't guess).

---

## The corpus

The same 5 public-domain Sherlock Holmes story summaries from MP2
(`corpus/*.txt`). Reusing them lets you see the *upgrade* on a familiar corpus.
As with MP2, you may **swap in your own** `.txt` corpus — just rewrite the
questions in `data/` to match, and say so in your reflection. That makes MP3 a
true portfolio piece.

---

## What you need to do

### Step 1 — Environment (15 min)

```bash
cd learner/
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         # then edit with your OPENAI_API_KEY + QDRANT_URL
source .env
```

Rename `mp3_grounded_qa_starter.py` → `mp3_grounded_qa.py`.

### Step 2 — Implement the four functions (2.5–3.5 hrs)

Open `mp3_grounded_qa.py` and fill in the four `TODO` functions. They're small
and focused — the pipeline around them already works.

| Function | Capability | What it does |
|---|---|---|
| `format_context` | Citations | Number the retrieved chunks `[1..n]` for the prompt; return a parallel `sources` list |
| `validate_citations` | Citations | Parse the `[n]` markers from the answer; flag any that don't map to a real source |
| `should_abstain` | Abstention | Return `True` when there's no sufficiently-relevant chunk (no chunks, or top score below threshold) |
| `evaluate` | Evaluation | Score the assistant over a question set → the four rates |

**Check your work offline as you go — no API key needed:**

```bash
python mp3_grounded_qa.py selftest
```

`selftest` exercises all four functions with synthetic data. When it prints
**`✓ selftest passed`**, your logic is correct. (It stops at the first
unimplemented function and names it.)

### Step 3 — Run it for real (30 min)

```bash
python mp3_grounded_qa.py ingest      # build the collection (once)
python mp3_grounded_qa.py ask         # ask a few questions — see the [n] citations
```

Try an **answerable** question ("What killed Julia Stoner?") and an
**unanswerable** one ("Who is Professor Moriarty?"). The first should answer with
citations; the second should say *"I don't know based on the provided
documents."* If it doesn't abstain, check your `should_abstain` threshold.

### Step 4 — The evaluation table (30 min)

```bash
python mp3_grounded_qa.py eval
```

This runs `data/predefined_questions.jsonl` (4 answerable + 2 that should abstain)
and prints your table. **Save this output** — it's the graded artifact. Read it:
is your hit rate sensible? Did both abstain-questions abstain (abstention accuracy
1.0, hallucination rate 0.0)?

### Step 5 — Author your questions (45 min)

Open `data/learner_questions.jsonl` and replace the three templates with your own:
**two answerable** (spanning different stories; make one multi-hop) and **one that
must abstain** (something plausible the corpus doesn't contain). Re-run
`python mp3_grounded_qa.py eval` — your questions are appended to the table.

### Step 6 — Reflection (45 min)

Create `mp3_reflection.md`:

```markdown
# MP3 Reflection

## Grounding & abstention
Where did your assistant abstain correctly? Did it ever answer something it
shouldn't have (or refuse something it could have answered)? What threshold did
you settle on, and why?

## Citations
Did the model ever cite a chunk that didn't support the claim, or invent a marker?
What did `validate_citations` catch?

## Reading your eval table
What do your four numbers say? Which is weakest, and what would you change?

## One surprise
Something from the build that genuinely surprised you.
```

Be specific: "at threshold 0.25 the Moriarty question still got answered from the
Roylott chunk (score 0.28), so I raised it to 0.35" beats "abstention was tricky."

### Step 7 (Optional STRETCH, 45 min) — Tune the threshold

Sweep the `RETRIEVAL_THRESHOLD` across a few values and re-run `eval`. Plot or
tabulate how abstention accuracy and hit rate trade off. Add a short note: where's
the sweet spot, and what does moving it cost you?

---

## What you submit

| File | What |
|---|---|
| `mp3_grounded_qa.py` | Your assistant (four functions implemented) |
| `mp3_eval.txt` | The output of `python mp3_grounded_qa.py eval` |
| `data/learner_questions.jsonl` | Your 3 questions (2 answerable + 1 abstain) |
| `mp3_reflection.md` | Your reflection |
| A 2–3 min demo | Recorded or live: a couple of cited answers + one correct "I don't know" |

Push to a personal GitHub repo (separate from the capstone). Submit the repo URL.

---



## Constraints

- OpenAI (`gpt-4o-mini` + `text-embedding-3-small`) and Qdrant only — same as the
  capstone. No other providers.
- No frameworks (no LangChain / LlamaIndex). The `openai` + `qdrant-client`
  libraries directly. MP3 adds **no** new dependencies beyond MP2.
- Expected spend: **$0.05–$0.20** on OpenAI. Set a $5 cap if you're nervous.

---

## Help

- **`selftest` fails / raises `NotImplementedError`?** That function isn't
  implemented yet — it names which one. Fill it in and re-run.
- **It answers when it should abstain?** Your threshold is too low, or
  `should_abstain` isn't reading the top score. Print the retrieved scores.
- **It abstains on a question it should answer?** Threshold too high, or the
  chunk genuinely didn't retrieve — inspect with `ask`.
- **A `[n]` in the answer with no matching source?** That's the fabricated-citation
  case `validate_citations` exists to catch. Good — note it in your reflection.
- **Qdrant asleep?** Free-tier clusters take ~60s to wake. Try again.
- See `MP3_FAQ.md`.

---

*"It is a capital mistake to theorise before one has data."* — Sherlock Holmes.
Now go measure it. Good luck.
