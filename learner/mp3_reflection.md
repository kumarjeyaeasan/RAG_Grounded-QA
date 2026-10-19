# MP3 Reflection

*Grounded Q&A: a cited document assistant over the 5-story Sherlock Holmes corpus.
It extends my MP2 Mini-RAG (v4), not the generic starter pipeline. All numbers are from
the real run in `mp3_eval.txt` and `mp3_sweep.txt`.*

**What carried over from MP2:**
- Chunking: 600 characters with 150 overlap, paragraph-aware, with section-heading tracking. This gives 72 chunks, exactly as in MP2.
- Retrieval: hybrid BM25 + dense with weighted RRF (1.5 / 1.0), with both candidate pools restricted to the story of the top dense hit, and TOP_K = 7.
- The anti-conflation prompt paragraph.
- The Vocareum `base_url` fix.

**What MP3 adds:**
- The four functions: `format_context`, `validate_citations`, `should_abstain`, `evaluate`.
- An `inspect` command (retrieval only, no chat calls).
- A `sweep` command (the threshold Stretch, run offline from cached answers).
- A local response cache, so re-runs cost $0.

The whole live build cost about **$0.002** in API calls.

---

## Grounding & abstention

**Final threshold: 0.50 on the top dense cosine. The starter default was 0.30.**

The abstention gate compares the **top dense cosine across the whole 12-candidate dense pool,
taken before the same-story restriction.** It can't use the hybrid score: RRF fused scores are
rank-based (≈ 0.01–0.04 whatever the relevance), so a cosine threshold means nothing against
them. Chunks that only BM25 surfaced carry no cosine (`score=None`), and `should_abstain`
ignores them, because a keyword match can't vouch for relevance.

`inspect` (embedding calls only) gave this picture:

| Question | Should… | Top cosine |
|---|---|---|
| q1 Red-Headed assistant | answer | 0.689 |
| q3 Irene Adler | answer | 0.653 |
| q5 Carbuncle / Horner *(mine)* | answer | 0.602 |
| q4 Hydraulic press | answer | 0.572 |
| q2 Speckled band | answer | 0.547 |
| q6 Two disguises *(mine, multi-hop)* | answer | **0.540** (lowest answerable) |
| Moriarty | abstain | **0.468** (highest abstain) |
| Toby the dog *(mine)* | abstain | 0.449 |
| Boiling point of water | abstain | 0.157 |

At the default 0.30, the gate only stopped the boiling-water question. **Moriarty (0.468) and
my Toby question (0.449) went straight past it to the LLM.** Both are "Holmes-shaped": they
share the corpus's vocabulary (Holmes, deduction, the Victorian setting), so they score
nearly as high as real questions. The LLM still refused both, so the run at 0.30 was correct,
but only because the backup layer (the prompt's "I don't know" instruction) was doing the
primary layer's job.

I raised the gate to **0.50**. That's the middle of the 0.468 → 0.540 gap. Now all three
should-abstain questions are refused by the gate itself, with no chat call at all, and the
prompt stays as a second layer behind it. Nothing answerable was refused (false-abstention
rate 0.00).

**Caveat I'd want a reviewer to know:** the margins are thin (+0.03 above Moriarty, −0.04 below
q6), and I picked the threshold on the same 9 questions I then evaluated on. A differently
phrased real question could easily score 0.49 and be wrongly refused. With a bigger corpus or
question set, I'd tune on a held-out set and track the false-abstention rate.

## Citations

**`validate_citations` caught nothing in the wild.** gpt-4o-mini never emitted an out-of-range
marker: 0 fabricated `[n]` across 6 answered questions. Every marker mapped to a real excerpt.
I made the validator robust anyway: it also parses grouped markers like `[1, 3]` and flags `[0]`.
The selftest covers both cases.

**The real citation problem was one the validator can't see.** On q2 ("What killed Julia
Stoner… and how was the method delivered?") the answer said the snake came through the
ventilator, which is correct, and then added *"Holmes's blow on the rope had turned the snake
back through the ventilator, indicating that it had **attacked Julia** in her room [2]."* That
is wrong: in the corpus the snake attacked **Doctor Roylott**. The `[2]` is a valid marker
pointing at the right story, so q2 counts as **grounded ✓**. But the cited chunk doesn't
support the claim; the model misread it.

Digging in with `retrieve()` showed why:
- The two chunks that actually explain the method (#11 "The Vigil", which names the **swamp
  adder**, and #12 "The Explanation", which has *"trained the snake to climb the bell-rope
  through the ventilator"*) ranked **11th and 12th of 12** in the dense pool. Neither made the
  top-7 context.
- The model only saw chunk #13, whose 150-character overlap carries the *tail* of the
  explanation ("…it had attacked the only living thing in the room beyond") without the
  sentence saying who that was.

This is the same "rare-but-correct sentence loses to repeated names" failure I diagnosed in
MP2, coming back because MP3 words q2 differently ("delivered" vs. MP2's "discovered").
It also means the q2 fact check (1/2, no "swamp adder") is a **retrieval miss, not a
phrasing quibble.**

One smaller wording gap: on q1 the model said Spaulding was "a character named **Clay**"
rather than "John Clay". Grounded and correct, but the literal fact check scores it 1/2.

## Reading your eval table

```
Retrieval hit rate    : 1.00   (6 answerable)
Grounded & cited rate : 1.00   (KPI: grounded response rate)
Abstention accuracy   : 1.00   (3 should-abstain)
Hallucination rate    : 0.00   (KPI: hallucination rate)
False-abstention rate : 0.00   (my extra diagnostic)
```

On paper this is perfect: the right story was always retrieved, every answer cited it with
valid markers, and all three unanswerable questions were refused. Two reasons it's less
impressive than it looks:

1. **The metrics check *where* a citation points, not *whether it supports the claim*.**
   q2 shows a 1.00 grounded rate can contain a factually wrong sentence. The weakest
   number isn't any of the four; it's the one we don't measure: **citation faithfulness.**
   The next step I'd add is a claim-level check: split the answer into sentences and verify
   each against its cited chunk, via an NLI model or an LLM judge, or with RAGAS faithfulness
   as in the capstone.
2. **Hit rate is too coarse.** It's file-level ("is *any* chunk from 02 in the top-7?"),
   so it scored q2 as a hit even though the two chunks with the answer were missing. A
   chunk-level hit rate (did the *answer-bearing* chunk make the context?) would have caught
   it. For q2 specifically, I'd try a cross-encoder reranker over the 12 candidates. That's
   the step I already named as "what I'd change" in MP2.

## One surprise

How *high* the unanswerable questions scored. I expected "Who is Professor Moriarty?" to
look unrelated to a corpus that never mentions him. Instead it scored 0.468, closer to the
real questions (≥ 0.540) than to the truly off-topic one (0.157). Embedding similarity
measures **"does this sound like these documents?"**, not **"is the answer in these
documents?"** A Holmes question about the wrong Holmes story sounds exactly right. That's why
a 0.30 gate that looked sensible in the brief barely did anything here. The thing that actually
saved the run was the plain-English instruction in the prompt.

Second, from the build itself: getting the first API call through took longer than writing the
four functions. AVG's HTTPS scanning re-signs every certificate with its own root, which Python's
certifi bundle doesn't trust. It then blocked `openai.vocareum.com` as "phishing", and finally
every Vocareum key returned "Budget exceeded". The fixes:
- The code now trusts the Windows certificate store, so verification stays on.
- An AVG exception for the API host.
- My own OpenAI key.

The response cache I added "to save money" turned out to matter most for keeping the sweep and
re-runs free once a working key was in place.

---

## Stretch: threshold sweep

`python mp3_grounded_qa.py sweep` answers every question **once** with the gate off
(9 chat calls, $0.0019) and then re-applies each threshold offline. Full output is in `mp3_sweep.txt`.

| Threshold | Hit rate | Grounded | Abst. acc. | Halluc. | False-abst. |
|---|---|---|---|---|---|
| 0.20 – 0.52 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| 0.55 | 1.00 | 0.67 | 1.00 | 0.00 | 0.33 |
| 0.60 | 1.00 | 0.50 | 1.00 | 0.00 | 0.50 |

- **Below about 0.47 the table looks perfect, but only because the LLM refuses on its own.** The
  sweep shows the model declined all 3 should-abstain questions even with no gate at all.
  Hallucination rate never moves from 0.00, so on this set the prompt layer alone was enough.
- **0.47 → 0.52 is the sweet spot:** the gate itself stops Moriarty and Toby (no chat call, no
  reliance on the model behaving), and nothing real is refused.
- **0.55+ costs real answers:** q6 (0.540) and then q2 (0.547) are refused, so grounded rate
  falls to 0.67 and then 0.50. Raising the threshold trades false abstentions for safety you
  already had.
- **Hit rate is flat at 1.00** across the sweep. It's measured on retrieval, which the
  threshold doesn't change. The cost of a high threshold shows up entirely in the grounded
  and false-abstention columns.
