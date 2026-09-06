# AI Admission Chatbot - St. Xavier's College, Ahmedabad

A retrieval-augmented (RAG) chatbot that answers admission questions **only** from
the college prospectus and the official website source, with an admin panel for
keeping that dataset up to date.

## Run it

```bash
venv\Scripts\activate
pip install -r requirements.txt                 # see the note at its top about torch's CUDA index

# Postgres must be running and the schema applied (see Database section below)
python -m backend.db.init_db

python -m backend.rag.ingest --reset            # build the search index
cd frontend && npm install && npm run build     # build the React UI into frontend/dist
cd .. && python -m uvicorn backend.api.main:app --reload
```

Open <http://127.0.0.1:8000> - chat is on the page, **Admin** is top-right.
Default login is `admin` / `admin123` (change it in `.env`, see `.env.example`) -
seeded automatically into Postgres on first boot.

### Frontend dev mode

For live-reloading UI work, run the Vite dev server alongside the backend instead
of rebuilding on every change:

```bash
# terminal 1
python -m uvicorn backend.api.main:app --reload
# terminal 2
cd frontend && npm run dev
```

Open <http://127.0.0.1:5173> - Vite proxies `/api/*` to the backend (see
`frontend/vite.config.js`). Run `npm run build` again before relying on the
FastAPI-served copy at :8000, since it serves the static `frontend/dist` output,
not the dev server.

## How it works

```
question -> [PDF pool: vector + BM25]     -\
         -> [website pool: vector + BM25] -+-> merge -> cross-encoder rerank
         -> relative-confidence cutoff -> top 5 -> Qwen2.5-3B -> grounded answer + sources
```

| Layer | Location |
|---|---|
| PDF / HTML text extraction | [backend/preprocessing/](backend/preprocessing/) |
| Ligature repair | [backend/preprocessing/ligatures.py](backend/preprocessing/ligatures.py) |
| Chunking | [backend/preprocessing/chunker.py](backend/preprocessing/chunker.py) |
| Embeddings | [backend/embeddings/embedder.py](backend/embeddings/embedder.py) |
| Vector store (Chroma) | [backend/vectorstore/store.py](backend/vectorstore/store.py) |
| Retrieval + prompt | [backend/rag/](backend/rag/) |
| Local Qwen model | [backend/llm/qwen.py](backend/llm/qwen.py) |
| API + auth + admin | [backend/api/](backend/api/) |
| Website crawler (BFS/DFS/A*) | [backend/crawler/](backend/crawler/) |
| Database (Postgres) | [backend/db/](backend/db/) |
| UI (React + Vite) | [frontend/src/](frontend/src/) |

### Grounding

The model is instructed to answer only from retrieved context and to reply with a
fixed "I don't have that information..." line otherwise. Off-topic questions are
refused, and every answer ships with the page/section it came from so students can
verify it.

### Retrieval pipeline (`backend/rag/retriever.py`)

Two stages, not one pass of vector search:

1. **Candidate pools, one per source (20 each)** - the PDF chunks and the
   website chunks are searched as two *separate* pools (`doc_type=pdf` /
   `doc_type=website` via Chroma's `where` filter), each combining vector
   search (bge-small, cosine) with BM25 (`rank_bm25`, a source-scoped
   in-memory index, rebuilt when that source's chunk count changes). Both
   pools are then merged and de-duplicated.

   Two pools instead of one shared top-k: the PDF and website corpora are
   very different sizes (a handful of PDFs vs. ~100 crawled pages), so a
   single shared search can end up dominated by whichever source has more
   chunks, starving the other out before reranking ever sees it. Separate,
   fixed-size pools guarantee both sources get a fair shot - verified in
   testing: "What documents are required for admission?" correctly answers
   from the website (`admissions-accommodation.html`) in the same index that
   answers seat counts and course details from PDFs.
2. **Rerank** - a cross-encoder (`BAAI/bge-reranker-base`) scores the *full
   question against the full text* of every candidate from both pools - a
   fundamentally different, stronger signal than cosine distance between two
   fixed vectors, because it can reason about "does this text answer this
   question" rather than "are these two vectors close," and it makes the
   final call on relevance regardless of which source a candidate came from.

Only after reranking does a candidate get filtered - and not with a fixed
score cutoff:

```python
top_score = candidates[0]["rerank_score"]
if top_score < RERANK_FLOOR:        # 0.02 - even the best hit looks like noise
    return []
cutoff = max(RERANK_FLOOR, top_score * RERANK_RELATIVE_MARGIN)  # 0.5
return [c for c in candidates if c["rerank_score"] >= cutoff][:top_k]
```

A single absolute threshold doesn't work here: calibration found a genuinely
correct match for a terse query ("holiday list ?" -> the actual holiday dates
in that year's calendar PDF) scoring only **0.045**, while a well-phrased
question's correct match scored **0.999** - both real, wildly different
scale, because reranker scores reflect how *clearly* worded the match is, not
just whether it's right. A relative floor (clear the noise floor, then be
within half of the best candidate) keeps the terse case without opening the
door to fabricated questions, which calibrated to a clean ~0.0 top score.

This replaced an earlier pure-vector-search design with a fixed cosine
distance cutoff, which had two failure modes in practice: a bare proper noun
("Dr. Pravida Raja") doesn't embed distinctively, so nearest-neighbour search
could rank the *wrong* page above the right one with deceptive confidence;
and a fixed distance threshold rejected genuine matches phrased tersely
("holiday list ?" scored above any reasonable fixed cutoff on cosine distance
alone) while sometimes accepting fabricated questions that happened to land
under it. The reranker - actually reading question against passage - resolves
both, and BM25 recovers exact lexical matches (a name, a course code) that a
semantic embedding alone can miss.

### Performance: the reranker, not the LLM, was the bottleneck

Profiling a real cold query (not estimated) found the cost was nowhere near
where the usual RAG assumption puts it:

| Stage | Time |
|---|---|
| Query embedding | ~14 ms |
| Vector search (both sources) | ~22 ms |
| BM25 search (both sources) | ~20 ms |
| **Reranker (40 real candidates)** | **~6,300-6,400 ms** |
| LLM generation | ~2,000 ms |

The reranker was ~75% of total latency - the opposite of the usual "LLM is
the slow part" assumption. Why: it was calibrated on short hand-written test
snippets (~100-200 chars); real chunks are 1600 chars each, and cross-encoder
cost doesn't scale linearly with input length.

Two tempting fixes were tested and **rejected** because they broke a real,
previously-fixed case:

* **Truncating chunk text before reranking** (e.g. to 400-600 chars) - the
  actual fact for "holiday list ?" (`Academic-Calendar-24-25.pdf`) sits at
  character ~650, well past any such cutoff. Measured: reranker score for
  that chunk goes from 0.0785 (full text) to **0.0000** at 400 chars - the
  fact is simply invisible to the reranker at that point, and the whole
  question fails again. Calendar/tabular PDF content doesn't front-load its
  facts the way prose does, so "first N characters" isn't a safe proxy for
  "the relevant part."
* **Cutting `PER_SOURCE_CANDIDATES`** (e.g. 20 -> 10 per source) - the same
  chunk ranks **13th** in its 20-item PDF pool on raw vector+BM25 score (a
  vague query like "holiday list ?" doesn't score calendar content highly on
  either signal). Cutting to 10 excludes it before the reranker - which is
  good at judging relevance - ever sees it.

What actually worked, verified end to end with no regressions: **moving the
reranker to GPU** (`RERANK_DEVICE`, auto-detected) **with a small batch size**
(`RERANK_BATCH_SIZE=8`). VRAM is tight with Qwen also loaded (~1.3GB free on
this 8GB card); the default batch size of 32 peaked at ~1.7GB (over budget),
but batch_size=8 peaked at ~1.29GB - under budget, and *faster* than 32, not
slower (smaller batches meant less allocator overhead here). No chunk content
or candidate count changed, so none of the retrieval-quality risk of the two
rejected approaches. `retriever.py` still catches a genuine
`torch.cuda.OutOfMemoryError` and retries that one request on CPU rather than
failing it - slow and correct beats fast and broken, especially for a
prospective student's admission question.

Real-world result across 5 diverse questions after the change: steady-state
latency dropped from ~10-27s (CPU reranking) to ~6.7-8.5s (GPU reranking),
with the "holiday list ?" case still answering correctly.

### 4-bit NF4 quantization for serving - tested, rejected

Qwen2.5-3B-Instruct is served in fp16. QLoRA fine-tuning already trains a
4-bit base (`backend/finetune/train.py`), so serving the same 4-bit weights
looked like a free win: less VRAM, more headroom for the reranker. Benchmarked
both, standalone (server stopped, clean GPU), same 5 real questions through
the actual retriever + prompt, greedy decoding both ways.

The first pass at this benchmark had a bug worth admitting rather than
burying: the timer around `model.generate()` also included `retrieve()`
(embedding + vector search + BM25 + reranking, ~6-8s on its own per the
profiling table above), so the first "tokens/sec" numbers reported here
described blended pipeline throughput, not LLM decode speed, and made 4-bit
look ~2.5x faster than it actually is. Re-run with the two timed separately:

| Metric | fp16 | 4-bit NF4 |
|---|---|---|
| Load peak VRAM | 5,886 MB | 3,102 MB |
| Generation peak VRAM | 8,021 MB | 4,088 MB |
| Pure LLM decode, 5 questions | 53.9 s | 57.2 s |
| Tokens/sec (LLM only) | 3.2 | 3.4 |

Corrected picture: the VRAM numbers were never affected by the bug and hold
- 4-bit is genuinely ~49% less peak VRAM, which matters on an 8GB card where
fp16 generation peaks at 8,021 MB, leaving almost nothing for the reranker.
But the speed win was mostly a measurement artifact - pure decode speed is
close to identical (3.2 vs 3.4 tok/s), not the 2.5x this section originally
claimed. So the actual tradeoff is "less VRAM, same speed," not "less VRAM
and faster," and either way comparing the actual answers disqualified it on
quality before speed was even the deciding factor:

* **Fabricated a faculty email address.** Asked "Who is Dr.Pravida Raja?",
  fp16 answered correctly from the retrieved passage (department + committee
  role, both present in the source). 4-bit added
  `pravida.raja@sxca.edu.in` - grepped the entire `data/` corpus for that
  string and it appears **nowhere**. Not a paraphrase of something in
  context; invented outright. This is precisely the failure mode the whole
  RAG design exists to prevent, and the pipeline's grounding check
  (`_is_refusal()` in `pipeline.py`) can't catch it - it only detects a full
  refusal, not a fabricated detail sitting inside an otherwise-correct
  answer.
* **Conflated non-holiday calendar entries into a "holiday list" answer.**
  4-bit's answer included "BoS meetings" and "Academic Council Meeting" as if
  they were holidays. Milder than the email case - both phrases are real text
  in the same retrieved passage (`Academic-Calendar-24-25.pdf`), so this is a
  relevance/categorization slip, not invention - but still a correctness
  regression fp16 didn't have on the same question.

Kept fp16 for serving. Both the fabricated email and the holiday/meeting
conflation reproduced identically on the corrected re-run, so this isn't a
one-off sampling fluke (decoding is greedy/deterministic on both configs
anyway). The VRAM saving is real and would matter if GPU headroom becomes the
binding constraint later, but for an admissions chatbot a hallucinated staff
email is a worse failure than a VRAM-only win with no speed benefit to offset
the risk. If revisited, this would need a stricter fact-level grounding check
(not just "did it refuse"), not just a quantization flag flip.

### Answer cache (`backend/rag/cache.py`)

Retrieval (two BM25 indexes + vector search) plus reranking plus LLM
generation is several seconds to tens of seconds - and for a college FAQ
chatbot, many students ask close to the literal same question. A repeated
(normalized) first-turn question skips the entire pipeline and returns the
stored answer directly: **verified 30s cold -> 0.14s on a cache hit**, and a
repeat with different capitalization/spacing ("HOW many SEATS...") hits the
same cache entry.

Two things keep this safe rather than just fast:

* **Scoped to fresh questions only** - anything with conversation history
  skips the cache, since the same text asked as a follow-up ("and the
  fees?") can mean something different depending on what came before it.
* **Invalidated on any index change** - the cache clears itself the moment
  the total indexed chunk count changes (document upload, crawl, delete),
  reusing the same staleness signal the per-source BM25 indexes already
  check. Verified: uploading a test document changed the chunk count by one
  and the very next identical question correctly missed the cache and
  re-ran the full pipeline, rather than serving a stale answer.

### Conversation memory + query rewriting (`backend/rag/query_rewrite.py`)

A follow-up like "what about MSc AI?" then "what is the fee?" only works if
the *second* question's retrieval knows it means MSc AI's fee - `history` in
the prompt tells the LLM what was said, but `retrieve()` only ever saw the
literal question text, so a bare follow-up searched on its own few words
regardless of how good the LLM was at reading the conversation. This was a
confirmed, reproducible gap before this change (retrieval for "what is the
fee?" alone finds nothing course-specific).

Now: `conversation_id` (client-generated per page load, see `api.js`) is
threaded through `/api/chat` and `/api/chat/stream` into
`pipeline.answer()`/`answer_stream()`. Postgres becomes the source of truth
for a conversation's turns (`chat_history.conversation_id`, reconstructed via
`repo.get_recent_turns()`) rather than trusting whatever the client happens
to send - the client's `history` is only a fallback for callers with no
`conversation_id`. Before retrieval, `query_rewrite.rewrite()` asks Qwen to
turn the latest question into a standalone **search-style** query and to
report any newly-mentioned course/qualification, which is persisted to
`conversation_memory` so later turns already know it. The original question
(not the rewritten one) is still what's shown to the LLM and logged - only
the *retrieval* query changes.

Two real failures surfaced while testing this against the literal
"courses -> MSc AI -> fee" scenario, both fixed with evidence, not guesses:

* **The rewriter initially failed the actual task.** First version echoed
  "What is the fee?" back unchanged with `course: null`, despite MSc AI
  being right there in the conversation - valid JSON, wrong content. Fixed
  the same way `prompt.py`'s rule 3 was fixed earlier in this project: added
  a worked example showing the correct vs. wrong output for this exact
  scenario. Confirmed fix: re-ran the same input, got
  `{"query": "MSc Artificial Intelligence fee", "course": "MSc Artificial Intelligence", ...}`.
* **Full-question phrasing scores far lower on the reranker than a compact
  query.** Measured directly: "What is the fee for MSc Artificial
  Intelligence?" scored 0.053 against the correct fee-table chunk;
  "MSc Artificial Intelligence fee" scored **0.447** against the same
  chunk - an order of magnitude apart, and enough to flip the answer from a
  refusal to a grounded one. This is a pre-existing reranker characteristic,
  not something query rewriting introduced (confirmed: a fresh, single-turn
  "What is the fee for MSc AI?" with no conversation history at all refuses
  too). Since the rewriter already controls the retrieval query text
  independently of what's shown to the LLM, its prompt now asks for a
  compact, keyword-style query ("subject + fee", not "What is the fee
  for...") rather than a full grammatical question.
* **A refusal-detection gap let a wrong deflection through as "grounded."**
  One test run had the model reply "MSc AI fee: Please contact the
  admission office for confirmation." attached to completely unrelated
  sources (a seminar page) - it didn't match the single hardcoded refusal
  string, so `pipeline._is_refusal()` missed it and the UI would have shown
  irrelevant sources next to a non-answer. Broadened the check to a tuple of
  markers. This is a whack-a-mole fix, not a structural one - the model can
  still phrase a deflection a third way that neither marker catches. A
  proper fix (checking the answer actually restates something from the
  retrieved passage, not just string-matching known refusal phrasings) is
  future work, not done here.

**Known limitation, found but not fixed in this pass:** the admission
brochure's fee table spans multiple programs (UG grant-in-aid, UG
self-financed, PG) across chunk boundaries that don't cleanly separate them.
On one test run, the top-scoring chunk for "MSc Artificial Intelligence fee"
was a *different* program's fee row (UG chemistry/electronics/maths) even
though the correct PG row scored high enough to pass the cutoff in isolated
testing - the two candidates didn't both survive into the same request's
pool. The model correctly declined rather than fabricating a number it
couldn't see (the safe failure mode), but the answer was incomplete when a
complete one exists in the corpus. Root cause looks like chunk-boundary
placement inside a wide, multi-program table rather than anything
conversation-memory-specific - worth a dedicated look at table-aware
chunking, not a quick patch.

### Source priority (`backend/rag/source_priority.py`)

The requirement: the official website should outrank an official PDF when
both are genuinely relevant, since the live site is more likely to reflect
the current admission cycle than a point-in-time PDF snapshot - but a
barely-relevant website page still shouldn't outrank a clearly-relevant PDF
section just because of where it came from. Implemented as exactly that: a
capped 8% multiplicative boost applied to `rerank_score` before the sort and
confidence cutoff in `retrieve()`, not a hard override. Verified the
boundary explicitly, not just by inspection - a ~4% score gap flips to the
website, a 40% gap stays with the PDF:

```text
close tie (website 0.50, pdf 0.52): boosted 0.540 vs 0.520 -> website wins
clear PDF win (website 0.50, pdf 0.70): boosted 0.540 vs 0.700 -> pdf wins
```

The raw `rerank_score` is still what's shown in the UI's source list (via
`as_sources()`) so "why did this rank first" stays inspectable even though
ranking itself runs on the boosted score.

**A second half of this feature - boosting a more recently updated PDF over
an older one covering the same topic - was built, tested, and deliberately
not shipped.** The obvious signal, `documents.updated_at`, turned out to
mean "last re-ingested," not "actual document vintage": a full
`ingest_all()` sweep touches every file in the same run, so relative
`updated_at` order among bulk-ingested files is closer to filesystem
traversal order than publication date. Measured directly on this corpus:
`SXC-Handbook-2025-26.pdf` (the **older** edition) came out with a *higher*
freshness boost than `SXCA-Handbook-2026-27.pdf` (the newer one) - backwards
from what the feature was for. Shipping it would have actively made the
stale-document-version problem worse, not better, so it was left out rather
than patched around. A real fix needs an actual vintage signal (parsing a
year out of the filename/title, or an admin-set "supersedes" relationship)
- not attempted here, since a wrong freshness signal is worse than none.

**"Wrong college/location" filtering turned out to already be solved**,
not new work - `documents`/chunk metadata already carry `college_id`
(scoped via `default_college_id()`, see the Multi-college schema section
below), the crawler is hard-restricted to `sxca.edu.in` at the API level
(`admin.py`'s `crawl_website` rejects any other domain), and no ingest path
accepts a college override. Confirmed empirically rather than just
by reading the code: scanned every indexed chunk for "St. Xavier's" mentions
alongside another campus city (Mumbai, Kolkata, Delhi, etc.) and found zero.
A same-named college in another city was never retrievable in the first
place, so no new filter was needed for it.

### Structured eligibility (`course_eligibility`, `backend/rag/eligibility.py`)

Some questions - "I completed 12th Science, which courses can I apply for?",
"can a BCA graduate apply for MSc AI?" - aren't a single-passage lookup, they
need matching against a table of courses. Letting the LLM reconstruct that
table from whatever chunks retrieval happens to find is exactly what
produced the fee-table chunk-boundary failure documented above. Phase 3.1
addresses the data side of that: a curated `course_eligibility` table
(schema in `backend/db/schema.sql`) and a lookup module
(`backend/rag/eligibility.py`), populated from one verbatim, hand-verified
transcription of the admission brochure's fee/eligibility table
(`backend/db/seed_eligibility.py`) rather than an automated extraction pass.

Schema-and-lookup was Phase 3.1, standalone-tested but not yet wired into
the chat flow. Phase 3.4 (below) connects it to `pipeline.answer()`.

One schema detail worth calling out: rows are keyed on
`(course_name, category, academic_year)`, not just `course_name`, because
the source data itself has programmes offered under two different
categories with very different fees - verified in the seeded data, not
assumed: BA-Economics, BA-English and BA-Psychology each appear as both
"Government Aided" (~₹6,655/sem) and "Self-Financed" (₹32,000-35,750/sem)
rows. Collapsing those into one row per course would either lose a real fee
or force picking one arbitrarily - same principle as `prompt.py`'s rule 3
for RAG answers ("don't merge different categories' numbers").

### Wiring structured eligibility into the chat flow (Phase 3.4)

`query_rewrite.py`'s single LLM call was extended from pure follow-up
rewriting into full "understanding": alongside the search query and
course/qualification, it now classifies `needs_structured_lookup` - whether
the question is the kind a college eligibility+fee table would answer, in
any phrasing or language. It also runs on *every* question now, not just
follow-ups, since "Can a BCA graduate apply for MSc AI?" needs this even as
someone's first message - a question this phase specifically had to handle,
per the project's own natural-language-variation test.

Tested against 5 phrasings of the same question - "Can a BCA graduate apply
for MSc AI?", "I completed BCA, am I eligible for AI master's?", "After BCA
can I do MSc Artificial Intelligence?", "Is MSc AI possible after BCA?",
"BCA ke baad MSc AI kar sakte hai?" (Hindi) - which should all resolve to
the same lookup. Two real, reproducible bugs turned up before all 5 passed,
neither a hypothetical worry:

* **A course-specific structured match still got a wrong final answer, even
  though the correct data was in context.** Asked "Can a BCA graduate apply
  for MSc AI?" with the right MSc-AI row present, the model answered
  instead with BCA's *own* eligibility text - text that didn't appear in
  any of the 5 retrieved RAG passages either, so it wasn't grounded in them.
  The highest-scoring passage (0.92, the top of all 5) was an unrelated UG
  chemistry/electronics fee-table fragment; the model appears to have
  reached for its own general knowledge of typical BCA admission criteria
  when the given passages were noisy, despite correct structured data being
  right there. Fixed two ways: (1) when a question names a specific course
  and structured data has an exact match (`find_by_course`), RAG passages
  are skipped entirely for that turn - structured data is complete and
  authoritative on its own for that case, so there's nothing for a noisy
  passage to add except a chance to distract the model - and (2) a second
  worked example in `prompt.py`'s rule 7 targeting the specific confusion:
  the model was attributing the eligibility list to "BCA" (the student's
  stated qualification) instead of "MSc Artificial Intelligence" (the
  course actually being asked about), apparently pattern-matching on the
  word "BCA" appearing near the end of MSc AI's own eligibility string
  (`"...BE & BCA"`). A stream-based recommendation match (`find_by_stream`,
  many courses, no single exact course named) keeps RAG passages alongside
  it - that's a broader case without this demonstrated failure.
* **The extraction step didn't map casual phrasing or bare subject names to
  what the structured table actually indexes on.** "AI master's" was
  captured as `course: "AI master's"` verbatim instead of "MSc Artificial
  Intelligence" - a phrase that matches nothing. Separately, "I have Maths
  but not Biology, which courses can I apply for?" got
  `needs_structured_lookup: false` and `qualification: "Maths"`, missing
  the lookup and the stream entirely (the eligibility table only tracks the
  Science/Commerce/Arts stream, not individual subjects, and "which courses
  can I apply for" is exactly the case this feature exists for). Fixed with
  two more worked examples: one mapping a casual program reference to its
  full canonical name, one mapping named subjects to the stream they belong
  to (Maths/Biology/Physics/Chemistry → Science, etc.) - re-tested directly
  against `understand()` before re-testing the full pipeline, to confirm
  the extraction itself was fixed and not just the one case that happened
  to work.

**A known remaining gap, not fully closed:** rule 7's "don't turn unspecified
into a rejection" example was written for a single-course yes/no question.
For a multi-course recommendation list ("I have Maths but not Biology, which
courses can I apply for?"), the answer correctly lists only genuinely
Science-eligible courses and never states a false rejection - the core
safety property holds - but it doesn't always proactively add the caveat
that subject-level detail isn't tracked. Left as a known limitation rather
than chased further with more prompt examples, which was reaching
diminishing returns; this is exactly the class of problem an Answer
Validator (a later phase) is the more structural fix for, not another
prompt tweak.

End-to-end regression after all fixes: the original motivating scenario -
"What courses are available?" → "What about MSc AI?" → "What is the fee?" -
now resolves cleanly through conversation memory, understanding, and
structured lookup to "₹50000 per semester," with no chunk-boundary
ambiguity and no wrong-table risk. Phases 1-2's existing test cases
(holiday list, off-topic refusal, proper-noun lookup) were re-verified
unaffected.

### Ligature repair (important)

The 2026-27 prospectus font exports its `ti`/`tt`/`ft`/`tti` ligatures as `U+0000`,
so raw extraction produces `cri\0cal`, `so\0ware`, `a\0en\0on` - about 1000 broken
words that wrecked both search and answers. `ligatures.py` tries each expansion and
keeps the one the Qwen tokenizer scores as a real word (a real word costs fewer BPE
tokens than a misspelling), which needs no wordlist and keeps working on future
uploads. Verified 26/28 on a hand-checked sample; stubborn proper nouns are pinned
in `OVERRIDES`.

## Website crawler

`backend/crawler/crawler.py` walks the college's own domain (and subdomains)
starting from a URL, saves each fetched page under
`data/raw/websie/crawled/`, and indexes it immediately through the same
pipeline as an admin upload. One crawl loop, three interchangeable frontier
disciplines ([backend/crawler/frontier.py](backend/crawler/frontier.py)):

* **BFS** - FIFO queue, explores the site level by level.
* **DFS** - LIFO stack, follows one branch (e.g. Admissions -> UG -> ...) deep
  before backtracking to siblings.
* **A\*** - priority heap ordered by a relevance heuristic
  ([backend/crawler/heuristics.py](backend/crawler/heuristics.py): keyword
  matches on the URL/anchor text for admission, fees, eligibility,
  scholarship, hostel, courses, ...) minus a depth penalty, so the most
  admission-relevant pages are fetched first regardless of how deep they are
  linked from the homepage. This is the default strategy.

Run it from the CLI:

```bash
python -m backend.crawler.crawler --start-url https://sxca.edu.in/ --strategy astar --max-pages 30
```

or from **Admin -> Website crawler** in the UI, which also shows the history
of past runs (fetched/skipped/error counts per run, from the `crawl_runs`
table).

Etiquette and safety are not optional flags: it honours `robots.txt`, stays on
the college's own domain (the admin API rejects any other `start_url` to
prevent it being used as an open SSRF-style fetcher), sends one request at a
time with a delay between them, and both the CLI and the API enforce hard
caps on page count/depth (`CRAWL_HARD_MAX_PAGES` / `CRAWL_HARD_MAX_DEPTH` in
`backend/config.py`) so a typo can't turn it into a denial-of-service tool.

`python -m backend.rag.ingest` (used for `--reset` and the admin "Rebuild
index" button) treats the current files on disk as ground truth: it prunes
any registry/index entry whose file no longer exists, so deleting a crawled
page by hand doesn't leave stale chunks behind.

**On-site PDFs are followed too**, not just HTML pages - many pages here
(academic calendar, per-semester syllabi) are just buttons linking to a PDF
rather than showing the content inline. Discovered PDFs are downloaded into
`data/raw/pdfs/crawled/` and indexed through the same PDF pipeline as an
admin-uploaded prospectus (`backend/crawler/crawler.py`'s `_is_pdf_link`).

**Multiple years of the same document are a real risk, not a hypothetical
one.** A comprehensive crawl found four years of prospectuses and two of the
student handbook, all indexed as equals - and the chatbot answered "who is
the principal" from a stale 2023-24 prospectus instead of the current one,
because nothing told it which document was authoritative. There's no
automatic fix for this yet: after a broad crawl, check
**Admin -> Documents** for duplicate/dated versions of the same document
(prospectus, handbook, calendar) and delete the outdated ones - keeping only
current documents indexed is what keeps answers current.

### Live crawl on a retrieval miss

When neither the pre-built index nor a passage that merely *looked* relevant
actually answers a question, the chatbot doesn't just give up - it runs a
small crawl of the college site right now, prioritized by A* using **this
specific question's own words** (`backend/crawler/heuristics.py`'s
`build_query_heuristic`, not the fixed admission-keyword list used for the
general crawl), indexes whatever it finds, and retries once
(`backend/rag/pipeline.py`, `backend/crawler/live_search.py`). This is what
closed the academic-calendar gap above in practice: the general crawl's fixed
heuristic doesn't know "calendar" or "holiday" are worth prioritizing, but a
live crawl guided by "What are the college holidays in the academic
calendar?" heads straight for anything with "academic" or "calendar" in its
URL or link text - and reached the actual PDF the general crawl's budget
never got to.

This adds real latency (a live crawl round trip is ~20-50s) but only when
this fallback actually fires - already-answerable questions are unaffected.
Streaming shows "Checking the college website for more..." during the wait
(a `restart` SSE event discards whatever was streamed from the failed first
attempt, so a refusal never sits on screen next to its own correction). Two
safeguards keep a real content gap from turning into unbounded traffic
against the college's server: a 10-minute cooldown per distinct question, and
a global lock so only one live crawl runs at a time - a second question
missing at the same moment just falls through to "I don't have that
information" rather than starting its own concurrent crawl.

## Intent clustering

**Admin -> Intents** groups logged questions by *meaning* rather than exact
wording, so an admin sees "eligibility for BSc CS asked 3 different ways" as
one row instead of three ([backend/analytics/intents.py](backend/analytics/intents.py)).
It embeds every logged question (same bge-small model as retrieval) and runs
agglomerative clustering with cosine distance.

This is deliberately tuned to favour precision over recall. Short admission
questions are entity-heavy, and testing showed same-entity-different-intent
pairs can score a *higher* similarity than genuine paraphrases - "seats for BA
Economics" vs "duration of BA Economics" scored 0.81, above a real paraphrase
pair at 0.84. No single threshold perfectly separates those two cases with
this embedding model, so `INTENT_DISTANCE_THRESHOLD` (`backend/config.py`,
default 0.17) sits low enough to avoid merging different intents, at the cost
of occasionally leaving two genuine paraphrases as separate, similar-looking
clusters. A false split just shows up as two rows; a false merge would hide a
real distinct question behind one label, which is the worse failure mode for
an admin trying to see what's actually being asked.

## Admin panel

* **Documents** - upload a new/updated PDF; it is saved to `data/raw/` and indexed
  immediately. Re-uploading the same filename replaces the old chunks (no stale
  duplicates) and keeps a timestamped copy under `_versions/`. Also delete files
  and rebuild the whole index.
* **Website crawler** - trigger a crawl of the college site (see above) and view
  past run history.
* **Questions asked** - every Q&A is logged to the `chat_history` table,
  flagging the ones that could not be answered so you know what to add next.
* **Intents** - the same log grouped by meaning (see above), most-asked first,
  with the answered-from-docs rate per intent.
* **Account** - change the admin password.

Auth is bcrypt password hashing plus HMAC-signed expiring tokens; every
`/api/admin/*` route requires a valid token.

## Database

Everything except the vector index and the raw files on disk lives in
PostgreSQL: `users`, `colleges`, `documents` (+ `document_history`),
`chat_history`, and `crawl_runs` ([backend/db/schema.sql](backend/db/schema.sql)).
Application code goes through [backend/db/repo.py](backend/db/repo.py) -
`auth.py`, `registry.py`, `crawl_log.py`, and `analytics/intents.py` each kept
their existing function signatures and just swapped what's underneath, so
nothing above the repo layer had to change.

### Multi-college schema

The deployment currently serves one college, but `documents`, `chat_history`,
and `crawl_runs` all carry a `college_id` foreign key into `colleges`, and so
does every chunk's metadata in the Chroma vector store (`store.py`'s `_where`
filters vector search, BM25 index building, and deletes by it too) - so
adding a second college later is a data-migration problem, not a rewrite.
[repo.py](backend/db/repo.py)'s `default_college_id()` resolves (and
lazily creates via `ensure_default_college()`) this deployment's one row from
`COLLEGE_NAME`/`COLLEGE_CITY` in config, and every write/read path threads it
through automatically - nothing above `repo.py`/`store.py` needed to know a
college concept exists.

One migration pitfall worth noting: adding `college_id` to an already-populated
table via `ALTER TABLE ... ADD COLUMN` leaves existing rows with `NULL`. If the
upsert path then does `ON CONFLICT (college_id, source)`, Postgres won't match
those old `NULL` rows against a new insert with `college_id = 1` - `NULL` never
equals anything in SQL - so instead of updating in place it inserts a
duplicate. Caught this via `SELECT college_id, COUNT(*) FROM documents GROUP BY
college_id` showing two groups of identical size; fixed by deleting the stale
`NULL` copies (verified first that every `NULL`-group source also existed in
the `college_id = 1` group, so nothing but duplicates was lost) and
backfilling `chat_history`/`crawl_runs` (insert-only tables, so no duplication
there, just un-backfilled history).

One-time setup, as a Postgres superuser (a plain login role can't create its
own database):

```sql
CREATE ROLE admission_bot WITH LOGIN PASSWORD '<pick a password>';
CREATE DATABASE admission_chatbot OWNER admission_bot;
```

Put that password in `.env` as `DB_PASSWORD`, then:

```bash
python -m backend.db.init_db   # applies schema.sql - safe to re-run
```

This dev setup runs PostgreSQL 17 as a **portable, non-service** install
(EDB's binaries zip, not the installer) at `C:\PostgreSQL17`, since installing
it as a Windows service needs admin elevation. Start/stop it manually:

```bash
# start (logs to C:\PostgreSQL17\data\server.log)
C:\PostgreSQL17\pgsql\bin\pg_ctl.exe -D C:\PostgreSQL17\data -l C:\PostgreSQL17\data\server.log start
# stop
C:\PostgreSQL17\pgsql\bin\pg_ctl.exe -D C:\PostgreSQL17\data stop
```

If you have admin rights, installing PostgreSQL normally (as a service, so it
starts on its own) works exactly the same way from the app's side - only
`DB_HOST`/`DB_PORT`/credentials in `.env` need to match.

Switching storage backends means document/user data isn't carried over from
the old JSON files (now deleted) - `ensure_default_admin()` reseeds the
default admin fresh, and re-running `python -m backend.rag.ingest --reset`
repopulates `documents` from whatever's actually in `data/raw/`.

## Why RAG and not fine-tuning

Fine-tuning bakes facts into weights, so every prospectus update would need a
retrain and the model would still invent fees and dates. RAG lets an admin upload a
PDF and have answers change seconds later, with citations. `chat_history` is
exactly the material a *tone/style* LoRA fine-tune needs later - not facts,
which stay with RAG regardless of whether fine-tuning is ever turned on.

## Fine-tuning (QLoRA)

```bash
python -m backend.finetune.prepare_dataset   # build data/processed/finetune_dataset.jsonl
python -m backend.finetune.train             # trains a LoRA adapter (stop the API server first - see below)
python -m backend.finetune.compare           # base vs fine-tuned, side by side, before you trust it
```

`prepare_dataset.py` reuses the exact prompt structure the model sees in
production (system prompt + retrieved context + question, from
`backend/rag/prompt.py`) rather than bare Q&A pairs, and re-runs retrieval at
prep time instead of trusting whatever was logged - so a question logged
before the index changed still gets today's context, and a stale mismatch is
dropped rather than taught. A couple of hand-written borderline-refusal
examples are seeded in (see the module docstring for why - the "no context
found" refusal path in `pipeline.py` never calls the LLM at all, so there's
nothing to mine from logs for the case that actually needs the model to
decide: a weak, related-but-wrong passage).

`train.py` 4-bit-quantizes the base model (QLoRA) and trains LoRA adapter
matrices on top - the adapter alone is ~115MB versus the 6GB base model, and
training and inference can't run at once on an 8GB card, so stop the API
server before running it.

**This first run is demonstration-scale, not production-ready**: 31 examples
(29 real Q&A logged by asking the live pipeline ~30 admission questions,
plus 2 seeded refusals), 6 epochs, training loss down near zero - expected
for a dataset this small, and a real signal to distrust the adapter on
anything *not* checked. `compare.py` checks it on both a trained question and
three held-out ones never seen in training:

| Question | Base | Fine-tuned |
|---|---|---|
| BA Economics seats (trained) | "...runs for 4 years and has: - Grant in Aid: 60 seats - Self Financed: 45 seats" | "BA Economics has 60 seats (Grant in Aid) and 45 seats (Self Financed)." |
| MSc Big Data seats (held out) | "...has: - Self Financed: 35 seats" | "The seat count for the MSc Big Data Analytics programme is 35 (Self Financed)." |
| Capital of France (off-topic) | correct refusal | correct refusal |

Same facts both ways, and the fine-tuned model's more compact, less
bullet-heavy phrasing shows up on the *held-out* question too - a real,
if narrow, style shift rather than memorized answers. Still, don't take that
as proof it generalizes broadly off 31 examples; re-run `compare.py` on more
held-out questions before trusting a retrain, and expect to need real
accumulated traffic (weeks/months, not one seed batch) before this is a
fine-tune worth serving by default.

To actually serve it, set `USE_FINE_TUNED=true` in `.env` and restart the
server - off by default. `backend/llm/qwen.py` loads the base model, then
merges the adapter on top if the flag is set and the adapter directory
exists (raises clearly if the flag is set but no adapter was trained yet).

## Multilingual Query Understanding (Optional Gemini Integration)

The chatbot natively supports multilingual student queries in English, Hindi, Gujarati, and Hinglish (including informal spelling and colloquial abbreviations like *"mare bca ma admission levu che"* or *"12 sci can i take phy?"*):
1. **Local Deterministic Normalizer First**: Fast-paths check canonical courses and structured intents (eligibility, fees, application link, process, documents) directly in < 50ms without consuming external API tokens.
2. **Gemini Flash Fallback**: If a multilingual, misspelled, or ambiguous query is not resolved locally, it can route to Gemini to extract structured JSON `{language, intent, course, qualification, query}`.
3. **Strict Zero-Hallucination Isolation**: Gemini is strictly constrained to query understanding and **never** answers college facts. College facts, fees, eligibility criteria, and application links are retrieved exclusively from PostgreSQL and the local RAG vector store.

## Project Structure

```text
AI-Admission-Chatbot/
├── backend/
│   ├── analytics/        # Intent clustering & query analytics
│   ├── api/              # FastAPI endpoints (chat, streaming SSE, admin, auth)
│   ├── crawler/          # Web crawler with BFS, DFS, and A* strategies
│   ├── db/               # PostgreSQL schema, connection pool, and repository
│   ├── embeddings/       # BAAI/bge-small-en-v1.5 embedding generator
│   ├── finetune/         # QLoRA fine-tuning scripts and datasets
│   ├── llm/              # Local Qwen2.5-3B-Instruct & optional Gemini multilingual provider
│   ├── preprocessing/    # PDF loader, HTML loader, table extraction, ligature repair, chunker
│   ├── rag/              # Two-stage retriever, BM25, reranker, cache, eligibility, pipeline
│   ├── utils/            # Document registry, crawl logging, helper functions
│   └── vectorstore/      # Chroma vector database client and operations
├── data/
│   └── raw/              # Official college prospectus PDFs and crawled website HTML
├── frontend/
│   ├── src/              # React components (ChatPage, Composer, AdminModal, SourcesPanel)
│   ├── index.html        # Main HTML shell
│   ├── package.json      # Frontend dependencies (React 18, Vite 5)
│   └── vite.config.js    # Vite configuration and API proxy
├── .env.example          # Sample environment variables template
├── .gitignore            # Git exclusion rules
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `dev-secret-change-me` | Secret key for JWT admin authentication |
| `ADMIN_USERNAME` | `admin` | Default admin username (seeded on first run) |
| `ADMIN_PASSWORD` | `admin123` | Default admin password |
| `COLLEGE_NAME` | `St. Xavier's College...` | Canonical institution name |
| `DB_HOST` | `127.0.0.1` | PostgreSQL database host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `admission_chatbot` | Database name |
| `DB_USER` | `admission_bot` | PostgreSQL user role |
| `DB_PASSWORD` | - | PostgreSQL password |
| `LLM_PATH` | `./models/Qwen2.5-3B-Instruct`| Path to local Qwen model weights |
| `LLM_DEVICE` | `auto` | Device for LLM (`cuda`, `cpu`, or `auto`) |
| `MAX_NEW_TOKENS` | `180` | Maximum new tokens generated per LLM answer |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model identifier |
| `EMBED_DEVICE` | `cpu` | Device for embedding model |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder reranker model |
| `RERANK_DEVICE` | `cuda` (if available) | Device for reranker |
| `RERANK_BATCH_SIZE` | `8` | Batch size for reranker GPU inference |
| `RERANK_FLOOR` | `0.02` | Minimum noise floor for reranker scores |
| `RERANK_RELATIVE_MARGIN` | `0.5` | Relative score margin against top candidate |
| `GEMINI_API_KEY` | - | (Optional) Google Gemini API key for multilingual intent fallback |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Gemini model variant for intent extraction |
| `LIVE_CRAWL_FALLBACK_ENABLED` | `false` | Enable live crawl on retrieval misses |

## Future Improvements

- **Speech-to-Text & Text-to-Speech**: Voice input and read-aloud capabilities for accessibility.
- **WhatsApp Bot Integration**: Official WhatsApp Business API webhook integration for prospective applicants.
- **Multi-Campus Multi-Tenancy**: Expanding the schema to serve multiple collegiate institutions from a single multi-tenant deployment.
- **Automated Document Expiry Alerts**: Notifying administrators when academic year prospectuses and fee structures are due for renewal.

## Notes / limits

* Broad "list everything" questions (e.g. *all* UG courses) return only what fits in
  the top-5 reranked chunks, since the course list is spread across many pages.
* First request after boot is slow - the 3B model loads into VRAM (~6 GB), and
  the reranker's first use downloads/loads `bge-reranker-base` and builds
  *two* BM25 indexes (one per source) from scratch (a later request only
  pays this cost again if that source's chunk count changed since).
  Steady-state latency is also higher than a single-pool design would be -
  reranking up to 40 candidates (20 per source) instead of one shared 25 -
  which is the real tradeoff for not letting either source starve the other.
* **The source documents themselves can disagree**, and nothing here can
  resolve that: asking "who is the principal" returned "Dr. Sebastian V. A."
  from the prospectus PDF in one test and "Dr. Fr. David K. Roy, SJ" from the
  student handbook in another - both real, both cited, likely a leadership
  transition that the college's own documents haven't been reprinted to
  agree on consistently. The chatbot surfaces whichever grounded answer wins
  reranking; it can't adjudicate which PDF is more current than another.
* `PER_SOURCE_CANDIDATES`, `TOP_K`, `RERANK_FLOOR`, `RERANK_RELATIVE_MARGIN` in
  `backend/config.py` tune the retrieval pipeline - see "Retrieval pipeline"
  above for what each one does and why the cutoff is relative, not fixed.

