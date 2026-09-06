"""Two-stage retrieval: PDF and website chunks are searched as two SEPARATE
candidate pools, then a cross-encoder reranker decides what's actually worth
answering from.

    question
       |
       +--> normalize (lowercase, collapse whitespace)
       |
       +--> PDF pool:     vector search (doc_type=pdf)     + BM25 (pdf-only index)
       +--> website pool: vector search (doc_type=website) + BM25 (website-only index)
       |
       +--> merge both pools (deduped) --> rerank --> source-priority boost
         --> relative-confidence cutoff --> top TOP_K

Two pools instead of one shared top-k: the PDF and website corpora are very
different sizes (a handful of PDFs vs. ~100 crawled pages), so a single
shared top-k vector search can end up dominated by whichever source has more
chunks, starving the other out before the reranker - which is actually good
at judging relevance - ever gets a chance to see it. Searching each source
separately guarantees both get a fair, fixed-size shot at the candidate pool;
the reranker still makes the final call on which chunks actually answer the
question, regardless of which source they came from, so a PDF-only or
website-only answer isn't favoured just for being in a bigger pool.

Within each source's pool, vector search and BM25 both run (hybrid): pure
embeddings are weak at exact proper-noun lookups (a bare name doesn't embed
distinctively, so nearest-neighbour search can rank the wrong page above the
right one), and BM25 recovers exact/lexical matches embeddings miss.

Live-crawl-on-miss lives in pipeline.py, not here, since that decision needs
the generation result (a passage can pass every check here and still not
actually contain the answer), which this module doesn't see.
"""

import re
import threading

import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from backend.config import (
    PER_SOURCE_CANDIDATES,
    RERANK_BATCH_SIZE,
    RERANK_DEVICE,
    RERANK_FLOOR,
    RERANK_MODEL,
    RERANK_RELATIVE_MARGIN,
    TOP_K,
)
from backend.db import repo
from backend.db.repo import default_college_id
from backend.embeddings.embedder import embed_query
from backend.rag import source_priority
from backend.vectorstore import store

SOURCE_TYPES = ("pdf", "website")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")


def _tokenize(text: str) -> list[str]:
    # Splitting on anything non-alphanumeric means "dr.pravida" (typed with
    # no space after the period) tokenizes to ["dr", "pravida"], same as
    # "Dr. Pravida" in the source document - BM25 scores it on term overlap,
    # so this one normalization gets both proper-noun lookups and ordinary
    # wording variation without needing a separate exact-match code path.
    return _TOKEN_RE.findall(text.lower())


def normalize_query(question: str) -> str:
    """Lowercase and collapse whitespace before embedding/searching. Cheap,
    and most of the real robustness work is already done downstream: BM25
    tokenizes on word boundaries regardless of case/punctuation, and
    bge-small's own tokenizer is fairly case-insensitive already - this just
    keeps the literal string consistent for logging/comparison, and doubles
    as the cache key basis in backend/rag/cache.py."""
    return _WHITESPACE_RE.sub(" ", question.strip().lower())


_reranker_lock = threading.Lock()
_reranker_instance: CrossEncoder | None = None


def _reranker() -> CrossEncoder:
    global _reranker_instance
    if _reranker_instance is None:
        with _reranker_lock:
            if _reranker_instance is None:
                _reranker_instance = CrossEncoder(RERANK_MODEL, device=RERANK_DEVICE)
    return _reranker_instance


def _rerank_predict(pairs: list[list[str]]):
    """CPU reranking measured ~6.3s for a real 40-candidate batch - the
    actual bottleneck of the whole request. GPU cuts that to ~0.4-0.6s, but
    VRAM is tight with Qwen also loaded, so a small batch_size caps peak
    usage (measured ~1.3GB at batch_size=8 vs ~1.7GB at the default 32 - and
    faster, not slower). If the GPU still genuinely runs out of memory
    (longer conversation history, concurrent requests), fall back to CPU for
    this one request rather than failing it outright - slow and correct
    beats fast and broken."""
    model = _reranker()
    try:
        return model.predict(pairs, batch_size=RERANK_BATCH_SIZE)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return model.predict(pairs, batch_size=RERANK_BATCH_SIZE, device="cpu")


class _Bm25Index:
    """In-memory BM25 index over one source's chunks, rebuilt whenever that
    source's chunk count changes (admin upload, crawl, delete). A count
    check is a cheap enough staleness signal at this corpus size - it
    doesn't catch a same-count replace, but ingest_file() always
    deletes-then-adds a document's old chunks first, so a same-count
    in-place edit is rare."""

    def __init__(self, doc_type: str):
        self._doc_type = doc_type
        self._lock = threading.Lock()
        self._count: int | None = None
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict] = []

    def _ensure_built(self) -> None:
        college_id = default_college_id()
        count = store.count(doc_type=self._doc_type, college_id=college_id)
        if count == self._count:
            return
        with self._lock:
            if count == self._count:
                return
            chunks = store.get_all_chunks(doc_type=self._doc_type, college_id=college_id)
            tokenized = [_tokenize(c["text"]) for c in chunks]
            self._bm25 = BM25Okapi(tokenized) if tokenized else None
            self._chunks = chunks
            self._count = count

    def search(self, query_text: str, top_k: int) -> list[dict]:
        self._ensure_built()
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query_text))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"text": self._chunks[i]["text"], "metadata": self._chunks[i]["metadata"]}
            for i in ranked if scores[i] > 0
        ]


_bm25_indexes = {doc_type: _Bm25Index(doc_type) for doc_type in SOURCE_TYPES}


def _source_pool(question: str, embedding: list[float], doc_type: str, pool_size: int) -> list[dict]:
    vector_hits = store.query(
        embedding, top_k=pool_size, doc_type=doc_type, college_id=default_college_id()
    )
    bm25_hits = _bm25_indexes[doc_type].search(question, pool_size)

    seen, pool = set(), []
    for h in vector_hits + bm25_hits:
        key = (h["metadata"].get("source"), h["text"][:80])
        if key in seen:
            continue
        seen.add(key)
        pool.append(h)
    return pool[:pool_size]


def _extract_entity_name(question: str) -> str | None:
    m = re.search(
        r"\b(?:who\s+is|tell\s+me\s+about|about|information\s+about|profile\s+of|dr\.?|prof\.?|mr\.?|ms\.?)\s+([A-Za-z\s]+?)(?:\?|\.|$)",
        question,
        re.I,
    )
    if m:
        name = m.group(1).strip()
        stop = {
            "the", "a", "an", "this", "college", "admission", "course", "fee", "fees",
            "eligibility", "documents", "cutoff", "holiday", "holiday list", "holidays",
            "calendar", "academic calendar", "python", "python programming", "the python programming",
            "syllabus", "curriculum", "notices", "details", "credits", "machine learning",
            "janmashtami", "diwali", "christmas", "events", "event", "upcoming events", "activities",
            "culfest", "sportsfest", "seminar", "workshop", "fest"
        }
        if name and len(name) >= 3 and name.lower() not in stop:
            return name
    return None


def _candidates(question: str) -> list[dict]:
    embedding = embed_query(question)
    seen, merged = set(), []
    for doc_type in SOURCE_TYPES:
        for h in _source_pool(question, embedding, doc_type, PER_SOURCE_CANDIDATES):
            key = (h["metadata"].get("source"), h["text"][:80])
            if key in seen:
                continue
            seen.add(key)
            merged.append(h)

    # Exact entity lookup fallback so faculty members/named individuals aren't missed
    entity = _extract_entity_name(question)
    if entity and len(entity) >= 3:
        college_id = default_college_id()
        ent_lower = entity.lower()
        for doc_type in SOURCE_TYPES:
            for c in store.get_all_chunks(doc_type=doc_type, college_id=college_id):
                if ent_lower in c["text"].lower():
                    key = (c["metadata"].get("source"), c["text"][:80])
                    if key not in seen:
                        seen.add(key)
                        merged.append(c)

    return merged


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Reranked passages, filtered by relative confidence so off-topic
    questions get nothing and terse-but-genuine ones aren't dropped just
    because their absolute score looks unimpressive next to a well-phrased
    question's.
    """
    question = normalize_query(question)
    candidates = _candidates(question)
    if not candidates:
        return []

    entity = _extract_entity_name(question)
    ent_lower = entity.lower() if entity else None

    # If an entity query was asked, check if the exact entity exists in any candidate
    if ent_lower and not any(ent_lower in c["text"].lower() for c in candidates):
        return []

    pairs = [[question, c["text"]] for c in candidates]
    scores = _rerank_predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
        boost = source_priority.boost_for(c["metadata"])

        # Prioritize core faculty/department rosters over general committee lists
        if ent_lower and ent_lower in c["text"].lower():
            txt_lower = c["text"].lower()
            if any(k in txt_lower for k in ["faculty details", "faculty members", "department of", "assistant professor", "coordinator of", "head of the department"]):
                boost *= 10.0  # Strongly prefer the definitive faculty profile chunk
            else:
                boost *= 3.0

        c["boosted_score"] = float(s) * boost

    candidates.sort(key=lambda c: c["boosted_score"], reverse=True)

    top_score = candidates[0]["boosted_score"]
    if top_score < RERANK_FLOOR:
        return []

    cutoff = max(RERANK_FLOOR, top_score * RERANK_RELATIVE_MARGIN)
    return [c for c in candidates if c["boosted_score"] >= cutoff][:top_k]


def as_sources(passages: list[dict]) -> list[dict]:
    """Compact, de-duplicated citation list with verified URLs."""
    seen, sources = set(), []
    for passage in passages:
        meta = passage["metadata"]
        key = (meta.get("source"), meta.get("page"), meta.get("section"))
        if key in seen:
            continue
        seen.add(key)
        src_name = meta.get("source")
        url = repo.get_url_for_source(src_name)
        sources.append({
            "source": src_name,
            "doc_type": meta.get("doc_type"),
            "page": meta.get("page") or None,
            "section": meta.get("section") or None,
            "url": url,
            "score": round(passage.get("rerank_score", 0.0), 3),
            "excerpt": passage["text"][:220].strip() + "...",
        })
    return sources
