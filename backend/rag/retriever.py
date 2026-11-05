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

from rank_bm25 import BM25Okapi

from backend.config import (
    LOW_MEMORY_MODE,
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
_reranker_instance = None


def _reranker():
    if LOW_MEMORY_MODE:
        return None
    global _reranker_instance
    if _reranker_instance is None:
        with _reranker_lock:
            if _reranker_instance is None:
                from sentence_transformers import CrossEncoder

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
    import torch

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
        max_s = float(scores[ranked[0]]) if ranked and scores[ranked[0]] > 0 else 1.0
        return [
            {
                "text": self._chunks[i]["text"],
                "metadata": self._chunks[i]["metadata"],
                "bm25_score": float(scores[i]),
                "bm25_max": max_s,
            }
            for i in ranked if scores[i] > 0
        ]


_bm25_indexes = {doc_type: _Bm25Index(doc_type) for doc_type in SOURCE_TYPES}


def _source_pool(question: str, embedding: list[float], doc_type: str, pool_size: int) -> list[dict]:
    vector_hits = store.query(
        embedding, top_k=pool_size, doc_type=doc_type, college_id=default_college_id()
    )
    bm25_hits = _bm25_indexes[doc_type].search(question, pool_size)

    seen_map: dict[tuple, dict] = {}
    pool: list[dict] = []
    for h in vector_hits + bm25_hits:
        key = (h["metadata"].get("source"), h["text"][:80])
        if key in seen_map:
            existing = seen_map[key]
            if "distance" in h and "distance" not in existing:
                existing["distance"] = h["distance"]
            if "bm25_score" in h and "bm25_score" not in existing:
                existing["bm25_score"] = h["bm25_score"]
                existing["bm25_max"] = h.get("bm25_max", 1.0)
            continue
        item = dict(h)
        seen_map[key] = item
        pool.append(item)
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


_QUERY_STOP_WORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "the", "a", "an", "and", "or", "but", "if", "then",
    "this", "that", "these", "those", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "can", "will", "just", "should", "now",
    "sxca", "college", "st", "xavier", "xaviers", "xavier's", "ahmedabad", "tell",
    "give", "info", "information", "please", "me", "much", "many", "i", "you",
}


def _extract_distinctive_query_tokens(question: str) -> list[str]:
    tokens = _tokenize(question)
    return [t for t in tokens if len(t) >= 3 and t not in _QUERY_STOP_WORDS]


def _has_strong_query_alignment(distinctive_tokens: list[str], passage_text: str) -> bool:
    if not distinctive_tokens:
        return False
    passage_tokens = set(_tokenize(passage_text))

    def matches_token(qt: str) -> bool:
        if qt in passage_tokens:
            return True
        base = qt.rstrip("s")
        if len(base) >= 3 and any(pt == base or pt.rstrip("s") == base for pt in passage_tokens if len(pt) >= 3):
            return True
        return False

    matched = [qt for qt in distinctive_tokens if matches_token(qt)]
    if len(distinctive_tokens) <= 2:
        return len(matched) == len(distinctive_tokens)
    return len(matched) >= len(distinctive_tokens) * 0.75


def _low_memory_rerank_scores(
    question: str, distinctive_tokens: list[str], candidates: list[dict]
) -> list[float]:
    """Calibrated hybrid scorer (ChromaDB dense cosine similarity + BM25 lexical
    relevance + query token alignment) used in 512 MB LOW_MEMORY_MODE so the
    1.11 GB BAAI/bge-reranker-base model never has to be loaded into memory."""
    scores: list[float] = []
    for c in candidates:
        text = c.get("text", "")
        passage_tokens = set(_tokenize(text))

        matched = 0
        for qt in distinctive_tokens:
            if qt in passage_tokens:
                matched += 1
            else:
                base = qt.rstrip("s")
                if len(base) >= 3 and any(
                    pt == base or pt.rstrip("s") == base for pt in passage_tokens if len(pt) >= 3
                ):
                    matched += 1

        lex_cov = (matched / len(distinctive_tokens)) if distinctive_tokens else 0.0
        dist = c.get("distance")
        cos_sim = max(0.0, 1.0 - float(dist)) if dist is not None else 0.55
        bm25_norm = float(c.get("bm25_score", 0.0)) / max(float(c.get("bm25_max", 1.0)), 1e-6)

        # Reject off-topic queries that have weak lexical overlap and weak semantic similarity
        if distinctive_tokens:
            if matched == 0 and cos_sim < 0.68:
                scores.append(0.0)
                continue
            if len(distinctive_tokens) >= 3 and lex_cov < 0.34 and cos_sim < 0.68:
                scores.append(0.0)
                continue

        dense_scaled = max(0.0, min(1.0, (cos_sim - 0.45) / 0.40))
        hybrid = (0.45 * dense_scaled) + (0.35 * lex_cov) + (0.20 * bm25_norm)
        if _has_strong_query_alignment(distinctive_tokens, text):
            hybrid = max(hybrid, 0.55 + (0.25 * dense_scaled) + (0.20 * bm25_norm))
        scores.append(round(float(min(0.999, hybrid)), 4))
    return scores


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

    distinctive_tokens = _extract_distinctive_query_tokens(question)
    if LOW_MEMORY_MODE:
        scores = _low_memory_rerank_scores(question, distinctive_tokens, candidates)
    else:
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

    margin_cutoff = top_score * RERANK_RELATIVE_MARGIN
    distinctive_tokens = _extract_distinctive_query_tokens(question)

    retained = []
    for i, c in enumerate(candidates):
        score = c["boosted_score"]
        if score < RERANK_FLOOR:
            continue

        # 1. Preserve the top-ranked result (above floor)
        if i == 0:
            retained.append(c)
            continue

        # 2. Also preserve candidates that have strong lexical/semantic alignment with distinctive query terms
        if _has_strong_query_alignment(distinctive_tokens, c["text"]):
            retained.append(c)
            continue

        # 3. Apply the relative-margin rule only when the lower-ranked passage has no meaningful query alignment
        if score >= margin_cutoff:
            retained.append(c)

    return retained[:top_k]



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
