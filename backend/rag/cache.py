"""In-memory cache for repeated first-turn questions.

Retrieval (two BM25 indexes + vector search across both) plus reranking
plus LLM generation is the dominant cost of every request - several seconds
to tens of seconds. For a college FAQ-style chatbot, many students ask
close to the literal same question ("who is the principal", "BA Economics
seats"), so caching the full answer for an exact (normalized) repeat is a
much bigger win than optimizing any single step of that pipeline - a cache
hit skips all of it and returns near-instantly.

Scoped to fresh questions only (no conversation history): the same text
asked as a follow-up ("and the fees?") can mean something different
depending on what came before it, so those always go through the full
pipeline. Invalidated whenever the total indexed chunk count changes (a
document upload, crawl, or delete), so a stale answer can't outlive the
content it was grounded in - reuses the same staleness signal
backend/rag/retriever.py's BM25 indexes already check.
"""

import threading
from collections import OrderedDict

from backend.vectorstore import store

_MAX_ENTRIES = 200

_lock = threading.Lock()
_cache: "OrderedDict[str, dict]" = OrderedDict()
_cache_chunk_count: int | None = None


def _check_staleness() -> None:
    global _cache_chunk_count
    count = store.stats()["chunks"]
    if count != _cache_chunk_count:
        with _lock:
            _cache.clear()
            _cache_chunk_count = count


def get(key: str) -> dict | None:
    _check_staleness()
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)  # LRU: mark as recently used
            return _cache[key]
    return None


def put(key: str, value: dict) -> None:
    _check_staleness()
    with _lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > _MAX_ENTRIES:
            _cache.popitem(last=False)  # evict least-recently-used
