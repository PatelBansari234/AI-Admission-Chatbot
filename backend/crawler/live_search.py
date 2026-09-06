"""On-demand live crawl fallback for the RAG retriever.

When the pre-built index has nothing for a question, retriever.py calls
maybe_live_crawl() before giving up: it runs a small, targeted crawl of the
college site right now, guided by the question's own words via A*
(heuristics.build_query_heuristic), instead of just returning "I don't have
that information." A fresh page found this way is indexed immediately, so
the very next retrieval attempt (still within the same request) can use it.

This runs inside the request (chat/chat_stream are sync FastAPI routes, so
FastAPI already thread-pools them - a blocking crawl here doesn't stall other
users), which trades a slower answer on a genuine index gap for not needing
a separate background job system. Two safeguards keep one gap in the index
from turning into unbounded traffic against the college's server:

- a per-question cooldown, so asking the same unanswered thing twice in a
  row doesn't re-crawl every time
- a global lock, so concurrent different questions that all miss don't each
  launch their own simultaneous crawl - only one live crawl runs at a time,
  everyone else just falls through to "I don't have that information" until
  it's free again
"""

import re
import threading
import time

from backend.config import ALLOWED_CRAWL_DOMAIN
from backend.crawler.crawler import crawl

LIVE_MAX_PAGES = 20
LIVE_MAX_DEPTH = 3
LIVE_DELAY = 0.4

_COOLDOWN_SECONDS = 600  # don't re-crawl for the same failed lookup within 10 min
_recent_attempts: dict[str, float] = {}
_crawl_lock = threading.Lock()


def _normalize(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def maybe_live_crawl(question: str) -> bool:
    """Returns True if a live crawl actually ran (whether or not it found
    anything new) - the caller should retry retrieval only when this is True."""
    key = _normalize(question)
    if not key:
        return False

    now = time.monotonic()
    last = _recent_attempts.get(key)
    if last is not None and now - last < _COOLDOWN_SECONDS:
        return False

    if not _crawl_lock.acquire(blocking=False):
        return False  # another question's live crawl is already running

    try:
        _recent_attempts[key] = now
        crawl(
            f"https://{ALLOWED_CRAWL_DOMAIN}/",
            strategy="astar",
            max_pages=LIVE_MAX_PAGES,
            max_depth=LIVE_MAX_DEPTH,
            delay=LIVE_DELAY,
            query=question,
        )
        return True
    finally:
        _crawl_lock.release()
