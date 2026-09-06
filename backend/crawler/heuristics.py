"""Relevance heuristic for the A* crawl strategy.

A page is worth visiting early if its URL or link text mentions something an
applicant actually searches for. This is the h(n) in A*'s f = g + h: cost g is
modelled separately as a depth penalty in frontier.py.
"""

import re

# (pattern, weight) - checked against the URL path and the anchor text that
# linked to it. Keep this list small and admission-specific rather than
# generic SEO keywords, so it doesn't just reward every page equally.
KEYWORDS: list[tuple[str, float]] = [
    (r"admission", 5.0),
    (r"prospectus", 5.0),
    (r"eligib", 4.0),
    (r"\bfee(s)?\b", 4.0),
    (r"scholarship", 4.0),
    (r"\bhostel", 3.0),
    (r"\bcourse(s)?\b", 3.0),
    (r"\bprogram(me)?s?\b", 3.0),
    (r"\bapply|application\b", 3.0),
    (r"\b(ug|pg)\b", 2.5),
    (r"undergraduate|postgraduate", 2.5),
    (r"syllabus|curriculum", 2.0),
    (r"faculty|department", 1.5),
    (r"contact", 1.0),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), w) for p, w in KEYWORDS]


def score_page(url: str, anchor_text: str = "") -> float:
    haystack = f"{url} {anchor_text}"
    return sum(weight for pattern, weight in _COMPILED if pattern.search(haystack))


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "when", "where",
    "who", "how", "which", "of", "for", "in", "on", "at", "to", "and", "or",
    "do", "does", "did", "give", "me", "tell", "about", "please", "i",
    "want", "know", "information", "there", "this", "that",
}


def build_query_heuristic(query: str):
    """A* heuristic for the on-demand live-crawl fallback (see
    backend/crawler/live_search.py): score a page by how many meaningful
    words from THIS specific question appear in its URL or the anchor text
    that linked to it, instead of the fixed admission-keyword list above.
    Used when the pre-built index has nothing for a question - the crawl
    needs to chase whatever the user actually asked about, not just the
    generic "admission-relevant" pages.
    """
    terms = [
        w for w in re.findall(r"[a-zA-Z0-9]+", query.lower())
        if len(w) >= 3 and w not in _STOPWORDS
    ]
    terms = list(dict.fromkeys(terms))  # dedupe, keep first-seen order
    # Word-boundary, not substring: plain "in" matching let "academic" from a
    # query match inside "academics" in every /academics/departments/... URL
    # (confirmed live: this flooded the frontier with false-positive matches
    # at depth 1, so the budget was gone before reaching the actual
    # depth-2 target, an academic-calendar PDF, even though its URL and
    # anchor text both said "Academic-Calendar" outright).
    patterns = [re.compile(r"\b" + re.escape(term) + r"\b") for term in terms]

    def score(url: str, anchor_text: str = "") -> float:
        haystack = f"{url} {anchor_text}".lower()
        return float(sum(1 for pattern in patterns if pattern.search(haystack)))

    return score
