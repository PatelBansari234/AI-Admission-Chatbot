"""Structured lookups against course_eligibility - answers "what courses can
I apply for with 12th Science?" and "what's MSc AI's eligibility/fee?" from
a curated table instead of asking Qwen to reconstruct an answer table from
whichever chunks retrieval happened to find (see the README's fee-table
chunk-boundary limitation this is meant to replace).

Scope note: this module only does the lookup. Deciding *when* a question
should use it (intent detection) and merging its result into the answer
prompt alongside RAG evidence is Phase 3.4, not here - callers are expected
to already have a clean course name or stream keyword in hand (e.g. from
query_rewrite.py's course extraction), not a raw, possibly-abbreviated
student question.
"""

import re

from backend.db import repo

_STREAM_RE = re.compile(r"\b(science|commerce|arts|any stream)\b", re.I)


def find_by_course(course_name: str) -> list[dict]:
    """All category rows (grant-in-aid / self-financed, UG / PG) for one
    course. Expects a reasonably clean name - "MSc Artificial Intelligence"
    will match the stored "MSc-Artificial Intelligence", but a raw
    abbreviation like "MSc AI" won't (see repo.find_course_eligibility);
    resolving that is query_rewrite.py's job, upstream of this call."""
    return repo.find_course_eligibility(course_name)


def find_by_stream(text: str) -> list[dict]:
    """Courses whose eligibility mentions a stream named in `text`
    ("I completed 12th Science" -> "science" -> every course whose
    eligibility contains "Science", e.g. "12th Science",
    "12th Science & Commerce"). Only recognizes the vocabulary the source
    data actually uses (Science/Commerce/Arts/Any stream) - deliberately
    narrow rather than a fuzzy/semantic match, since a wrong course
    recommendation is worse than a missed one."""
    match = _STREAM_RE.search(text)
    if not match:
        return []
    return repo.find_courses_by_eligibility(match.group(1))


def lookup(course: str | None, qualification: str | None) -> list[dict]:
    """Combines both lookup paths and dedupes by (course_name, category) -
    a question can name both a course and the student's own qualification
    ("Can a BCA graduate apply for MSc AI?"), and either alone can return
    rows the other also found."""
    rows, seen = [], set()
    for row in (find_by_course(course) if course else []) + (find_by_stream(qualification) if qualification else []):
        key = (row["course_name"], row["category"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def as_sources(rows: list[dict]) -> list[dict]:
    """Same shape as retriever.py's as_sources() so the UI's source list can
    show structured and RAG evidence side by side without a special case."""
    sources = []
    for row in rows:
        excerpt = f"Eligibility: {row['eligibility']}"
        if row.get("fee_amount") is not None:
            excerpt += f" | Fee: ₹{row['fee_amount']}/semester"
        src_doc = row["source_document"]
        url = repo.get_url_for_source(src_doc)
        sources.append({
            "source": src_doc,
            "doc_type": "structured",
            "page": None,
            "section": f"{row['course_name']} ({row['category']})",
            "url": url,
            "score": None,
            "excerpt": excerpt,
        })
    return sources
