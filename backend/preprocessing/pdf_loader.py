"""Extract clean, page-tagged text out of a prospectus PDF."""

import re
from pathlib import Path

from pypdf import PdfReader

from backend.preprocessing.ligatures import repair_ligatures

# Ligatures and typographic junk that survive PDF extraction and confuse the model.
_REPLACEMENTS = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ", "•": "- ",
}


def clean_text(text: str) -> str:
    for bad, good in _REPLACEMENTS.items():
        text = text.replace(bad, good)
    # Rebuild ti/tt/ft ligatures the font exported as U+0000.
    text = repair_ligatures(text)
    # Words broken across a line by a hyphen: "admis-\nsion" -> "admission"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dedupe_pages(pages: list[dict], prefer_pages: set[int] | None = None) -> list[dict]:
    """Collapses byte-identical duplicate pages down to one representative
    per group of identical text. Some prospectus/brochure PDFs are
    print-shop imposition files with the same content laid out on several
    physical pages for cutting/binding - confirmed on
    SXCA-Admission-Brochure-2526.pdf, where pypdf's extracted text is
    identical across two separate groups of 4 pages each. Indexed as-is,
    that's the same content chunked 3-4x over.

    `prefer_pages` (e.g. pages table_extractor.py found a genuine table on)
    is used to break ties within a duplicate group, instead of always
    keeping the first occurrence - duplicate pages don't always render
    identically enough for table detection to work on all of them; keeping
    whichever page actually has usable tables matters more than keeping
    page 1 by default.
    """
    prefer_pages = prefer_pages or set()
    best_for_text: dict[str, dict] = {}
    for rec in pages:
        current = best_for_text.get(rec["text"])
        if current is None or (rec["page"] in prefer_pages and current["page"] not in prefer_pages):
            best_for_text[rec["text"]] = rec
    kept = list(best_for_text.values())
    kept.sort(key=lambda r: r["page"])
    return kept


def load_pdf(path: Path, prefer_pages: set[int] | None = None) -> list[dict]:
    """Return one record per (deduplicated) page: {"text", "source", "page"}."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            # A single malformed page shouldn't kill the whole ingest.
            continue
        text = clean_text(raw)
        # Skip near-empty pages (cover art, dividers) - they add retrieval noise.
        if len(text) < 60:
            continue
        pages.append({"text": text, "source": path.name, "page": i})
    return dedupe_pages(pages, prefer_pages)
