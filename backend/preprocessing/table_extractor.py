"""Table-aware extraction for PDFs where pdfplumber can recover genuine
course/eligibility/fee table grids, alongside the flowing-text extraction
pdf_loader.py already does. Exists to fix two problems, both demonstrated
on data/raw/pdfs/SXCA-Admission-Brochure-2526.pdf - not assumed:

1. The brochure is a print-shop imposition file: pypdf's flowing text is
   byte-identical across two groups of pages (verified - pages {1,6,7,8}
   are one exact-duplicate group, {2,3,4,5} another). Indexed as-is, that's
   the same content chunked 4x over. Page-level dedup handles this - see
   dedupe_pages() in pdf_loader.py.
2. Even on a deduplicated page, pypdf's flowing text runs a 3-column
   magazine layout (two prose columns alongside a narrower fee-table
   column) into one linear stream, so a character-count text splitter has
   no way to know which category heading a course's row falls under, or to
   keep a row's course/eligibility/fee cells together. This is the exact
   chunk-boundary failure documented in README.md's structured-eligibility
   section. pdfplumber's lattice-based table detection recovers the real
   grid instead - verified empirically, not assumed to work: on this PDF,
   the *duplicate* pages produce zero usable tables (their gridlines
   apparently don't survive whatever produced the duplicate rendering),
   while the one representative page pdfplumber can parse produces four
   clean, correctly-separated tables matching the hand-transcribed data
   from backend/db/seed_eligibility.py cell-for-cell.

Never fabricates: a section heading, column label, or cell value that
doesn't visibly appear on the page is left empty, not guessed.
"""

import hashlib
import re
from pathlib import Path

import pdfplumber

# pdf_loader.py's clean_text() already normalizes these in the flowing text
# an anchor gets matched against; pdfplumber's own extraction (used here)
# doesn't go through that normalization, so an anchor built straight from
# pdfplumber text can silently fail to match otherwise-identical flowing
# text over a dash/quote-style difference alone.
_DASH_QUOTE_NORMALIZE = {"–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"'}

_NUMBER_RE = re.compile(r"\d[\d,]*")

# Generic column-caption words this brochure's layout repeats near a table
# (e.g. a floating "Programme / Eligibility / *Fees per Sem / Email"
# caption line sitting between a table's real section title and the table
# itself) - a candidate heading line made up mostly of these isn't a
# section identity, just a caption, and the nearest-line heuristic below
# needs to see past it to the real title further up.
_CAPTION_WORDS = {
    "programme", "programmes", "eligibility", "fees", "fee", "per", "sem",
    "email", "male", "female", "course",
}


def _clean_cell(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _looks_like_data(text: str) -> bool:
    """True if `text` contains a number or an email - i.e. it reads like a
    table row, not a section/column heading."""
    return bool(_NUMBER_RE.search(text)) or "@" in text


_FEE_LIKE_RE = re.compile(r"\d{1,3}(?:,\d{3})+|\d{4,}")  # "6,655" / "38,000" / "50000"
# A prior-qualification stream or degree abbreviation, as this corpus's real
# eligibility columns actually write them: "12th Science", "BSc Chemistry",
# "BA Psychology", "BCA", "B.Com", etc.
_ELIGIBILITY_LIKE_RE = re.compile(
    r"\b12th\b|\bB\.?\s?(Sc|A|Com|CA|E|PA)\b|\bM\.?\s?(Sc|A|Com)\b", re.I
)


def _is_data_table(rows: list[list]) -> bool:
    """Filters out three different kinds of false positive pdfplumber's
    lattice detector finds across this corpus's varied PDFs:

    1. Decorative grids traced from background rectangles/logos on the
       heavily-designed admission brochure - empty or near-empty (0-2
       populated cells out of dozens), vs. a real table's 20-50+.
    2. Genuinely tabular content that has nothing to do with admissions -
       confirmed on a real document: a syllabus PDF's "Unit I / Unit II"
       reading-list layout has plenty of populated cells and would pass a
       populated-cell-count check alone, producing nonsense chunks like
       "Course: I, Unit: I, Title/Author/Topic: ...".
    3. A fee-shaped number alone isn't enough either - confirmed on a real
       document: the 150+ page Handbook's course-structure/curriculum
       tables (paper codes like "PBD - 1804"), milestone timelines
       ("1955 - 1965"), and contact-info blocks all contain 4+ digit
       numbers and passed a fee-number-only check, producing over 500
       false-positive chunks from that one file alone. Phase 3.2 exists to
       fix course/eligibility/FEE tables specifically, so require BOTH a
       fee-shaped number AND an eligibility-shaped value (a stream like
       "12th Science" or a degree abbreviation like "BSc"/"BA"/"BCA") -
       the actual, distinguishing signature of an admission table, not
       "contains some big number."
    """
    if len(rows) < 2:
        return False
    populated = sum(1 for row in rows for cell in row if cell and cell.strip())
    if populated < 6:
        return False
    cells = [cell for row in rows for cell in row if cell]
    has_fee = any(_FEE_LIKE_RE.search(c) for c in cells)
    has_eligibility = any(_ELIGIBILITY_LIKE_RE.search(c) for c in cells)
    return has_fee and has_eligibility


def _section_for_table(table, page_words: list[dict]) -> str:
    """Section heading for a table. Two strategies, in order:

    1. The table's own first row, if it's a single populated cell that
       doesn't look like data - e.g. "Self-Financed Programmes - PG" is its
       own row in this brochure's PG table.
    2. A heading-like line positioned just above the table AND inside its
       own column (matched by x-range, not just y-position) - this
       brochure runs a 3-column layout where a neighbouring prose column
       sits at nearly the same height as a table heading, so y-position
       alone finds the wrong text; restricting to the table's own x-range
       filters that out.

    Returns "" if neither finds a confident match - never a guess.
    """
    rows = table.extract()
    first_nonempty = [c for c in rows[0] if c and c.strip()]
    if len(first_nonempty) == 1 and not _looks_like_data(first_nonempty[0]):
        return _clean_cell(first_nonempty[0])

    # Nearest line first, but a line that's mostly column-caption
    # vocabulary (e.g. "Programme Eligibility *Fees per Sem Email") isn't a
    # section identity - keep looking further up for one that isn't.
    for words in _lines_above(table, page_words):
        if _is_caption_line(words):
            continue
        return " ".join(w["text"] for w in words)
    return ""


def _lines_above(table, page_words: list[dict]) -> list[list[dict]]:
    """Word-dicts (text + x-position) for text lines positioned just above
    `table` AND inside its own column (x-range) - restricting to the
    table's x-range matters because this brochure's 3-column layout puts a
    neighbouring prose column at nearly the same height as a table's own
    heading text. Returns lines nearest the table first; keeps x-position
    (not just text) so a caption line's words can later be grouped into
    column labels by their horizontal position."""
    x0, top, x1, _ = table.bbox
    candidates = [
        w for w in page_words
        if top - 40 < w["top"] < top - 1
        and w["x0"] >= x0 - 5 and w["x1"] <= x1 + 5
    ]
    lines: dict[int, list[dict]] = {}
    for w in candidates:
        key = round(w["top"] / 3) * 3
        lines.setdefault(key, []).append(w)
    return [lines[t] for t in sorted(lines, reverse=True)]


def _is_caption_line(words: list[dict]) -> bool:
    texts = [w["text"] for w in words]
    line = " ".join(texts)
    if _looks_like_data(line) or len(line) > 90 or not texts:
        return False
    caption_hits = sum(1 for t in texts if t.strip(".,()-*").lower() in _CAPTION_WORDS)
    return caption_hits / len(texts) > 0.5


def _caption_header_near(table, page_words: list[dict], n_cols: int) -> list[str] | None:
    """When a table has no column-header row of its own (see
    _columns_and_rows), this brochure sometimes has a floating caption line
    above it instead ("Programme Eligibility *Fees per Sem Email") - the
    same line _section_for_table() deliberately skips past when looking for
    a section title. Reused here for the opposite purpose: group that
    caption's words into `n_cols` column labels by their x-position gaps
    (a multi-word label like "*Fees per Sem" has small gaps between its own
    words and a larger gap before the next label), rather than leave every
    column unlabelled when the real header text is right there on the
    page. Falls back to None (never a guessed label) if the word count
    doesn't look like it lines up with the table's columns."""
    for words in _lines_above(table, page_words):
        if not _is_caption_line(words):
            continue
        ordered = sorted(words, key=lambda w: w["x0"])
        groups: list[list[dict]] = [[ordered[0]]]
        for w in ordered[1:]:
            gap = w["x0"] - groups[-1][-1]["x1"]
            if gap > 15:
                groups.append([w])
            else:
                groups[-1].append(w)
        if len(groups) != n_cols:
            return None
        return [" ".join(w["text"] for w in g).strip(".,()-*") for g in groups]
    return None


def _columns_and_rows(
    rows: list[list], section: str, table, page_words: list[dict],
) -> tuple[list[str], list[list]]:
    """Splits off the section-label row (already consumed) and a genuine
    column-header row (kept only if it doesn't look like a data row). If
    the table has neither, tries a floating caption line above it
    (_caption_header_near) before falling back to unlabelled "Column N"."""
    data_rows = rows
    first_nonempty = [c for c in rows[0] if c and c.strip()]
    if len(first_nonempty) == 1 and _clean_cell(first_nonempty[0]) == section:
        data_rows = rows[1:]

    columns = None
    if data_rows:
        joined = " ".join(c or "" for c in data_rows[0])
        if joined.strip() and not _looks_like_data(joined):
            columns = [_clean_cell(c) for c in data_rows[0]]
            data_rows = data_rows[1:]

    n_cols = len(rows[0])
    if not columns:
        columns = [""] * n_cols
    if any(not c for c in columns):
        caption = _caption_header_near(table, page_words, n_cols)
        if caption:
            columns = [c or caption[i] for i, c in enumerate(columns)]
    columns = [c if c else f"Column {i + 1}" for i, c in enumerate(columns)]
    if not columns[0] or columns[0] == "Column 1":
        columns[0] = "Course"
    return columns, data_rows


def extract_course_tables(path: Path) -> list[dict]:
    """Returns one entry per genuine table found:
    {"page": int, "section": str, "columns": [...], "rows": [{col: val, ...}]}

    A row with no course name (first cell empty) is dropped - there's
    nothing to anchor it to, and keeping it risks a blank-labelled fact
    later. Column labels default to "Column N" rather than a guessed name
    when the source has no header row for that position.
    """
    results = []
    with pdfplumber.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = None
            for table in page.find_tables():
                rows = table.extract()
                if not _is_data_table(rows):
                    continue
                if words is None:
                    words = page.extract_words()

                section = _section_for_table(table, words)
                columns, data_rows = _columns_and_rows(rows, section, table, words)

                clean_rows = []
                for row in data_rows:
                    cells = [_clean_cell(c) for c in row]
                    if not cells or not cells[0]:
                        continue
                    clean_rows.append(dict(zip(columns, cells)))

                if clean_rows:
                    results.append({
                        "page": page_num,
                        "section": section,
                        "columns": columns,
                        "rows": clean_rows,
                    })
    return results


def strip_table_regions(text: str, tables: list[dict]) -> str:
    """Removes each table's row text from `text` (the page's flowing-text
    extraction) so a course's fee/eligibility isn't ALSO indexed as messy,
    unlabelled flowing text once it has a clean structured chunk from
    format_table_chunk(). Anchored on values actually found in the table
    (section heading, course names, the last row's last cell) - not on any
    wording hardcoded to this one document - so this works for any PDF
    table_extractor.py finds a genuine table in, not just this brochure.

    If an anchor can't be located in `text`, that table's region is left in
    place rather than risking a truncation at the wrong point: a harmless,
    redundant flowing-text chunk is a safer failure than an incorrectly cut
    one.
    """
    def _find(anchor: str, start_at: int = 0) -> tuple[int, int]:
        """(start, end) of `anchor` in `text`, tolerant of the flowing-text
        extraction preserving a literal line break where pdfplumber's
        cell-based extraction joined the same words with a plain space
        (e.g. a wrapped cell "BSc-Biochem.\\nwith Voc. BT" vs the cleaned
        cell text "BSc-Biochem. with Voc. BT") - matching on a literal
        substring alone misses exactly this case, confirmed on real rows
        from this brochure. Returns (-1, -1) if truly not found."""
        for bad, good in _DASH_QUOTE_NORMALIZE.items():
            anchor = anchor.replace(bad, good)
        pattern = r"\s+".join(re.escape(tok) for tok in anchor.strip().split())
        match = re.compile(pattern).search(text, start_at)
        return (match.start(), match.end()) if match else (-1, -1)

    for table in tables:
        anchors = [table["section"]] if table["section"] else []
        anchors += [row.get("Course") for row in table["rows"] if row.get("Course")]
        starts = [s for s, _ in (_find(a) for a in anchors if a) if s != -1]
        if not starts:
            continue
        start = min(starts)

        # Anchor the end search on the LAST ROW'S OWN course name first,
        # then look for its last cell value after that point - searching
        # for the last cell's value directly from `start` finds an EARLIER
        # row's value instead when two rows share one (e.g. several
        # courses in the same department sharing one contact email),
        # confirmed on this brochure's Government Aided Science table
        # ("chemistry@sxca.edu.in" appears on 3 different rows).
        last_row = table["rows"][-1]
        last_course = last_row.get("Course")
        row_start, _ = _find(last_course, start) if last_course else (-1, -1)
        search_from = row_start if row_start != -1 else start

        last_values = [v for v in last_row.values() if v]
        end_anchor = last_values[-1] if last_values else None
        _, end = _find(end_anchor, search_from) if end_anchor else (-1, -1)
        if end == -1:
            continue
        text = text[:start] + text[end:]
    return text


def format_table_chunk(document: str, table: dict, row: dict) -> str:
    """One row -> one self-contained chunk, in the
    [DOCUMENT]/[PAGE]/[SECTION] + labelled-field form - every field a
    retriever sees is explicitly labelled, so "eligibility" and "fee"
    can't be silently swapped the way free-flowing table text could be."""
    lines = [f"[DOCUMENT: {document}]", f"[PAGE: {table['page']}]"]
    if table["section"]:
        lines.append(f"[SECTION: {table['section']}]")
    lines.append("")
    for col, val in row.items():
        if val:
            lines.append(f"{col}: {val}")
    return "\n".join(lines)


def table_chunks(tables: list[dict], source: str, college_id: int) -> list[dict]:
    """One row -> one chunk, never split (a logical table row is never
    broken across chunks, unlike the character-count splitter used for
    prose). Same {"id", "text", "metadata"} shape chunker.py's
    chunk_records() produces, plus "extraction": "table" so retrieval/
    debugging can tell a structured chunk from a generic flowing-text one -
    everything else (doc_type, college_id, source, page, section) matches
    what the rest of the pipeline already filters and scopes on, so no
    changes were needed in retriever.py or vectorstore/store.py."""
    chunks = []
    doc_row_index = 0  # unique across every table in the document, not just within one
    for table in tables:
        for row in table["rows"]:
            text = format_table_chunk(source, table, row)
            meta = {
                "source": source,
                "doc_type": "pdf",
                "page": table["page"],
                "section": table["section"],
                "college_id": college_id,
                "extraction": "table",
            }
            # The identifying value is always the FIRST column - "Course"
            # only when a table had no header of its own (see
            # _columns_and_rows); some documents' own tables label it
            # "Programme" or "Program Name" instead. Hardcoding the key
            # "Course" here previously meant row.get("Course") silently
            # returned "" for every row in those tables, collapsing every
            # row in the same section onto the same hash - confirmed as the
            # cause of real Chroma "duplicate IDs" upsert failures that
            # skipped 35 files' worth of content entirely during a full
            # reingest. doc_row_index is an extra tiebreaker, unique across
            # the whole document (not reset per table) - confirmed needed:
            # two separate same-page tables with no section heading and
            # identical first-row values ("Unit I" appearing in two
            # different tables) still collided with a per-table index.
            first_value = next(iter(row.values()), "")
            digest = hashlib.sha1(
                f"{college_id}|{source}|{table['page']}|{table['section']}|{first_value}|{doc_row_index}".encode("utf-8")
            ).hexdigest()
            chunks.append({"id": digest, "text": text, "metadata": meta})
            doc_row_index += 1
    return chunks
