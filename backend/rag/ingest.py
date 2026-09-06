"""Build the vector index from the PDFs and saved website source.

    python -m backend.rag.ingest              # ingest everything new/changed
    python -m backend.rag.ingest --reset      # wipe the index and rebuild
    python -m backend.rag.ingest --file x.pdf # re-ingest one file
"""

import argparse
from pathlib import Path

from backend.config import RAW_PDF_DIR, RAW_WEB_DIR
from backend.db.repo import default_college_id
from backend.embeddings.embedder import embed_documents
from backend.preprocessing.chunker import chunk_records
from backend.preprocessing.html_loader import load_html
from backend.preprocessing.pdf_loader import load_pdf
from backend.preprocessing.table_extractor import (
    extract_course_tables,
    strip_table_regions,
    table_chunks,
)
from backend.utils.registry import load_registry, record_document, remove_document
from backend.vectorstore import store

WEB_SUFFIXES = {".txt", ".html", ".htm"}


def _load_pdf_records(path: Path, college_id: int) -> tuple[list[dict], list[dict]]:
    """Returns (flowing_text_records, table_chunk_dicts). Runs
    table_extractor.py's table detection first, so page-level dedup can
    prefer whichever duplicate page actually has usable tables (see
    pdf_loader.dedupe_pages), then strips each captured table's own text
    out of its page's flowing text so the same course/fee data isn't
    indexed twice - once cleanly labelled, once as easily-fragmented
    running text (the exact chunk-boundary failure this phase exists to
    fix). A table whose text can't be confidently located for stripping is
    left in the flowing text rather than risking a wrong cut - see
    strip_table_regions()."""
    tables = extract_course_tables(path)
    prefer_pages = {t["page"] for t in tables}
    records = load_pdf(path, prefer_pages=prefer_pages)

    tables_by_page: dict[int, list[dict]] = {}
    for t in tables:
        tables_by_page.setdefault(t["page"], []).append(t)
    for rec in records:
        page_tables = tables_by_page.get(rec["page"])
        if page_tables:
            rec["text"] = strip_table_regions(rec["text"], page_tables)

    return records, table_chunks(tables, path.name, college_id)


def ingest_file(path: Path, uploaded_by: str = "system", replace: bool = True) -> dict:
    """Index one file and return {"source", "chunks"}."""
    college_id = default_college_id()
    extracted_table_chunks: list[dict] = []

    if path.suffix.lower() == ".pdf":
        doc_type = "pdf"
        records, extracted_table_chunks = _load_pdf_records(path, college_id)
    elif path.suffix.lower() in WEB_SUFFIXES:
        records, doc_type = load_html(path), "website"
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    chunks = chunk_records(records, doc_type, college_id) + extracted_table_chunks
    if not chunks:
        raise ValueError(f"No usable text extracted from {path.name}")

    # Replacing an updated prospectus must drop the stale chunks first.
    if replace:
        store.delete_source(path.name, college_id)

    embeddings = embed_documents([c["text"] for c in chunks])
    store.add_chunks(chunks, embeddings)
    record_document(path.name, doc_type, len(chunks), uploaded_by)
    return {"source": path.name, "doc_type": doc_type, "chunks": len(chunks)}


def ingest_all(reset: bool = False) -> list[dict]:
    if reset:
        print("Resetting collection...")
        store.reset_collection()

    # rglob (not glob) so crawler output under raw/pdfs/crawled/ and
    # raw/websie/crawled/ is picked up too; "_versions" holds admin-upload
    # backups and must never be re-indexed.
    targets = sorted(
        p for p in RAW_PDF_DIR.rglob("*.pdf") if "_versions" not in p.parts
    )
    targets += sorted(
        p for p in RAW_WEB_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in WEB_SUFFIXES and "_versions" not in p.parts
    )

    results = []
    for path in targets:
        print(f"Ingesting {path.name} ...")
        try:
            result = ingest_file(path)
        except Exception as exc:
            print(f"  ! skipped: {exc}")
            continue
        print(f"  -> {result['chunks']} chunks")
        results.append(result)

    # A full scan just ran, so it's authoritative: drop any registry entry
    # (and its chunks) whose file is no longer on disk - e.g. removed by hand
    # rather than through the admin delete endpoint.
    on_disk = {path.name for path in targets}
    for entry in load_registry():
        if entry["source"] not in on_disk:
            print(f"Pruning stale entry: {entry['source']}")
            store.delete_source(entry["source"], default_college_id())
            remove_document(entry["source"])

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the admission-chatbot index")
    parser.add_argument("--reset", action="store_true", help="wipe the index first")
    parser.add_argument("--file", help="ingest a single file path")
    args = parser.parse_args()

    if args.file:
        result = ingest_file(Path(args.file))
        print(f"{result['source']}: {result['chunks']} chunks")
    else:
        ingest_all(reset=args.reset)

    print("Index stats:", store.stats())


if __name__ == "__main__":
    main()
