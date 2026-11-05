"""Split loader records into embedding-sized chunks with their metadata intact."""

import hashlib
from functools import lru_cache

from backend.config import CHUNK_OVERLAP, CHUNK_SIZE, LOW_MEMORY_MODE


class _PureRecursiveSplitter:
    """Lightweight drop-in replacement for RecursiveCharacterTextSplitter(separators=[...])
    used in LOW_MEMORY_MODE to avoid langchain_text_splitters' ~390 MB import footprint."""

    def __init__(self, chunk_size: int, chunk_overlap: int, separators: list[str]):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def split_text(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        final_chunks: list[str] = []
        separator = separators[-1]
        new_separators: list[str] = []
        for i, s in enumerate(separators):
            if s == "" or s in text:
                separator = s
                new_separators = separators[i + 1 :]
                break

        splits = [s for s in (text.split(separator) if separator else list(text)) if s]
        good_splits: list[str] = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    final_chunks.extend(self._merge(good_splits, separator))
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    final_chunks.extend(self._split(s, new_separators))
        if good_splits:
            final_chunks.extend(self._merge(good_splits, separator))
        return final_chunks

    def _merge(self, splits: list[str], separator: str) -> list[str]:
        docs: list[str] = []
        current_doc: list[str] = []
        total = 0
        sep_len = len(separator)
        for d in splits:
            d_len = len(d)
            if total + d_len + (sep_len if current_doc else 0) > self.chunk_size:
                if current_doc:
                    doc = separator.join(current_doc).strip()
                    if doc:
                        docs.append(doc)
                    while total > self.chunk_overlap or (
                        total + d_len + (sep_len if current_doc else 0) > self.chunk_size and total > 0
                    ):
                        total -= len(current_doc[0]) + (sep_len if len(current_doc) > 1 else 0)
                        current_doc.pop(0)
            current_doc.append(d)
            total += d_len + (sep_len if len(current_doc) > 1 else 0)
        doc = separator.join(current_doc).strip()
        if doc:
            docs.append(doc)
        return docs


@lru_cache(maxsize=1)
def _get_splitter():
    if LOW_MEMORY_MODE:
        return _PureRecursiveSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_records(records: list[dict], doc_type: str, college_id: int) -> list[dict]:
    """records -> [{"id", "text", "metadata"}] ready for the vector store.

    college_id goes into both the metadata (so retrieval can filter by it,
    see backend/rag/retriever.py) and the hash id, since two colleges could
    upload a same-named, same-numbered-page file and would otherwise collide
    on one chunk id and silently overwrite each other's content.
    """
    splitter = _get_splitter()
    chunks: list[dict] = []
    for rec in records:
        for piece in splitter.split_text(rec["text"]):
            piece = piece.strip()
            if len(piece) < 40:
                continue
            meta = {
                "source": rec["source"],
                "doc_type": doc_type,
                "page": rec.get("page", 0),
                "section": rec.get("section", ""),
                "college_id": college_id,
            }
            # Content-hash id makes re-ingesting the same file idempotent.
            digest = hashlib.sha1(
                f"{college_id}|{meta['source']}|{meta['page']}|{piece}".encode("utf-8")
            ).hexdigest()
            chunks.append({"id": digest, "text": piece, "metadata": meta})
    return chunks
