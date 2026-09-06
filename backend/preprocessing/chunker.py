"""Split loader records into embedding-sized chunks with their metadata intact."""

import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import CHUNK_OVERLAP, CHUNK_SIZE

_splitter = RecursiveCharacterTextSplitter(
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
    chunks: list[dict] = []
    for rec in records:
        for piece in _splitter.split_text(rec["text"]):
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
