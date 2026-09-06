"""Persistent Chroma collection holding every prospectus / website chunk."""

from functools import lru_cache

import chromadb
from chromadb.config import Settings

from backend.config import COLLECTION_NAME, VECTOR_DB_DIR


@lru_cache(maxsize=1)
def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def get_collection():
    return get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _where(doc_type: str | None = None, college_id: int | None = None) -> dict | None:
    """Combines whichever metadata filters are given into one Chroma `where`
    clause - a single condition is a plain dict, more than one needs `$and`."""
    conditions = []
    if doc_type:
        conditions.append({"doc_type": doc_type})
    if college_id is not None:
        conditions.append({"college_id": college_id})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def add_chunks(chunks: list[dict], embeddings: list[list[float]]) -> int:
    """Upsert so re-ingesting an unchanged file is a no-op instead of a duplicate."""
    if not chunks:
        return 0
    collection = get_collection()
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        embeddings=embeddings,
    )
    return len(chunks)


def delete_source(source: str, college_id: int | None = None) -> None:
    """Remove every chunk that came from one file (used when a PDF is
    replaced). Scope by college_id too once there's more than one college -
    otherwise two colleges' same-named files would delete each other's
    chunks."""
    where = {"source": source} if college_id is None else {"$and": [{"source": source}, {"college_id": college_id}]}
    get_collection().delete(where=where)


def query(
    embedding: list[float], top_k: int, doc_type: str | None = None, college_id: int | None = None
) -> list[dict]:
    """doc_type restricts the search to one source ("pdf" or "website") -
    used to run separate PDF/website candidate pools in parallel so a much
    larger source can't crowd the smaller one out of a single shared top-k.
    college_id restricts it to one college once there's more than one."""
    res = get_collection().query(
        query_embeddings=[embedding],
        n_results=top_k,
        where=_where(doc_type, college_id),
        include=["documents", "metadatas", "distances"],
    )
    if not res["ids"] or not res["ids"][0]:
        return []
    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def get_all_chunks(doc_type: str | None = None, college_id: int | None = None) -> list[dict]:
    """Every chunk currently indexed (optionally restricted by source and/or
    college) - powers the in-memory, per-source BM25 lexical indexes in
    backend/rag/retriever.py, since building one from scratch needs the full
    corpus. Fine at this corpus size (thousands of chunks); would need a real
    inverted index (e.g. Postgres full-text search) well before this scales
    to tens of thousands.
    """
    data = get_collection().get(
        where=_where(doc_type, college_id),
        include=["documents", "metadatas"],
    )
    return [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(data["documents"], data["metadatas"])
    ]


def count(doc_type: str | None = None, college_id: int | None = None) -> int:
    """Cheap staleness check for the per-source BM25 indexes in
    retriever.py - fetches only ids, not the documents/metadatas
    get_all_chunks() would return."""
    where = _where(doc_type, college_id)
    if where is None:
        return get_collection().count()
    result = get_collection().get(where=where, include=[])
    return len(result["ids"])


def stats() -> dict:
    collection = get_collection()
    return {"collection": COLLECTION_NAME, "chunks": collection.count()}


def reset_collection() -> None:
    try:
        get_client().delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    get_collection()
