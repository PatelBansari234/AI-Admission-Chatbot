"""Document registry + chat log - backed by Postgres (backend/db/repo.py).

Kept as a thin wrapper so callers (backend/rag/ingest.py, backend/api/admin.py,
backend/rag/pipeline.py) didn't need to change when the storage moved from
JSON files to the database.
"""

from backend.db import repo


def load_registry() -> list[dict]:
    return repo.list_documents()


def record_document(
    source: str, doc_type: str, chunks: int, uploaded_by: str = "system",
    status: str = "PENDING", filename: str | None = None,
) -> dict:
    """Insert or update the entry for one source file and keep an upload history."""
    return repo.upsert_document(source, doc_type, chunks, uploaded_by, status=status, filename=filename)



def remove_document(source: str) -> bool:
    return repo.delete_document(source)


def log_chat(
    question: str, answer: str, sources: list[dict], grounded: bool,
    conversation_id: str | None = None,
    owner_id: str | None = None,
) -> None:
    """Persist one Q&A turn - also the raw material for a future LoRA run."""
    trimmed_sources = [
        {"source": s.get("source"), "page": s.get("page"), "section": s.get("section")}
        for s in sources
    ]
    repo.insert_chat(question, answer, grounded, trimmed_sources, conversation_id, owner_id=owner_id)

