"""Admin routes: upload/replace prospectus PDFs and manage the indexed dataset."""

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

import torch
from backend.api.auth import change_password, require_admin
from backend.config import (
    ALLOWED_CRAWL_DOMAIN,
    COLLEGE_NAME,
    COLLECTION_NAME,
    CRAWL_DEFAULT_DELAY,
    CRAWL_DEFAULT_MAX_DEPTH,
    CRAWL_DEFAULT_MAX_PAGES,
    CRAWL_HARD_MAX_DEPTH,
    CRAWL_HARD_MAX_PAGES,
    DB_HOST,
    DB_NAME,
    DB_PORT,
    LLM_DEVICE,
    LLM_PATH,
    RAW_PDF_CRAWL_DIR,
    RAW_PDF_DIR,
    RAW_WEB_CRAWL_DIR,
    RAW_WEB_DIR,
)
from backend.analytics.intents import compute_intents
from backend.crawler.crawler import crawl as run_crawl
from backend.crawler.frontier import FRONTIERS
from backend.db import repo
from backend.rag.ingest import WEB_SUFFIXES, ingest_all, ingest_file
from backend.utils.admin_reviews import get_all_reviews, get_review, set_review_status
from backend.utils.crawl_log import load_runs
from backend.utils.registry import load_registry, remove_document
from backend.vectorstore import store

router = APIRouter(prefix="/api/admin", tags=["admin"])

ALLOWED_SUFFIXES = {".pdf"} | WEB_SUFFIXES
MAX_UPLOAD_BYTES = 60 * 1024 * 1024


def _safe_name(filename: str) -> str:
    """Strip any path components so an upload can't escape the data directory."""
    name = filename.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".")
    if not name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return name


@router.get("/documents")
def list_documents(status: str | None = None, _: str = Depends(require_admin)):
    docs = repo.list_documents(status_filter=status)
    return {"documents": docs, "index": store.stats()}


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    admin: str = Depends(require_admin),
):
    """Save a new/updated file into dataset and record as PENDING (not indexed into RAG yet)."""
    name = _safe_name(file.filename or "")
    suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Only {', '.join(sorted(ALLOWED_SUFFIXES))} files are accepted",
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 60 MB")

    target_dir = RAW_PDF_DIR if suffix == ".pdf" else RAW_WEB_DIR
    target = target_dir / name

    replaced = target.exists()
    if replaced:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup_dir = target_dir / "_versions"
        backup_dir.mkdir(exist_ok=True)
        shutil.copy2(target, backup_dir / f"{target.stem}.{stamp}{target.suffix}")

    target.write_bytes(payload)

    try:
        result = ingest_file(target, uploaded_by=admin, replace=True, status="PENDING")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse uploaded file: {exc}")

    return {
        "message": f"{'Replaced' if replaced else 'Added'} {name} and stored as PENDING (awaiting admin approval)",
        "replaced": replaced,
        **result,
        "index": store.stats(),
    }


def _find_target_file(source: str) -> Path | None:
    p = RAW_PDF_DIR / source
    if p.exists():
        return p
    for found in RAW_PDF_DIR.rglob(source):
        if found.is_file():
            return found
    p_web = RAW_WEB_DIR / source
    if p_web.exists():
        return p_web
    for found in RAW_WEB_DIR.rglob(source):
        if found.is_file():
            return found
    return None


@router.post("/documents/{doc_id}/approve")
def approve_document(doc_id: str, admin: str = Depends(require_admin)):
    """Approve a PENDING/REJECTED/ARCHIVED document and activate it in RAG vector search."""
    doc = repo.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    target = _find_target_file(doc["source"])
    if not target:
        raise HTTPException(status_code=404, detail=f"Document file '{doc['source']}' not found on disk")

    try:
        res = ingest_file(target, uploaded_by=doc.get("uploaded_by") or admin, replace=True, status="ACTIVE")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not index document chunks: {exc}")

    updated = repo.update_document_status(doc["id"], status="ACTIVE", reviewed_by=admin)
    return {
        "message": f"Document '{doc['source']}' approved and activated in production RAG.",
        "document": updated or res,
        "index": store.stats(),
    }


@router.post("/documents/{doc_id}/reject")
def reject_document(doc_id: str, admin: str = Depends(require_admin)):
    """Reject a document, preventing RAG retrieval and deleting its chunks from vector store."""
    doc = repo.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    store.delete_source(doc["source"], repo.default_college_id())
    updated = repo.update_document_status(doc["id"], status="REJECTED", reviewed_by=admin)
    return {
        "message": f"Document '{doc['source']}' rejected and removed from RAG.",
        "document": updated,
        "index": store.stats(),
    }


@router.post("/documents/{doc_id}/archive")
def archive_document(doc_id: str, admin: str = Depends(require_admin)):
    """Archive an active document, removing its chunks from production vector search."""
    doc = repo.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    store.delete_source(doc["source"], repo.default_college_id())
    updated = repo.update_document_status(doc["id"], status="ARCHIVED", reviewed_by=admin)
    return {
        "message": f"Document '{doc['source']}' archived and unindexed.",
        "document": updated,
        "index": store.stats(),
    }



@router.post("/reindex")
def reindex(reset: bool = Form(default=False), _: str = Depends(require_admin)):
    results = ingest_all(reset=reset)
    return {"message": "Re-index complete", "files": results, "index": store.stats()}


class CrawlRequest(BaseModel):
    start_url: str = Field(default=f"https://{ALLOWED_CRAWL_DOMAIN}/")
    strategy: str = Field(default="astar")
    max_pages: int = Field(default=CRAWL_DEFAULT_MAX_PAGES, ge=1)
    max_depth: int = Field(default=CRAWL_DEFAULT_MAX_DEPTH, ge=0)


@router.post("/crawl")
def crawl_website(body: CrawlRequest, _: str = Depends(require_admin)):
    if body.strategy not in FRONTIERS:
        raise HTTPException(status_code=400, detail=f"strategy must be one of {sorted(FRONTIERS)}")

    netloc = urlparse(body.start_url).netloc.lower().removeprefix("www.")
    if netloc != ALLOWED_CRAWL_DOMAIN:
        raise HTTPException(
            status_code=400,
            detail=f"start_url must be on {ALLOWED_CRAWL_DOMAIN}",
        )

    # Endpoint-level clamp mirrors crawl()'s own clamp - keeps the reported
    # numbers honest even before the crawl runs.
    max_pages = min(body.max_pages, CRAWL_HARD_MAX_PAGES)
    max_depth = min(body.max_depth, CRAWL_HARD_MAX_DEPTH)

    summary = run_crawl(
        body.start_url,
        strategy=body.strategy,
        max_pages=max_pages,
        max_depth=max_depth,
        delay=CRAWL_DEFAULT_DELAY,
    )
    return {
        "message": (
            f"Crawl complete ({body.strategy}): {summary['fetched']} pages indexed, "
            f"{summary['skipped']} skipped, {summary['errors']} errors"
        ),
        "summary": summary,
        "index": store.stats(),
    }


@router.get("/crawl-runs")
def crawl_runs(_: str = Depends(require_admin)):
    return {"runs": list(reversed(load_runs()))}


class CrawlerSettingsRequest(BaseModel):
    scheduler_enabled: bool | None = None
    interval_hours: int | None = Field(default=None, ge=1, le=168)


@router.get("/crawler/settings")
def get_crawler_settings(_: str = Depends(require_admin)):
    settings = repo.get_crawler_settings()
    return {"settings": settings, "index": store.stats()}


@router.post("/crawler/settings")
def update_crawler_settings_route(body: CrawlerSettingsRequest, _: str = Depends(require_admin)):
    kwargs = {}
    if body.scheduler_enabled is not None:
        kwargs["scheduler_enabled"] = body.scheduler_enabled
    if body.interval_hours is not None:
        kwargs["interval_hours"] = body.interval_hours
    repo.update_crawler_settings(**kwargs)
    return {"message": "Crawler settings updated", "settings": repo.get_crawler_settings()}


@router.post("/crawler/run-now")
def trigger_crawl_now(_: str = Depends(require_admin)):
    from backend.crawler.scheduler import run_scheduled_crawl
    try:
        summary = run_scheduled_crawl()
        return {
            "message": f"Crawl completed: {summary.get('fetched', 0)} fetched, {summary.get('pages_changed', 0)} changed",
            "summary": summary,
            "index": store.stats(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Crawl failed: {exc}")


@router.post("/crawler/rollback")
def rollback_crawler(_: str = Depends(require_admin)):
    from backend.crawler.scheduler import rollback_dataset
    res = rollback_dataset()
    return {"message": f"Rollback complete: {res['restored_files']} files restored", "details": res}


@router.delete("/documents/{source}")
def delete_document(source: str, _: str = Depends(require_admin)):
    name = _safe_name(source)
    store.delete_source(name, repo.default_college_id())
    removed = remove_document(name)
    if not removed:
        raise HTTPException(status_code=404, detail="Document not in the registry")
    # Crawler-discovered files live one level deeper (raw/pdfs/crawled/,
    # raw/websie/crawled/) than admin-uploaded ones - without checking those
    # too, the file survives on disk and a later full reindex silently
    # re-discovers and re-indexes the very thing just "deleted".
    for folder in (RAW_PDF_DIR, RAW_PDF_CRAWL_DIR, RAW_WEB_DIR, RAW_WEB_CRAWL_DIR):
        path = folder / name
        if path.exists():
            path.unlink()
    return {"message": f"Removed {name} from the dataset", "index": store.stats()}


@router.get("/chat-logs")
def chat_logs(limit: int = 100, _: str = Depends(require_admin)):
    """Recent Q&A - useful for spotting gaps and for building a fine-tune set."""
    rows = repo.list_chat(limit=limit)
    unanswered = sum(1 for r in rows if not r.get("grounded"))
    return {
        "rows": list(reversed(rows)),
        "total": repo.count_chat(),
        "unanswered": unanswered,
    }


@router.get("/intents")
def intents(refresh: bool = False, _: str = Depends(require_admin)):
    """Logged questions grouped by intent (embedding similarity), most-asked
    first - lets the admin see recurring topics instead of a raw log."""
    clusters = compute_intents(force=refresh)
    return {
        "clusters": clusters,
        "total_questions": sum(c["count"] for c in clusters),
    }


@router.post("/password")
def update_password(
    new_password: str = Form(...),
    admin: str = Depends(require_admin),
):
    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be 8+ characters")
    change_password(admin, new_password)
    return {"message": "Password updated"}


# =============================================================================
# MINIMAL NIA ADMIN PANEL ISOLATED ENDPOINTS
# =============================================================================

@router.get("/me")
def admin_me(admin: str = Depends(require_admin)):
    """Verify admin authentication status."""
    return {"username": admin, "role": "admin"}


@router.get("/overview")
def admin_overview(_: str = Depends(require_admin)):
    """Dashboard metrics: total questions, unanswered count, indexed documents, system health."""
    college_id = repo.default_college_id()
    total_q = 0
    unanswered_q = 0
    with repo.get_conn() as conn:
        total_q = conn.execute(
            "SELECT COUNT(*) AS n FROM chat_history WHERE college_id = %s",
            (college_id,),
        ).fetchone()["n"]
        unanswered_q = conn.execute(
            "SELECT COUNT(*) AS n FROM chat_history WHERE college_id = %s AND grounded = false",
            (college_id,),
        ).fetchone()["n"]

    reviews = get_all_reviews()
    reviewed_ids = {cid for cid, r in reviews.items() if r.get("reviewed")}

    active_docs = repo.list_documents(status_filter="ACTIVE")
    all_docs = repo.list_documents()
    pdf_docs = [
        d for d in all_docs
        if d.get("doc_type") == "pdf" or str(d.get("source", "")).lower().endswith(".pdf")
    ]

    idx_stats = store.stats()

    sys_status = "Healthy"
    if not Path(LLM_PATH).exists():
        sys_status = "Degraded"

    return {
        "total_questions": total_q,
        "unanswered_questions": unanswered_q,
        "unreviewed_count": max(0, unanswered_q - len(reviewed_ids)),
        "indexed_documents": len(active_docs),
        "pdf_sources_count": len(pdf_docs),
        "searchable_chunks": idx_stats.get("chunks", 0),
        "system_status": sys_status,
        "collection_name": idx_stats.get("collection", COLLECTION_NAME),
    }


@router.get("/system-health")
def system_health_details(_: str = Depends(require_admin)):
    """Safe, non-blocking component health checks without leaking secrets or credentials."""
    import time

    # 1. API Service
    api_health = {
        "status": "Healthy",
        "service": "SXCA Admission Assistant Backend (Niaa)",
        "version": "1.0.0",
        "college": COLLEGE_NAME,
    }

    # 2. Database Connectivity
    db_start = time.perf_counter()
    db_status = "Healthy"
    latency_ms = None
    try:
        with repo.get_conn() as conn:
            conn.execute("SELECT 1").fetchone()
            latency_ms = round((time.perf_counter() - db_start) * 1000, 2)
    except Exception:
        db_status = "Unavailable"

    db_health = {
        "status": db_status,
        "engine": "PostgreSQL 17",
        "host": DB_HOST,
        "port": DB_PORT,
        "database": DB_NAME,
        "latency_ms": latency_ms,
    }

    # 3. Vector DB
    vec_status = "Healthy"
    chunk_count = 0
    try:
        s = store.stats()
        chunk_count = s.get("chunks", 0)
    except Exception:
        vec_status = "Unavailable"

    vec_health = {
        "status": vec_status,
        "engine": "ChromaDB (Persistent)",
        "collection": COLLECTION_NAME,
        "chunks": chunk_count,
    }

    # 4. Local Model Status
    model_path = Path(LLM_PATH)
    path_exists = model_path.exists()
    is_cached = False
    try:
        from backend.llm import qwen
        is_cached = qwen._load.cache_info().currsize > 0
    except Exception:
        pass

    if path_exists:
        model_status = "Healthy" if is_cached else "Degraded"
        model_note = "Model loaded in active memory" if is_cached else "Model present on disk, loaded on demand"
    else:
        model_status = "Unavailable"
        model_note = "Model weights directory not found on disk"

    model_health = {
        "status": model_status,
        "model_name": model_path.name,
        "device": LLM_DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "cached_in_memory": is_cached,
        "note": model_note,
    }

    overall = "Healthy"
    if any(c["status"] == "Unavailable" for c in [db_health, vec_health, model_health]):
        overall = "Degraded" if db_health["status"] == "Healthy" else "Unavailable"
    elif any(c["status"] == "Degraded" for c in [db_health, vec_health, model_health]):
        overall = "Degraded"

    return {
        "status": overall,
        "api": api_health,
        "database": db_health,
        "vector_db": vec_health,
        "model": model_health,
    }


@router.get("/pdf-knowledge")
def list_pdf_knowledge(_: str = Depends(require_admin)):
    """List all official SXCA PDF sources with chunk counts, file sizes, and indexing status."""
    all_docs = repo.list_documents()
    pdf_docs_by_source = {
        d["source"]: d for d in all_docs
        if d.get("doc_type") == "pdf" or str(d.get("source", "")).lower().endswith(".pdf")
    }

    found_sources = set(pdf_docs_by_source.keys())
    for folder in (RAW_PDF_DIR, RAW_PDF_CRAWL_DIR):
        if folder.exists():
            for p in folder.rglob("*.pdf"):
                if "_versions" not in p.parts:
                    found_sources.add(p.name)

    results = []
    for src in sorted(found_sources):
        doc = pdf_docs_by_source.get(src)
        target = _find_target_file(src)
        file_size = target.stat().st_size if (target and target.exists()) else None

        status = doc.get("status", "ACTIVE") if doc else "ACTIVE"
        chunks = doc.get("chunks", 0) if doc else 0
        added_at = doc.get("added_at") if doc else None
        updated_at = doc.get("updated_at") if doc else None
        uploaded_by = doc.get("uploaded_by") if doc else "system"

        results.append({
            "source": src,
            "filename": src,
            "doc_type": "pdf",
            "chunks": chunks,
            "status": status,
            "file_size_bytes": file_size,
            "exists_on_disk": target is not None and target.exists(),
            "added_at": added_at.isoformat() if hasattr(added_at, "isoformat") else str(added_at or ""),
            "updated_at": updated_at.isoformat() if hasattr(updated_at, "isoformat") else str(updated_at or ""),
            "uploaded_by": uploaded_by,
        })
    return {"pdfs": results, "index": store.stats()}


@router.post("/pdf/upload")
async def upload_pdf_knowledge(
    file: UploadFile = File(...),
    admin: str = Depends(require_admin),
):
    """Admin PDF Upload: validates file type and magic bytes, extracts text/tables, computes
    embeddings, and indexes directly into ChromaDB. Reports success only after indexing finishes."""
    name = _safe_name(file.filename or "")
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files (.pdf) are accepted")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 60 MB")

    if not payload.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid PDF document (magic bytes mismatch)",
        )

    target = RAW_PDF_DIR / name
    replaced = target.exists()
    if replaced:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup_dir = RAW_PDF_DIR / "_versions"
        backup_dir.mkdir(exist_ok=True)
        shutil.copy2(target, backup_dir / f"{target.stem}.{stamp}{target.suffix}")

    target.write_bytes(payload)

    try:
        result = ingest_file(target, uploaded_by=admin, replace=True, status="ACTIVE")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse and index PDF: {exc}")

    return {
        "success": True,
        "message": f"Successfully ingested and indexed {name} ({result['chunks']} chunks created).",
        "replaced": replaced,
        "source": name,
        "filename": name,
        "chunks": result["chunks"],
        "status": "ACTIVE",
        "index": store.stats(),
    }


@router.delete("/pdf/{source}")
def delete_pdf_knowledge(source: str, _: str = Depends(require_admin)):
    """Safely delete a PDF: removes its chunks and embeddings from ChromaDB without touching
    any other documents, cleans up database metadata, and unlinks file from disk."""
    name = _safe_name(source)
    college_id = repo.default_college_id()

    # 1. Safely remove ONLY this document's chunks from ChromaDB
    store.delete_source(name, college_id)

    # 2. Clean database records & registry
    repo.delete_document(name)
    remove_document(name)

    # 3. Clean files from disk
    unlinked = False
    for folder in (RAW_PDF_DIR, RAW_PDF_CRAWL_DIR):
        path = folder / name
        if path.exists():
            path.unlink()
            unlinked = True

    return {
        "success": True,
        "message": f"Safely removed {name} from knowledge base and cleared its chunks from vector search.",
        "source": name,
        "unlinked_from_disk": unlinked,
        "index": store.stats(),
    }


@router.get("/unanswered-questions")
def list_unanswered_questions(limit: int = 100, _: str = Depends(require_admin)):
    """List student questions where the chatbot fell back or could not find verified evidence,
    joined with admin review status."""
    college_id = repo.default_college_id()
    reviews = get_all_reviews()

    with repo.get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, question, answer, grounded, sources, created_at
            FROM chat_history
            WHERE college_id = %s AND grounded = false
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (college_id, limit),
        ).fetchall()

    items = []
    for r in rows:
        cid = str(r["id"])
        rev = reviews.get(cid, {})
        items.append({
            "id": r["id"],
            "question": r["question"],
            "answer": r["answer"],
            "grounded": r["grounded"],
            "sources": r["sources"] if isinstance(r["sources"], list) else [],
            "created_at": r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else str(r["created_at"]),
            "reviewed": rev.get("reviewed", False),
            "reviewed_by": rev.get("reviewed_by"),
            "reviewed_at": rev.get("reviewed_at"),
            "notes": rev.get("notes", ""),
        })

    return {
        "unanswered_questions": items,
        "total_unanswered": len(items),
    }


class ReviewRequest(BaseModel):
    reviewed: bool = True
    notes: str = ""


@router.post("/unanswered-questions/{chat_id}/review")
def mark_unanswered_question_reviewed(
    chat_id: int,
    body: ReviewRequest = ReviewRequest(),
    admin: str = Depends(require_admin),
):
    """Mark an unanswered student question as reviewed without altering the database schema."""
    entry = set_review_status(chat_id, admin_user=admin, reviewed=body.reviewed, notes=body.notes)
    status_label = "reviewed" if body.reviewed else "pending review"
    return {
        "success": True,
        "message": f"Question #{chat_id} marked as {status_label}.",
        "review": entry,
    }

