"""Admin routes: upload/replace prospectus PDFs and manage the indexed dataset."""

import re
import shutil
from datetime import datetime, timezone
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.api.auth import change_password, require_admin
from backend.config import (
    ALLOWED_CRAWL_DOMAIN,
    CRAWL_DEFAULT_DELAY,
    CRAWL_DEFAULT_MAX_DEPTH,
    CRAWL_DEFAULT_MAX_PAGES,
    CRAWL_HARD_MAX_DEPTH,
    CRAWL_HARD_MAX_PAGES,
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
def list_documents(_: str = Depends(require_admin)):
    return {"documents": load_registry(), "index": store.stats()}


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    admin: str = Depends(require_admin),
):
    """Save a new/updated PDF into the dataset and re-index it immediately."""
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

    # Keep the previous version instead of silently destroying it.
    replaced = target.exists()
    if replaced:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup_dir = target_dir / "_versions"
        backup_dir.mkdir(exist_ok=True)
        shutil.copy2(target, backup_dir / f"{target.stem}.{stamp}{target.suffix}")

    target.write_bytes(payload)

    try:
        result = ingest_file(target, uploaded_by=admin, replace=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not index file: {exc}")

    return {
        "message": f"{'Replaced' if replaced else 'Added'} {name} and re-indexed it",
        "replaced": replaced,
        **result,
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
