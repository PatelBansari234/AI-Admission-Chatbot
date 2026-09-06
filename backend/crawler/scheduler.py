"""Automated website freshness scheduler and version manager."""
import os
import shutil
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.config import (
    ALLOWED_CRAWL_DOMAIN,
    CRAWL_DEFAULT_DELAY,
    CRAWL_DEFAULT_MAX_DEPTH,
    CRAWL_DEFAULT_MAX_PAGES,
    RAW_PDF_CRAWL_DIR,
    RAW_WEB_CRAWL_DIR,
)
from backend.crawler.crawler import crawl
from backend.db import repo
from backend.rag.ingest import ingest_file
from backend.vectorstore import store

_scheduler_thread: threading.Thread | None = None
_scheduler_stop = threading.Event()
_crawl_lock = threading.Lock()


def _scheduler_loop():
    while not _scheduler_stop.is_set():
        try:
            settings = repo.get_crawler_settings()
            if settings.get("scheduler_enabled"):
                now = datetime.now(timezone.utc)
                next_run = settings.get("next_run_at")
                if not next_run:
                    interval = settings.get("interval_hours", 24)
                    next_run = now + timedelta(hours=interval)
                    repo.update_crawler_settings(next_run_at=next_run)

                if now >= next_run:
                    run_scheduled_crawl()
        except Exception as exc:
            repo.update_crawler_settings(last_status="error", last_error=str(exc))

        # Check every 30 seconds
        _scheduler_stop.wait(30)


def start_scheduler():
    global _scheduler_thread
    if _scheduler_thread is None or not _scheduler_thread.is_alive():
        _scheduler_stop.clear()
        _scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True, name="WebsiteFreshnessScheduler")
        _scheduler_thread.start()


def stop_scheduler():
    _scheduler_stop.set()


def run_scheduled_crawl(start_url: str | None = None, max_pages: int | None = None, max_depth: int | None = None) -> dict:
    if not _crawl_lock.acquire(blocking=False):
        return {"status": "skipped", "message": "A crawl is already in progress"}

    try:
        settings = repo.get_crawler_settings()
        url = start_url or f"https://{ALLOWED_CRAWL_DOMAIN}/"
        pages = max_pages or CRAWL_DEFAULT_MAX_PAGES
        depth = max_depth or CRAWL_DEFAULT_MAX_DEPTH
        interval = settings.get("interval_hours", 24)

        repo.update_crawler_settings(last_status="running", last_error=None)
        summary = crawl(url, strategy="astar", max_pages=pages, max_depth=depth, delay=CRAWL_DEFAULT_DELAY, index=True)

        now = datetime.now(timezone.utc)
        next_run = now + timedelta(hours=interval)
        repo.update_crawler_settings(
            last_run_at=now,
            next_run_at=next_run,
            last_status="completed" if summary.get("errors", 0) == 0 else "completed_with_errors",
        )
        return summary
    except Exception as exc:
        repo.update_crawler_settings(last_status="failed", last_error=str(exc))
        raise
    finally:
        _crawl_lock.release()


def rollback_dataset() -> dict:
    """Restores most recently backed-up page versions from _versions/ for recovery."""
    restored_count = 0
    for base_dir in (RAW_WEB_CRAWL_DIR, RAW_PDF_CRAWL_DIR):
        backup_dir = base_dir / "_versions"
        if not backup_dir.exists():
            continue
        # Find latest backup for each file stem
        backups: dict[str, Path] = {}
        for f in sorted(backup_dir.iterdir(), key=lambda p: p.stat().st_mtime):
            if f.is_file():
                stem = f.name.split(".")[0]
                backups[stem] = f

        for stem, backup_file in backups.items():
            # Target original file name
            target_name = f"{stem}{backup_file.suffix}"
            target_path = base_dir / target_name
            shutil.copy2(backup_file, target_path)
            try:
                ingest_file(target_path, replace=True)
                restored_count += 1
            except Exception:
                pass

    return {"status": "ok", "restored_files": restored_count, "index": store.stats()}
