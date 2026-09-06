"""Crawl run history - backed by Postgres (backend/db/repo.py)."""

from backend.db import repo

MAX_RUNS = 20


def load_runs() -> list[dict]:
    return repo.list_crawl_runs(limit=MAX_RUNS)


def record_run(run: dict) -> None:
    repo.insert_crawl_run(run)
