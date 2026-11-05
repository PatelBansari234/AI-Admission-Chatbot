"""Lightweight, non-disruptive review tracking for unanswered/low-evidence questions.

Stores admin review state in data/admin_reviews.json to strictly preserve the
existing database schema while satisfying audit review workflows.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from backend.config import DATA_DIR

REVIEWS_FILE = DATA_DIR / "admin_reviews.json"
_LOCK = threading.Lock()


def _ensure_store() -> dict:
    if not REVIEWS_FILE.exists():
        initial = {"reviews": {}}
        REVIEWS_FILE.parent.mkdir(parents=True, exist_ok=True)
        REVIEWS_FILE.write_text(json.dumps(initial, indent=2), encoding="utf-8")
        return initial
    try:
        data = json.loads(REVIEWS_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "reviews" not in data:
            return {"reviews": {}}
        return data
    except Exception:
        return {"reviews": {}}


def get_all_reviews() -> dict[str, dict]:
    with _LOCK:
        local_reviews = dict(_ensure_store().get("reviews", {}))
    try:
        from backend.db import repo
        db_reviews = repo.get_all_admin_reviews()
        if db_reviews:
            local_reviews.update(db_reviews)
    except Exception:
        pass
    return local_reviews


def get_review(chat_id: int | str) -> dict | None:
    reviews = get_all_reviews()
    return reviews.get(str(chat_id))


def set_review_status(chat_id: int | str, admin_user: str, reviewed: bool = True, notes: str = "") -> dict:
    with _LOCK:
        store_data = _ensure_store()
        cid = str(chat_id)
        entry = {
            "reviewed": bool(reviewed),
            "reviewed_by": admin_user,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
        }
        store_data.setdefault("reviews", {})[cid] = entry
        REVIEWS_FILE.write_text(json.dumps(store_data, indent=2), encoding="utf-8")
    try:
        from backend.db import repo
        repo.upsert_admin_review(int(chat_id), bool(reviewed), admin_user, notes)
    except Exception:
        pass
    return entry
