"""Initialize and seed Neon PostgreSQL database for SXCA Admission Assistant.

Usage:
  1. Set DATABASE_URL=postgresql://... in your local .env file
  2. Run:
     .\\venv\\Scripts\\python.exe scripts/init_neon_db.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")


def main() -> None:
    from backend.api.auth import ensure_default_admin
    from backend.db import repo
    from backend.db.init_db import apply_schema, ensure_database
    from backend.db.repo import default_college_id, ensure_default_college
    from backend.db.seed_eligibility import seed
    from backend.vectorstore import store

    print("1. Connecting to PostgreSQL and applying schema...")
    ensure_database()
    apply_schema()
    ensure_default_college()
    ensure_default_admin()

    print("2. Seeding SXCA course eligibility & fee structure rows...")
    n = seed()
    print(f"   Seeded {n} course_eligibility rows.")

    print("3. Syncing indexed ChromaDB document metadata to PostgreSQL...")
    cid = default_college_id()
    total_docs = 0
    for dt in ("pdf", "website"):
        chunks = store.get_all_chunks(doc_type=dt, college_id=cid)
        counts: dict[str, int] = {}
        for c in chunks:
            src = (c.get("metadata") or {}).get("source")
            if src:
                counts[src] = counts.get(src, 0) + 1
        for src, cnt in counts.items():
            repo.upsert_document(source=src, doc_type=dt, chunks=cnt, uploaded_by="system", status="ACTIVE")
            total_docs += 1
    print(f"   Synced {total_docs} active document records.")
    print("Done! Neon PostgreSQL is ready for production.")


if __name__ == "__main__":
    main()
