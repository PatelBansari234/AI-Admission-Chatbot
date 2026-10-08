"""Data-access layer over the tables in schema.sql.

Kept separate from auth.py / registry.py / crawl_log.py / analytics/intents.py
so those modules keep their existing public function signatures (unchanged
call sites in admin.py, pipeline.py) while swapping their storage backend
from JSON files to Postgres underneath.

This deployment serves one college today. Every content function below still
scopes its rows to a college_id - resolved via default_college_id() rather
than threaded through every caller - so the data model is ready for more
than one college without forcing registry.py/crawl_log.py/admin.py/
pipeline.py to all learn about college selection before there's an actual
second college to select.
"""

import threading

from psycopg.types.json import Jsonb

from backend.config import COLLEGE_CITY, COLLEGE_NAME, COLLEGE_STATE, COLLEGE_WEBSITE
from backend.db.connection import get_conn

# --- colleges ----------------------------------------------------------------

_default_college_lock = threading.Lock()
_default_college_id: int | None = None


def ensure_default_college() -> int:
    """Get-or-create this deployment's one college row, return its id."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM colleges WHERE name = %s AND city = %s",
            (COLLEGE_NAME, COLLEGE_CITY),
        ).fetchone()
        if row:
            return row["id"]
        row = conn.execute(
            """
            INSERT INTO colleges (name, city, state, website)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (COLLEGE_NAME, COLLEGE_CITY, COLLEGE_STATE, COLLEGE_WEBSITE),
        ).fetchone()
        conn.commit()
        return row["id"]


def default_college_id() -> int:
    """Cached after first lookup - every insert/query below calls this
    rather than hitting the colleges table on every single request."""
    global _default_college_id
    if _default_college_id is None:
        with _default_college_lock:
            if _default_college_id is None:
                _default_college_id = ensure_default_college()
    return _default_college_id


# --- users -----------------------------------------------------------------


def get_user(username: str) -> dict | None:
    with get_conn() as conn:
        return conn.execute(
            "SELECT username, password_hash, role FROM users WHERE username = %s",
            (username,),
        ).fetchone()


def create_user(username: str, password_hash: str, role: str = "admin") -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s) "
            "ON CONFLICT (username) DO NOTHING",
            (username, password_hash, role),
        )
        conn.commit()


def count_users() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]


def update_password(username: str, password_hash: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE users SET password_hash = %s WHERE username = %s",
            (password_hash, username),
        )
        conn.commit()
        return cur.rowcount > 0


# --- documents ---------------------------------------------------------------


def upsert_document(source: str, doc_type: str, chunks: int, uploaded_by: str) -> None:
    college_id = default_college_id()
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO documents (college_id, source, doc_type, chunks, uploaded_by)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (college_id, source) DO UPDATE
                SET doc_type = EXCLUDED.doc_type,
                    chunks = EXCLUDED.chunks,
                    uploaded_by = EXCLUDED.uploaded_by,
                    updated_at = now()
            RETURNING id
            """,
            (college_id, source, doc_type, chunks, uploaded_by),
        ).fetchone()
        conn.execute(
            "INSERT INTO document_history (document_id, uploaded_by, chunks) "
            "VALUES (%s, %s, %s)",
            (row["id"], uploaded_by, chunks),
        )
        conn.commit()


def list_documents() -> list[dict]:
    with get_conn() as conn:
        docs = conn.execute(
            "SELECT id, source, doc_type, chunks, uploaded_by, added_at, updated_at "
            "FROM documents WHERE college_id = %s ORDER BY updated_at DESC",
            (default_college_id(),),
        ).fetchall()
        result = []
        for doc in docs:
            history = conn.execute(
                "SELECT at, uploaded_by AS by, chunks FROM document_history "
                "WHERE document_id = %s ORDER BY at ASC LIMIT 20",
                (doc["id"],),
            ).fetchall()
            entry = {k: v for k, v in doc.items() if k != "id"}
            entry["history"] = history
            result.append(entry)
        return result


def document_freshness(doc_type: str | None = None) -> dict[str, object]:
    """{source: updated_at}, cheapest possible shape for retriever.py's
    source-priority freshness boost - unlike list_documents() this skips
    the per-document upload-history query, since ranking chunks on every
    single retrieve() call doesn't need it. doc_type matters here: PDFs and
    website pages shouldn't be ranked against each other in the same
    freshness ordering (website already gets its own flat boost - see
    source_priority.py), so the caller filters to one type at a time."""
    with get_conn() as conn:
        if doc_type:
            rows = conn.execute(
                "SELECT source, updated_at FROM documents WHERE college_id = %s AND doc_type = %s",
                (default_college_id(), doc_type),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT source, updated_at FROM documents WHERE college_id = %s",
                (default_college_id(),),
            ).fetchall()
    return {row["source"]: row["updated_at"] for row in rows}


def delete_document(source: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM documents WHERE college_id = %s AND source = %s",
            (default_college_id(), source),
        )
        conn.commit()
        return cur.rowcount > 0


# --- conversations -----------------------------------------------------------


def verify_conversation_owner(conversation_id: str, owner_id: str | None) -> bool:
    """Checks if conversation exists and belongs to owner_id.
    Returns True if conversation doesn't exist yet, or belongs to owner_id, or has NULL owner.
    Returns False if conversation belongs to a different owner.
    """
    if not conversation_id or not owner_id:
        return True
    with get_conn() as conn:
        row = conn.execute(
            "SELECT owner_id FROM conversations WHERE id = %s",
            (conversation_id,),
        ).fetchone()
        if not row:
            return True  # New conversation, allowed to create
        db_owner = row["owner_id"]
        if db_owner is None:
            return True  # Legacy unowned conversation
        return db_owner == owner_id


def get_or_create_conversation(conversation_id: str, owner_id: str | None = None) -> None:
    """id comes from the frontend (crypto.randomUUID(), one per page load) -
    this just makes sure a row exists for it and bumps last_activity, so a
    conversation never needs an explicit "start" call before its first
    message."""
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO conversations (id, college_id, owner_id)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                owner_id = COALESCE(conversations.owner_id, EXCLUDED.owner_id),
                last_activity = now()
            """,
            (conversation_id, default_college_id(), owner_id),
        )
        conn.commit()


_IN_MEMORY_CONTEXTS: dict[str, dict] = {}


def get_conversation_context(conversation_id: str | None, owner_id: str | None = None) -> dict | None:
    if not conversation_id:
        return _IN_MEMORY_CONTEXTS.get("default")
    if owner_id and not verify_conversation_owner(conversation_id, owner_id):
        return None
    with get_conn() as conn:
        row = conn.execute(
            "SELECT course, qualification, summary, profile FROM conversation_memory "
            "WHERE conversation_id = %s",
            (conversation_id,),
        ).fetchone()
        if row:
            d = dict(row)
            if not d.get("profile") or not isinstance(d["profile"], dict):
                d["profile"] = {}
            return d
        return _IN_MEMORY_CONTEXTS.get(conversation_id)


def update_conversation_context(
    conversation_id: str | None,
    course: str | None = None,
    qualification: str | None = None,
    summary: str | None = None,
    profile: dict | None = None,
    owner_id: str | None = None,
) -> None:
    """Upsert, and only overwrite a field when a non-None value is given -
    so e.g. a turn that mentions a qualification but not a course doesn't
    blank out the course learned two turns ago."""
    if conversation_id and owner_id and not verify_conversation_owner(conversation_id, owner_id):
        return
    cid_key = conversation_id or "default"
    current = _IN_MEMORY_CONTEXTS.setdefault(cid_key, {"course": None, "qualification": None, "summary": None, "profile": {}})
    if course is not None:
        current["course"] = course
    if qualification is not None:
        current["qualification"] = qualification
    if summary is not None:
        current["summary"] = summary
    if profile is not None:
        current_prof = current.setdefault("profile", {})
        current_prof.update(profile)

    if not conversation_id:
        return

    profile_json = Jsonb(current["profile"]) if current.get("profile") else Jsonb({})
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO conversation_memory (conversation_id, course, qualification, summary, profile)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (conversation_id) DO UPDATE SET
                course = COALESCE(EXCLUDED.course, conversation_memory.course),
                qualification = COALESCE(EXCLUDED.qualification, conversation_memory.qualification),
                summary = COALESCE(EXCLUDED.summary, conversation_memory.summary),
                profile = CASE WHEN EXCLUDED.profile IS NOT NULL THEN EXCLUDED.profile ELSE conversation_memory.profile END,
                updated_at = now()
            """,
            (conversation_id, course, qualification, summary, profile_json),
        )
        conn.commit()


def get_recent_turns(conversation_id: str, limit_turns: int = 4, owner_id: str | None = None) -> list[dict]:
    """Most recent `limit_turns` Q&A pairs, oldest first, expanded into
    alternating user/assistant messages - chat_history already stores one
    row per turn, so there's no separate messages table to keep in sync."""
    if owner_id and not verify_conversation_owner(conversation_id, owner_id):
        return []
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT question, answer FROM chat_history "
            "WHERE conversation_id = %s ORDER BY created_at DESC LIMIT %s",
            (conversation_id, limit_turns),
        ).fetchall()
    turns = []
    for row in reversed(rows):
        turns.append({"role": "user", "content": row["question"]})
        turns.append({"role": "assistant", "content": row["answer"]})
    return turns


def get_recent_chat_sources(conversation_id: str, owner_id: str | None = None) -> list[dict]:
    if not conversation_id or (owner_id and not verify_conversation_owner(conversation_id, owner_id)):
        return []
    with get_conn() as conn:
        row = conn.execute(
            "SELECT sources FROM chat_history WHERE conversation_id = %s AND sources IS NOT NULL ORDER BY created_at DESC LIMIT 1",
            (conversation_id,),
        ).fetchone()
        if row and row.get("sources"):
            srcs = row["sources"]
            return srcs if isinstance(srcs, list) else []
    return []


# --- chat history ------------------------------------------------------------


def insert_chat(
    question: str, answer: str, grounded: bool, sources: list[dict],
    conversation_id: str | None = None,
    owner_id: str | None = None,
) -> None:
    if conversation_id and owner_id and not verify_conversation_owner(conversation_id, owner_id):
        return
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_history (college_id, conversation_id, question, answer, grounded, sources) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (default_college_id(), conversation_id, question, answer, grounded, Jsonb(sources)),
        )
        conn.commit()



def list_chat(limit: int | None = None) -> list[dict]:
    """Oldest first. With a limit, returns the most recent `limit` rows
    (still oldest-first) rather than an arbitrary head of the whole table."""
    college_id = default_college_id()
    with get_conn() as conn:
        if limit:
            rows = conn.execute(
                "SELECT question, answer, grounded, sources, created_at "
                "FROM chat_history WHERE college_id = %s "
                "ORDER BY created_at DESC LIMIT %s",
                (college_id, limit),
            ).fetchall()
            return list(reversed(rows))
        return conn.execute(
            "SELECT question, answer, grounded, sources, created_at "
            "FROM chat_history WHERE college_id = %s ORDER BY created_at ASC",
            (college_id,),
        ).fetchall()


def count_chat() -> int:
    with get_conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) AS n FROM chat_history WHERE college_id = %s",
            (default_college_id(),),
        ).fetchone()["n"]


# --- crawl runs ------------------------------------------------------------


def insert_crawl_run(run: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO crawl_runs
                (college_id, strategy, start_url, max_pages, max_depth, fetched, skipped, errors, pages)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                default_college_id(), run["strategy"], run["start_url"], run["max_pages"],
                run["max_depth"], run["fetched"], run["skipped"], run["errors"], Jsonb(run["pages"]),
            ),
        )
        conn.commit()


def list_crawl_runs(limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT strategy, start_url, max_pages, max_depth, fetched, skipped, errors,
                   pages, created_at
            FROM crawl_runs WHERE college_id = %s ORDER BY created_at DESC LIMIT %s
            """,
            (default_college_id(), limit),
        ).fetchall()


# --- course eligibility ------------------------------------------------------


def upsert_course_eligibility(row: dict) -> None:
    """row keys match course_eligibility's columns (see schema.sql) except
    college_id/id. Upserts on (college_id, course_name, category,
    academic_year) so re-running a seed/extraction pass updates in place
    rather than duplicating."""
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO course_eligibility (
                college_id, course_name, level, category, eligibility,
                required_subjects, minimum_percentage, fee_amount,
                fee_amount_female, fee_note, contact_email, academic_year,
                source_document
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (college_id, course_name, category, academic_year) DO UPDATE SET
                level = EXCLUDED.level,
                eligibility = EXCLUDED.eligibility,
                required_subjects = EXCLUDED.required_subjects,
                minimum_percentage = EXCLUDED.minimum_percentage,
                fee_amount = EXCLUDED.fee_amount,
                fee_amount_female = EXCLUDED.fee_amount_female,
                fee_note = EXCLUDED.fee_note,
                contact_email = EXCLUDED.contact_email,
                source_document = EXCLUDED.source_document,
                last_verified = now()
            """,
            (
                default_college_id(), row["course_name"], row["level"], row["category"],
                row["eligibility"], row.get("required_subjects"), row.get("minimum_percentage"),
                row.get("fee_amount"), row.get("fee_amount_female"), row.get("fee_note"),
                row.get("contact_email"), row["academic_year"], row["source_document"],
            ),
        )
        conn.commit()


def list_course_eligibility() -> list[dict]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM course_eligibility WHERE college_id = %s ORDER BY course_name, category",
            (default_college_id(),),
        ).fetchall()


def find_courses_by_eligibility(stream_or_qualification: str) -> list[dict]:
    """Case-insensitive substring match against the verbatim eligibility
    text (e.g. "Science" matches "12th Science", "12th Science & Commerce",
    "BSc - Maths, Stats, Phy, Chem, BA Eco, BE & BCA" does NOT match
    "Science" as a bare word unless it's actually in there) - deliberately
    simple rather than a semantic match, since eligibility.py's caller is
    responsible for deciding what counts as a match, not this query."""
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM course_eligibility WHERE college_id = %s "
            "AND eligibility ILIKE %s ORDER BY course_name, category",
            (default_college_id(), f"%{stream_or_qualification}%"),
        ).fetchall()


def find_course_eligibility(course_name: str) -> list[dict]:
    tokens = [t for t in course_name.split() if t]
    if not tokens:
        return []
    pattern = "%" + "%".join(tokens) + "%"
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM course_eligibility WHERE college_id = %s "
            "AND course_name ILIKE %s ORDER BY category",
            (default_college_id(), pattern),
        ).fetchall()


# --- crawler settings & page versions -----------------------------------------


def get_crawler_settings() -> dict:
    cid = default_college_id()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM crawler_settings WHERE college_id = %s", (cid,)
        ).fetchone()
        if not row:
            conn.execute(
                """
                INSERT INTO crawler_settings (college_id, scheduler_enabled, interval_hours, dataset_version)
                VALUES (%s, true, 24, 'v1.0')
                ON CONFLICT (college_id) DO NOTHING
                """,
                (cid,),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM crawler_settings WHERE college_id = %s", (cid,)
            ).fetchone()
        return dict(row) if row else {}


def update_crawler_settings(**kwargs) -> None:
    cid = default_college_id()
    get_crawler_settings()  # ensure row exists
    set_clauses = []
    values = []
    for k, v in kwargs.items():
        if v is not None:
            set_clauses.append(f"{k} = %s")
            values.append(v)
    if not set_clauses:
        return
    set_clauses.append("updated_at = now()")
    values.append(cid)
    query = f"UPDATE crawler_settings SET {', '.join(set_clauses)} WHERE college_id = %s"
    with get_conn() as conn:
        conn.execute(query, tuple(values))
        conn.commit()


def get_page_hash(url: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM page_hashes WHERE college_id = %s AND url = %s",
            (default_college_id(), url),
        ).fetchone()
        return dict(row) if row else None


def upsert_page_hash(url: str, content_hash: str, saved_as: str, doc_type: str = "website", chunks: int = 0) -> int:
    cid = default_college_id()
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO page_hashes (college_id, url, content_hash, saved_as, doc_type, chunks, version)
            VALUES (%s, %s, %s, %s, %s, %s, 1)
            ON CONFLICT (college_id, url) DO UPDATE SET
                content_hash = EXCLUDED.content_hash,
                saved_as = EXCLUDED.saved_as,
                chunks = EXCLUDED.chunks,
                version = page_hashes.version + 1,
                updated_at = now()
            RETURNING version
            """,
            (cid, url, content_hash, saved_as, doc_type, chunks),
        ).fetchone()
        conn.commit()
        return row["version"]


def list_page_hashes() -> list[dict]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM page_hashes WHERE college_id = %s ORDER BY updated_at DESC",
            (default_college_id(),),
        ).fetchall()


def get_url_for_source(saved_as: str) -> str | None:
    if not saved_as:
        return None
    with get_conn() as conn:
        row = conn.execute(
            "SELECT url FROM page_hashes WHERE college_id = %s AND saved_as = %s LIMIT 1",
            (default_college_id(), saved_as),
        ).fetchone()
        if row and row.get("url"):
            return row["url"]

    from backend.config import ALLOWED_CRAWL_DOMAIN
    if saved_as.endswith(".pdf"):
        return f"https://{ALLOWED_CRAWL_DOMAIN}/wp-content/uploads/{saved_as}"
    if saved_as.endswith(".html"):
        slug = saved_as.removesuffix(".html")
        if slug == "index":
            return f"https://{ALLOWED_CRAWL_DOMAIN}/"
        path = slug.replace("-", "/")
        return f"https://{ALLOWED_CRAWL_DOMAIN}/{path}/"
    return None


def create_dataset_version(version_tag: str, description: str, files_count: int, chunks_count: int) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO dataset_versions (college_id, version_tag, description, files_count, chunks_count)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (default_college_id(), version_tag, description, files_count, chunks_count),
        )
        conn.commit()


def list_dataset_versions() -> list[dict]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM dataset_versions WHERE college_id = %s ORDER BY created_at DESC",
            (default_college_id(),),
        ).fetchall()

