"""FastAPI entrypoint: student chat + admin dataset management + static UI."""

import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.api.admin import router as admin_router
from backend.api.auth import create_token, ensure_default_admin, resolve_owner_id, verify_password
from backend.config import COLLEGE_NAME, LLM_PROVIDER, PROJECT_ROOT, RAW_PDF_DIR, RAW_WEB_DIR
from backend.crawler.scheduler import start_scheduler, stop_scheduler
from backend.db import repo
from backend.db.repo import ensure_default_college
from backend.rag import pipeline
from backend.vectorstore import store

# Built by `npm run build` in frontend/ (Vite). Not present until that's run.
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


def _sync_persistence_and_index() -> None:
    """Restore admin-uploaded/deleted documents from PostgreSQL and auto-index if ChromaDB is empty."""
    try:
        blobs = repo.list_document_blobs()
        cid = repo.default_college_id()
        for b in blobs:
            src = b.get("source")
            if not src:
                continue
            target_dir = RAW_PDF_DIR if str(src).lower().endswith(".pdf") or b.get("doc_type") == "pdf" else RAW_WEB_DIR
            target_path = target_dir / src
            if b.get("is_deleted"):
                if target_path.exists():
                    try:
                        target_path.unlink()
                    except Exception:
                        pass
                store.delete_source(src, cid)
            elif b.get("content_bytes"):
                raw_bytes = bytes(b["content_bytes"])
                need_ingest = not target_path.exists()
                if need_ingest:
                    target_path.write_bytes(raw_bytes)
                    try:
                        from backend.rag.ingest import ingest_file
                        ingest_file(target_path, uploaded_by=b.get("uploaded_by") or "admin", replace=True, status="ACTIVE")
                    except Exception:
                        pass
    except Exception:
        pass

    try:
        if store.stats().get("chunks", 0) == 0:
            from backend.rag.ingest import ingest_all
            ingest_all(reset=False)
        elif not repo.list_documents():
            cid = repo.default_college_id()
            for dt in ("pdf", "website"):
                chunks = store.get_all_chunks(doc_type=dt, college_id=cid)
                counts: dict[str, int] = {}
                for c in chunks:
                    src = (c.get("metadata") or {}).get("source")
                    if src:
                        counts[src] = counts.get(src, 0) + 1
                for src, cnt in counts.items():
                    repo.upsert_document(source=src, doc_type=dt, chunks=cnt, uploaded_by="system", status="ACTIVE")
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    import threading
    try:
        from backend.db.init_db import apply_schema
        apply_schema()
    except Exception:
        pass
    try:
        ensure_default_college()
        ensure_default_admin()
        if not repo.list_course_eligibility():
            from backend.db.seed_eligibility import seed
            seed()
    except Exception:
        pass
    start_scheduler()
    threading.Thread(target=_sync_persistence_and_index, daemon=True).start()
    if LLM_PROVIDER == "qwen":
        try:
            from backend.llm import qwen
            threading.Thread(target=qwen.warmup, daemon=True).start()
        except Exception:
            pass
    yield
    stop_scheduler()


app = FastAPI(title="Admission Chatbot", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(admin_router)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    history: list[dict] = Field(default_factory=list)
    # Client-generated (crypto.randomUUID()), one per page load. Typed as a
    # real UUID so a malformed value is rejected here with a 422 instead of
    # surfacing as a raw psycopg error once it reaches a UUID column.
    conversation_id: uuid.UUID | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


@app.get("/api/health")
def health():
    return {"status": "ok", "college": COLLEGE_NAME, "index": store.stats()}


@app.post("/api/login")
def login(body: LoginRequest):
    if not verify_password(body.username, body.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    user = repo.get_user(body.username)
    role = user.get("role", "admin") if user else "admin"
    token = create_token(body.username, role=role)
    return {"token": token, "username": body.username, "role": role}


@app.post("/api/chat")
def chat(
    body: ChatRequest,
    authorization: str = Header(default=""),
    x_session_token: str = Header(default="", alias="X-Session-Token"),
):
    conv_id = str(body.conversation_id) if body.conversation_id else None
    owner_id = resolve_owner_id(authorization, x_session_token)
    if conv_id and not repo.verify_conversation_owner(conv_id, owner_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return pipeline.answer(body.question, body.history, conv_id, owner_id=owner_id)


@app.post("/api/chat/stream")
def chat_stream(
    body: ChatRequest,
    authorization: str = Header(default=""),
    x_session_token: str = Header(default="", alias="X-Session-Token"),
):
    """Server-sent events so the answer renders while it is generated."""
    conv_id = str(body.conversation_id) if body.conversation_id else None
    owner_id = resolve_owner_id(authorization, x_session_token)
    if conv_id and not repo.verify_conversation_owner(conv_id, owner_id):
        raise HTTPException(status_code=404, detail="Conversation not found")

    def events():
        try:
            for kind, payload in pipeline.answer_stream(body.question, body.history, conv_id, owner_id=owner_id):
                yield f"data: {json.dumps({'type': kind, 'data': payload})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'data': str(exc)})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )



if FRONTEND_DIST.exists():
    # Mounted last so it only catches paths no /api/* route already matched.
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
