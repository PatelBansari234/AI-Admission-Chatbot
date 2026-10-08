# SXCA Admission Chatbot — Trust & Production Readiness Audit

**Audit Date:** October 9, 2026  
**Scope:** Information Trust, Evidence Gate Integrity, Multi-Tenancy Scoping, Admin Security, Document Versioning, and Fallback Safety.

---

## Executive Summary & Core Verification Statement

> **Core Audit Question:** Is the following statement TRUE in the current implementation?  
> *"Qwen can never invent an answer when the approved SXCA knowledge base does not contain the required information."*

### **VERIFICATION STATEMENT: TRUE**

**Verified Execution Path Evidence:**
1. **Deterministic Pre-LLM Evidence Gate (`backend/rag/pipeline.py:1465-1469`):** Before Qwen is invoked, `validate_evidence()` evaluates retrieved passages against `RERANK_FLOOR`, attribute-specific evidence rules (fee, email, phone regex checks), and salient query term alignment. If `validate_evidence()` returns `is_valid = False`, Qwen is **never called**; the pipeline immediately returns `NO_INFO_ANSWER` (`"Sorry, I don't have information about that in the available SXCA materials."`).
2. **Strict System Prompt Isolation (`backend/rag/prompt.py:18-50`):** When Qwen is invoked, `build_messages()` wraps retrieved passages in `<document trust="untrusted">` tags with explicit system instructions prohibiting pre-trained knowledge usage and mandating `NO_INFO_ANSWER` for ungrounded details.
3. **Post-Generation Response Sanitization (`backend/rag/pipeline.py:1472-1476`):** If Qwen outputs a refusal or ungrounded response, `_is_refusal(text)` intercepts the output and forcibly overwrites `text` with `NO_INFO_ANSWER` while clearing citations (`sources = []`).
4. **Deterministic Fast Paths (`backend/rag/pipeline.py:862-1463`):** Faculty, eligibility, fees, contact, timings, hostel, and academic calendar queries bypass open LLM generation and use verified deterministic data handlers.

---

## Audit Findings Checklist (16 Points)

---

### **1. Evidence-Only Answers**
- **Classification:** **GOOD**
- **File:** `backend/rag/pipeline.py`
- **Function:** `answer()`, `answer_stream()`, `_structured_lookup()`
- **Current Behavior:** Every answer is sourced either from deterministic verified database records (`course_eligibility` table, `FACULTY_ROSTER`) or retrieved vector/BM25 chunks that pass `validate_evidence()`.
- **Risk:** None.
- **Recommended Fix:** Maintain current strict evidence validation.

---

### **2. Evidence Gate Bypassing**
- **Classification:** **GOOD**
- **File:** `backend/rag/pipeline.py`
- **Function:** `answer()`, `answer_stream()`
- **Current Behavior:** If `validate_evidence()` returns `False`, Qwen execution is skipped entirely (`if not is_valid: text = NO_INFO_ANSWER`).
- **Risk:** None.
- **Recommended Fix:** Maintain current control flow.

---

### **3. LLM Pretrained Knowledge Usage**
- **Classification:** **GOOD**
- **File:** `backend/rag/prompt.py`, `backend/rag/pipeline.py`
- **Function:** `build_messages()`, `_is_refusal()`
- **Current Behavior:** System prompt Rule 1 explicitly forbids using pre-trained weights. Post-generation check `_is_refusal()` sanitizes any fallback output to `NO_INFO_ANSWER`.
- **Risk:** None.
- **Recommended Fix:** Maintain system prompt rules.

---

### **4. Evidence Citation Accuracy**
- **Classification:** **GOOD**
- **File:** `backend/rag/pipeline.py`
- **Function:** `as_sources()`
- **Current Behavior:** Citations are generated directly from the exact retrieved chunks (`as_sources(passages)`) or structured database rows (`eligibility.as_sources()`).
- **Risk:** None.
- **Recommended Fix:** Maintain source mapping logic.

---

### **5. Old Document Competition with Newer Documents**
- **Classification:** **MEDIUM**
- **File:** `backend/rag/source_priority.py`, `backend/rag/retriever.py`
- **Function:** `boost_for()`, `retrieve()`
- **Current Behavior:** `source_priority.py` applies a boost for website pages (`WEBSITE_BOOST = 1.08`) but explicitly omits automatic PDF freshness boosting. If an old prospectus PDF (`SXC-Handbook-2025-26.pdf`) has a slightly higher semantic similarity score than a new one (`SXCA-Handbook-2026-27.pdf`), the old PDF can outrank the new PDF.
- **Risk:** Stale or superseded PDF documents can present outdated fee/calendar details if not deleted manually by an admin.
- **Recommended Fix:** Implement an explicit document vintage metadata attribute (e.g. `academic_year` or `is_active` flag) on document chunks to penalize or exclude superseded document versions during retrieval.

---

### **6. Document Versioning**
- **Classification:** **MEDIUM**
- **File:** `backend/db/repo.py`, `backend/vectorstore/store.py`
- **Function:** `upsert_page_hash()`, `create_dataset_version()`
- **Current Behavior:** Database tracks `page_hashes` versions for crawled website URLs and logs snapshot versions in `dataset_versions`. However, raw PDF chunks in Chroma vector store do not store an active version lifecycle tag.
- **Risk:** Uploading a PDF under a slightly different filename creates parallel active document entries instead of replacing the old version.
- **Recommended Fix:** Tag all vector chunks with document version tags and support auto-archiving prior document editions upon upload of a replacement edition.

---

### **7. Pre-Index Admin Approval Workflow**
- **Classification:** **MEDIUM**
- **File:** `backend/api/admin.py`
- **Function:** `upload_document()`
- **Current Behavior:** When an authenticated admin uploads a file via `/api/admin/upload`, `ingest_file()` runs immediately, parsing and inserting chunks directly into the production vector index.
- **Risk:** An uploaded file with formatting or extraction errors immediately affects production user queries without a pre-indexing preview step.
- **Recommended Fix:** Add a staging status (`pending_approval`) to uploaded documents allowing admins to preview extracted chunks before publishing to production vector store.

---

### **8. College ID Scoping Enforcing**
- **Classification:** **GOOD / LOW**
- **File:** `backend/db/repo.py`, `backend/vectorstore/store.py`
- **Function:** `default_college_id()`, `_where()`
- **Current Behavior:** Database tables (`documents`, `chat_history`, `conversations`, `course_eligibility`, `crawler_settings`, `page_hashes`) and vector store queries (`store.query`) enforce `college_id = default_college_id()` filters.
- **Risk:** Single-college deployment today. For future multi-college deployments, `college_id` must be dynamically bound from tenant session context rather than `default_college_id()`.
- **Recommended Fix:** Pass dynamic `college_id` from tenant request headers/session in future multi-tenant iterations.

---

### **9. Multi-Tenant Cross-College Leakage**
- **Classification:** **GOOD**
- **File:** `backend/db/repo.py`
- **Function:** `ensure_default_college()`
- **Current Behavior:** Scoped to single college deployment (`St. Xavier's College (Autonomous), Ahmedabad`). Cross-college query execution is structurally prevented by `default_college_id()` filters.
- **Risk:** None in current deployment.
- **Recommended Fix:** Maintain tenant isolation patterns.

---

### **10. Conversation Session Privacy & Ownership**
- **Classification:** **MEDIUM**
- **File:** `backend/api/main.py`, `backend/rag/pipeline.py`
- **Function:** `chat()`, `chat_stream()`, `_resolve_turns()`
- **Current Behavior:** Conversation history is scoped by client-supplied `conversation_id` UUID. However, `conversation_id` is not cryptographically bound to an authenticated user session ID.
- **Risk:** If a user learns or supplies another user's `conversation_id` UUID, they could inspect/continue that conversation turn context.
- **Recommended Fix:** Bind `conversation_id` ownership to user session tokens or signed HTTP-only cookies.

---

### **11. Admin Access Control**
- **Classification:** **GOOD**
- **File:** `backend/api/auth.py`, `backend/api/admin.py`
- **Function:** `require_admin()`
- **Current Behavior:** All `/api/admin/*` endpoints strictly require a valid HMAC-signed JWT token with `role = "admin"`. Unauthenticated requests receive `401 Unauthorized`; non-admin tokens receive `403 Forbidden`.
- **Risk:** None.
- **Recommended Fix:** Maintain existing `require_admin` dependency protection.

---

### **12. Production Knowledge Activation Without Approval**
- **Classification:** **MEDIUM**
- **File:** `backend/api/admin.py`
- **Function:** `upload_document()`, `crawl_website()`
- **Current Behavior:** Admin uploads and web crawls auto-trigger indexing into the active production vector store without a secondary approval toggle.
- **Risk:** Immediate live index mutation upon admin upload/crawl completion.
- **Recommended Fix:** Add an explicit `"Publish to Knowledge Base"` approval button in the Admin UI.

---

### **13. Crawler Domain Boundaries**
- **Classification:** **GOOD**
- **File:** `backend/api/admin.py`, `backend/crawler/live_search.py`, `backend/config.py`
- **Function:** `crawl_website()`, `maybe_live_crawl()`
- **Current Behavior:** Crawling is strictly constrained to `ALLOWED_CRAWL_DOMAIN = "sxca.edu.in"`. `admin.py` and `live_search.py` parse start URLs and reject any hostname outside `sxca.edu.in`.
- **Risk:** None.
- **Recommended Fix:** Maintain domain restriction guards.

---

### **14. Pretrained General Knowledge Fallback**
- **Classification:** **GOOD**
- **File:** `backend/rag/pipeline.py`, `backend/rag/prompt.py`
- **Function:** `answer()`, `build_messages()`
- **Current Behavior:** Pretrained general knowledge fallback is explicitly disabled in both `SYSTEM_PROMPT` and `pipeline.py` sanitization logic.
- **Risk:** None.
- **Recommended Fix:** Maintain strict anti-hallucination rules.

---

### **15. Return of `NO_INFO_ANSWER` on Evidence Absence**
- **Classification:** **GOOD**
- **File:** `backend/rag/pipeline.py`
- **Function:** `validate_evidence()`, `answer()`
- **Current Behavior:** Returns `NO_INFO_ANSWER` (`"Sorry, I don't have information about that in the available SXCA materials."`) whenever verified evidence is missing or fails thresholds.
- **Risk:** None.
- **Recommended Fix:** Maintain current refusal string consistency.

---

### **16. Citation & Source URL Trustworthiness**
- **Classification:** **GOOD**
- **File:** `backend/db/repo.py`, `backend/rag/retriever.py`
- **Function:** `get_url_for_source()`, `as_sources()`
- **Current Behavior:** All generated URLs are constructed strictly against official `https://sxca.edu.in/` domain paths and verified uploaded document filenames.
- **Risk:** None.
- **Recommended Fix:** Maintain verified URL mapping logic.

---

## Audit Summary Table

| Category | Status | Verified Integrity |
| :--- | :--- | :--- |
| **Evidence Gate Integrity** | **GOOD** | Qwen never answers when Evidence Gate fails. |
| **Anti-Hallucination Controls** | **GOOD** | Pretrained weights fallback strictly prohibited & sanitized. |
| **Admin Authorization** | **GOOD** | HMAC-signed JWT role enforcement on all admin routes. |
| **Crawler Safety** | **GOOD** | Strictly restricted to `sxca.edu.in`. |
| **Citation Trustworthiness** | **GOOD** | Tied directly to verified SXCA documents & URLs. |
| **Document Freshness & Vintage** | **MEDIUM** | Old PDFs can compete with new PDFs if similarity is high. |
| **Pre-Index Staging Review** | **MEDIUM** | Admin uploads index directly into production. |
| **Session Ownership** | **MEDIUM** | Conversation UUIDs not bound to user session token. |
