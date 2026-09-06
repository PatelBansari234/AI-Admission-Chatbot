-- Admission chatbot schema. Applied idempotently by backend/db/init_db.py.
--
-- This deployment serves one college today, but every content table carries
-- college_id so a future multi-college deployment can tell "St. Xavier's,
-- Ahmedabad" apart from a same-named college elsewhere without re-tagging
-- anything already ingested. See backend/db/repo.py's ensure_default_college().

CREATE TABLE IF NOT EXISTS colleges (
    id         SERIAL PRIMARY KEY,
    name       TEXT NOT NULL,
    city       TEXT NOT NULL,
    state      TEXT,
    website    TEXT,
    is_active  BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (name, city)
);

CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'admin',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    college_id  INTEGER REFERENCES colleges(id),
    source      TEXT NOT NULL,
    doc_type    TEXT NOT NULL,
    chunks      INTEGER NOT NULL DEFAULT 0,
    uploaded_by TEXT,
    added_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (college_id, source)
);

-- Per-(re)upload history for a document, mirrors the old registry.json "history" list.
CREATE TABLE IF NOT EXISTS document_history (
    id          SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    uploaded_by TEXT,
    chunks      INTEGER NOT NULL
);

-- One row per browser chat session. id is generated client-side
-- (crypto.randomUUID()) and sent with every request so the frontend can
-- start using it before the first message is even saved - see
-- backend/db/repo.py's get_or_create_conversation(). Defined before
-- chat_history since that table references it.
CREATE TABLE IF NOT EXISTS conversations (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    college_id    INTEGER REFERENCES colleges(id),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_activity TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Structured, cheap-to-read context extracted from the conversation so far
-- (e.g. which course/qualification the student mentioned) - lets the query
-- rewriter resolve "what about the fee?" without re-reading the whole
-- message history on every turn. One row per conversation, overwritten in
-- place as new turns update it.
CREATE TABLE IF NOT EXISTS conversation_memory (
    conversation_id UUID PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    course          TEXT,
    qualification   TEXT,
    summary         TEXT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_history (
    id              SERIAL PRIMARY KEY,
    college_id      INTEGER REFERENCES colleges(id),
    conversation_id UUID REFERENCES conversations(id),
    question        TEXT NOT NULL,
    answer          TEXT NOT NULL,
    grounded        BOOLEAN NOT NULL,
    sources         JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Structured admission facts, extracted (never inferred) from official
-- tables so "which courses can I apply for with 12th Science?" doesn't
-- depend on the LLM reconstructing a table it may only see a fragment of.
-- Columns match what the source documents actually state, not what a
-- schema might ideally want: e.g. eligibility is the verbatim stream/prior-
-- degree text ("12th Science", "BSc - Maths, Stats, Phy, Chem, BA Eco, BE &
-- BCA"), because that's the granularity the college's own admission
-- brochure uses - it never specifies individual required subjects or a
-- minimum percentage. required_subjects/minimum_percentage exist as
-- nullable columns for a future document that might specify them, but must
-- stay NULL rather than be filled with a plausible-sounding guess - NULL
-- here means "not stated by the college," not "unknown, so infer it."
--
-- One row per (course, category) rather than one per course, since a
-- programme can be offered under more than one admission category with a
-- different fee each (e.g. a UG science course as both Government Aided
-- and Self-Financed) - see backend/rag/prompt.py's rule 3 for the same
-- "don't merge different categories" principle applied to RAG answers.
CREATE TABLE IF NOT EXISTS course_eligibility (
    id                 SERIAL PRIMARY KEY,
    college_id         INTEGER REFERENCES colleges(id),
    course_name        TEXT NOT NULL,
    level              TEXT NOT NULL,   -- 'UG' or 'PG'
    category           TEXT NOT NULL,   -- 'grant_in_aid' or 'self_financed'
    eligibility        TEXT NOT NULL,   -- verbatim stream/prior-degree requirement
    required_subjects  TEXT,            -- NULL unless the source names specific subjects
    minimum_percentage TEXT,            -- NULL unless the source states one
    fee_amount         NUMERIC,         -- per semester - the male/general rate if split by gender
    fee_amount_female  NUMERIC,         -- NULL unless the source gives a separate female rate
    fee_note           TEXT,            -- e.g. add-on fees, "*Fee structure is subject to revision"
    contact_email      TEXT,
    academic_year      TEXT NOT NULL,
    source_document    TEXT NOT NULL,
    last_verified      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (college_id, course_name, category, academic_year)
);

CREATE TABLE IF NOT EXISTS crawl_runs (
    id         SERIAL PRIMARY KEY,
    college_id INTEGER REFERENCES colleges(id),
    strategy   TEXT NOT NULL,
    start_url  TEXT NOT NULL,
    max_pages  INTEGER NOT NULL,
    max_depth  INTEGER NOT NULL,
    fetched    INTEGER NOT NULL,
    skipped    INTEGER NOT NULL,
    errors     INTEGER NOT NULL,
    pages      JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Idempotent migrations for a database created before colleges existed - the
-- CREATE TABLE statements above only take effect on a brand new database, so
-- an already-running deployment needs these to actually gain the columns.
ALTER TABLE documents   ADD COLUMN IF NOT EXISTS college_id INTEGER REFERENCES colleges(id);
ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS college_id INTEGER REFERENCES colleges(id);
ALTER TABLE crawl_runs   ADD COLUMN IF NOT EXISTS college_id INTEGER REFERENCES colleges(id);
ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS conversation_id UUID REFERENCES conversations(id);
-- documents.source was globally UNIQUE, but two colleges could share a
-- filename (e.g. both call their prospectus "Prospectus.pdf"), so it needs
-- to move to a per-college constraint instead. init_db.py splits this file
-- on bare semicolons, so every statement here has to stand alone - no DO
-- blocks, and no semicolons inside comments either.
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_source_key;
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_college_source ON documents (college_id, source);

CREATE INDEX IF NOT EXISTS idx_chat_history_created_at ON chat_history (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents (updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_crawl_runs_created_at ON crawl_runs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_document_history_document_id ON document_history (document_id);
CREATE INDEX IF NOT EXISTS idx_documents_college_id ON documents (college_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_college_id ON chat_history (college_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_conversation_id ON chat_history (conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversations_last_activity ON conversations (last_activity DESC);
CREATE INDEX IF NOT EXISTS idx_course_eligibility_college_id ON course_eligibility (college_id);
CREATE INDEX IF NOT EXISTS idx_course_eligibility_course_name ON course_eligibility (course_name);
CREATE INDEX IF NOT EXISTS idx_course_eligibility_eligibility ON course_eligibility (eligibility);

CREATE TABLE IF NOT EXISTS crawler_settings (
    college_id        INTEGER PRIMARY KEY REFERENCES colleges(id),
    scheduler_enabled BOOLEAN NOT NULL DEFAULT true,
    interval_hours    INTEGER NOT NULL DEFAULT 24,
    last_run_at       TIMESTAMPTZ,
    next_run_at       TIMESTAMPTZ,
    last_status       TEXT NOT NULL DEFAULT 'idle',
    last_error        TEXT,
    pages_changed     INTEGER NOT NULL DEFAULT 0,
    pages_new         INTEGER NOT NULL DEFAULT 0,
    pages_unchanged   INTEGER NOT NULL DEFAULT 0,
    pages_failed      INTEGER NOT NULL DEFAULT 0,
    dataset_version   TEXT NOT NULL DEFAULT 'v1.0',
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS page_hashes (
    id           SERIAL PRIMARY KEY,
    college_id   INTEGER REFERENCES colleges(id),
    url          TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    saved_as     TEXT NOT NULL,
    doc_type     TEXT NOT NULL DEFAULT 'website',
    chunks       INTEGER NOT NULL DEFAULT 0,
    version      INTEGER NOT NULL DEFAULT 1,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (college_id, url)
);

CREATE TABLE IF NOT EXISTS dataset_versions (
    id           SERIAL PRIMARY KEY,
    college_id   INTEGER REFERENCES colleges(id),
    version_tag  TEXT NOT NULL,
    description  TEXT,
    files_count  INTEGER NOT NULL DEFAULT 0,
    chunks_count INTEGER NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_page_hashes_url ON page_hashes (url);
CREATE INDEX IF NOT EXISTS idx_dataset_versions_created_at ON dataset_versions (created_at DESC);
