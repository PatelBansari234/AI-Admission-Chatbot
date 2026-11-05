"""Central configuration. Everything path-related is derived from PROJECT_ROOT."""

import os
import re
from pathlib import Path
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlparse, urlsplit, urlunsplit

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

if os.name == "nt":
    for _pg_path in (r"C:\PostgreSQL17\pgsql\bin", r"C:\Program Files\PostgreSQL\17\bin", r"C:\Program Files\PostgreSQL\16\bin"):
        if os.path.exists(_pg_path) and _pg_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = os.environ.get("PATH", "") + ";" + _pg_path
            break


def _is_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _detect_low_memory_mode() -> bool:
    raw = os.getenv("LOW_MEMORY_MODE", "").strip().lower()
    if raw in ("1", "true", "yes"):
        return True
    if raw in ("0", "false", "no"):
        return False
    if os.getenv("RENDER", "").strip().lower() == "true":
        return True
    try:
        import psutil
        return psutil.virtual_memory().total <= int(1.5 * (1024**3))
    except Exception:
        return False


LOW_MEMORY_MODE = _detect_low_memory_mode()


# --- data layout -----------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw" / "pdfs"
RAW_WEB_DIR = DATA_DIR / "raw" / "websie"
RAW_WEB_CRAWL_DIR = RAW_WEB_DIR / "crawled"
# On-site PDFs the crawler discovers linked from HTML pages (e.g. an academic
# calendar button that links to a PDF rather than another page) - kept
# separate from admin-uploaded PDFs in RAW_PDF_DIR itself.
RAW_PDF_CRAWL_DIR = RAW_PDF_DIR / "crawled"
PROCESSED_DIR = DATA_DIR / "processed"

VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
COLLECTION_NAME = "sxca_admissions"

# --- database ----------------------------------------------------------
# Users, the document registry, chat history, and crawl logs all live in
# Postgres (backend/db/) - see backend/db/schema.sql and init_db.py. The
# vector index (Chroma) and model/data files on disk stay outside it.
_VALID_SSLMODES = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}


def _clean_env_val(val: str) -> str:
    return (val or "").strip().strip("'\"`").strip()


_URI_PATTERN = re.compile(
    r"postgres(?:ql)?(?:\+[a-z0-9_]+)?://([^:/\s'\"`]+)(?::([^\s'\"`]*)?)?@([^/@:\s'\"`?#]+)(?::(\d+))?/([^?#\s'\"`]+)(?:\?([^\s'\"`#]*))?",
    re.I,
)


def _parse_and_normalize_db_config() -> tuple[str, str, str, str, str, str]:
    raw_url = os.getenv("DATABASE_URL", "").strip()
    raw_host = _clean_env_val(os.getenv("DB_HOST", ""))
    raw_name = _clean_env_val(os.getenv("DB_NAME", ""))

    # Recover gracefully if the full Neon URI was pasted into DB_HOST or DB_NAME
    if not raw_url:
        for candidate in (raw_host, raw_name):
            if "postgres://" in candidate.lower() or "postgresql://" in candidate.lower():
                raw_url = candidate
                break

    if raw_url:
        m = _URI_PATTERN.search(raw_url)
        if m:
            u_raw, p_raw, h_raw, port_raw, db_raw, q_raw = m.groups()
            host = h_raw.strip()
            port = (port_raw or "5432").strip()
            dbname = db_raw.split("?")[0].strip("'\"`") or "neondb"
            user = unquote(u_raw or "") or "admission_bot"
            password = unquote(p_raw or "")
            q_dict = dict(parse_qsl(q_raw or "", keep_blank_values=True))
        else:
            clean_uri = _clean_env_val(raw_url)
            clean_uri = re.sub(r"^postgres(?:ql)?(?:\+[a-z0-9_]+)?://", "postgresql://", clean_uri, flags=re.I)
            sp = urlsplit(clean_uri)
            host = sp.hostname or "127.0.0.1"
            port = str(sp.port or 5432)
            dbname = (sp.path or "").lstrip("/").split("?")[0].strip("'\"`") or "admission_chatbot"
            user = unquote(sp.username or "") or "admission_bot"
            password = unquote(sp.password or "")
            q_dict = dict(parse_qsl(sp.query, keep_blank_values=True))

        sslmode = _clean_env_val(q_dict.get("sslmode", "")).lower()
        is_local = host.lower() in ("127.0.0.1", "localhost", "::1")
        if sslmode not in _VALID_SSLMODES:
            sslmode = "" if is_local else "require"

        auth_part = f"{quote(user, safe='')}:{quote(password, safe='')}" if password else quote(user, safe="")
        netloc = f"{auth_part}@{host}:{port}"
        query = urlencode({"sslmode": sslmode}) if sslmode else ""
        final_url = urlunsplit(("postgresql", netloc, f"/{dbname}", query, ""))
        return host, port, dbname, user, password, final_url

    host = raw_host or "127.0.0.1"
    port = _clean_env_val(os.getenv("DB_PORT", "5432")) or "5432"
    dbname = raw_name.split("?")[0] or "admission_chatbot"
    user = _clean_env_val(os.getenv("DB_USER", "admission_bot")) or "admission_bot"
    password = _clean_env_val(os.getenv("DB_PASSWORD", ""))
    is_local = host.lower() in ("127.0.0.1", "localhost", "::1")
    auth_part = f"{quote(user, safe='')}:{quote(password, safe='')}" if password else quote(user, safe="")
    query = "" if is_local else "?sslmode=require"
    final_url = f"postgresql://{auth_part}@{host}:{port}/{dbname}{query}"
    return host, port, dbname, user, password, final_url


DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DATABASE_URL = _parse_and_normalize_db_config()

# --- models ----------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
LLM_PATH = os.getenv("LLM_PATH", str(MODELS_DIR / "Qwen2.5-3B-Instruct"))
LLM_DEVICE = os.getenv("LLM_DEVICE", "auto")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "350"))

# QLoRA adapter trained on logged Q&A (tone/style) - see backend/finetune/.
# The base model above supplies the weights; this is loaded on top of it.
FINE_TUNED_ADAPTER_DIR = MODELS_DIR / "qwen-lora-adapter"
USE_FINE_TUNED = os.getenv("USE_FINE_TUNED", "false").lower() == "true"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip() or "gemini-3.8-flash"

# Inference provider: defaults to "gemini" when GEMINI_API_KEY is configured,
# or can be explicitly set via LLM_PROVIDER="gemini" | "qwen".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini" if GEMINI_API_KEY else "qwen").strip().lower()

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")  # keep the GPU free for Qwen
EMBED_CACHE = str(MODELS_DIR / "embeddings")
# bge models expect this prefix on the *query* side only.
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- retrieval -------------------------------------------------------------
# Bigger chunks than before (900/150): a 900-char chunk could split a short
# entry - e.g. a faculty listing's name, title and committee membership - in
# half at exactly the wrong point, leaving the model only part of it. 1600
# comfortably holds a whole entry like that, with enough overlap that a cut
# elsewhere still leaves both neighbouring chunks intact.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))

# Two-stage retrieval: PDF and website chunks are searched as two SEPARATE
# candidate pools (vector + BM25 each), so a much bigger source can't crowd a
# smaller one out of a shared top-k before reranking ever sees it. Each pool
# is capped at PER_SOURCE_CANDIDATES, then both pools are merged and a
# cross-encoder reranker scores every candidate against the *full* question
# (not just cosine similarity) - only the best TOP_K actually go to the LLM.
# See backend/rag/retriever.py.
PER_SOURCE_CANDIDATES = int(os.getenv("PER_SOURCE_CANDIDATES", "12"))
TOP_K = int(os.getenv("TOP_K", "3"))

RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
# CPU reranking measured ~6.3s for a real 40-candidate batch (1600-char
# chunks) - the actual bottleneck of the whole request, not the LLM. GPU cut
# that to ~0.4-0.6s in testing, but VRAM is tight with Qwen also loaded
# (~1.3GB free on an 8GB card): a small RERANK_BATCH_SIZE keeps peak usage
# under that budget (batch_size=8 measured ~1.29GB peak, and was faster than
# the default batch_size=32's ~1.71GB peak, not slower). retriever.py still
# falls back to CPU for that one request if the GPU genuinely runs out of
# memory, rather than failing the request.
_env_rerank_dev = os.getenv("RERANK_DEVICE", "").strip().lower()
if LOW_MEMORY_MODE or _env_rerank_dev == "cpu":
    RERANK_DEVICE = "cpu"
else:
    RERANK_DEVICE = _env_rerank_dev or ("cuda" if _is_cuda_available() else "cpu")
    if RERANK_DEVICE == "cuda" and not _is_cuda_available():
        RERANK_DEVICE = "cpu"

RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "8"))
# A single fixed cutoff on reranker score doesn't work: calibration showed a
# genuinely correct match for a terse query ("holiday list ?" -> the actual
# Summer Break passage) scoring only 0.045, while a well-phrased question's
# correct match scored 0.999 - both real, wildly different scale. So instead
# of one absolute bar, a candidate is kept if it clears a low noise floor
# AND is within RERANK_RELATIVE_MARGIN of the top score - i.e. "close to the
# best result", not "above some universal number". Off-topic queries
# calibrated to a clean ~0.0 top score in testing, so the floor alone
# correctly empties the result for those.
RERANK_FLOOR = float(os.getenv("RERANK_FLOOR", "0.02"))
RERANK_RELATIVE_MARGIN = float(os.getenv("RERANK_RELATIVE_MARGIN", "0.5"))

# --- auth ------------------------------------------------------------------
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
DEFAULT_ADMIN_USER = os.getenv("ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
TOKEN_TTL_SECONDS = int(os.getenv("TOKEN_TTL_SECONDS", str(8 * 3600)))

COLLEGE_NAME = os.getenv("COLLEGE_NAME", "St. Xavier's College (Autonomous), Ahmedabad")
# This deployment serves exactly one college today, but the schema (colleges
# table, college_id on documents/chunks/chat_history) is built for more than
# one - every document, chunk, and logged question is tagged with a
# college_id so a future multi-college deployment can filter "St. Xavier's,
# Ahmedabad" apart from a same-named college elsewhere without re-tagging
# existing data. See backend/db/schema.sql and repo.ensure_default_college().
COLLEGE_CITY = os.getenv("COLLEGE_CITY", "Ahmedabad")
COLLEGE_STATE = os.getenv("COLLEGE_STATE", "Gujarat")
COLLEGE_WEBSITE = os.getenv("COLLEGE_WEBSITE", "https://sxca.edu.in/")

# --- intent clustering ---------------------------------------------------
# Cosine distance below which two logged questions are folded into the same
# intent cluster. Chosen empirically, biased toward precision: pairs that
# share an entity but differ in intent (e.g. "seats for BA Economics" vs
# "duration of BA Economics") scored ~0.19 distance in testing - uncomfortably
# close to genuine paraphrases (~0.16), so this sits below that gap rather
# than at a threshold that would risk merging them. See backend/analytics/.
INTENT_DISTANCE_THRESHOLD = float(os.getenv("INTENT_DISTANCE_THRESHOLD", "0.17"))

# --- crawler -----------------------------------------------------------
# Master switch for the live-crawl-on-retrieval-miss fallback in pipeline.py.
# Off means a question the index can't answer just gets the fixed refusal -
# no reaching out to the website at all. Set False when the index is meant
# to be scoped to specific uploaded documents only (e.g. one prospectus).
LIVE_CRAWL_FALLBACK_ENABLED = os.getenv("LIVE_CRAWL_FALLBACK_ENABLED", "false").lower() == "true"

# The admin crawl endpoint only accepts start URLs on this domain, so an
# authenticated-but-careless (or compromised) admin session can't be used to
# make the server fetch arbitrary internal/external hosts (SSRF).
ALLOWED_CRAWL_DOMAIN = os.getenv("ALLOWED_CRAWL_DOMAIN", "sxca.edu.in")
CRAWL_USER_AGENT = os.getenv(
    "CRAWL_USER_AGENT", "AdmissionChatbotCrawler/1.0 (student project; read-only)"
)
CRAWL_TIMEOUT = int(os.getenv("CRAWL_TIMEOUT", "10"))
CRAWL_DEFAULT_DELAY = float(os.getenv("CRAWL_DEFAULT_DELAY", "1.0"))
CRAWL_DEFAULT_MAX_PAGES = int(os.getenv("CRAWL_DEFAULT_MAX_PAGES", "30"))
CRAWL_DEFAULT_MAX_DEPTH = int(os.getenv("CRAWL_DEFAULT_MAX_DEPTH", "3"))
# Hard ceilings the admin API enforces regardless of what's requested, so a
# typo (or a malicious request) can't turn this into a denial-of-service tool.
CRAWL_HARD_MAX_PAGES = int(os.getenv("CRAWL_HARD_MAX_PAGES", "200"))
CRAWL_HARD_MAX_DEPTH = int(os.getenv("CRAWL_HARD_MAX_DEPTH", "5"))

for _d in (RAW_PDF_DIR, RAW_PDF_CRAWL_DIR, RAW_WEB_DIR, RAW_WEB_CRAWL_DIR, PROCESSED_DIR, VECTOR_DB_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
