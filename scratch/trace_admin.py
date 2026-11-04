import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PATH"] = os.environ.get("PATH", "") + r";C:\PostgreSQL17\pgsql\bin"

def trace(msg):
    with open("trace_admin.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

with open("trace_admin.log", "w", encoding="utf-8") as f:
    f.write("Tracing admin.py imports...\n")

trace("a1: auth")
from backend.api.auth import change_password, require_admin

trace("a2: config")
from backend.config import ALLOWED_CRAWL_DOMAIN

trace("a3: intents")
from backend.analytics.intents import compute_intents

trace("a4: crawler")
from backend.crawler.crawler import crawl as run_crawl
from backend.crawler.frontier import FRONTIERS

trace("a5: repo")
from backend.db import repo

trace("a6: ingest")
from backend.rag.ingest import WEB_SUFFIXES, ingest_all, ingest_file

trace("a7: crawl_log")
from backend.utils.crawl_log import load_runs

trace("a8: registry")
from backend.utils.registry import load_registry, remove_document

trace("a9: store")
from backend.vectorstore import store

trace("a10: Done!")
