import sys
import os
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PATH"] = os.environ.get("PATH", "") + r";C:\PostgreSQL17\pgsql\bin"

def trace(msg):
    print(msg, flush=True)

trace("1: fastapi")
import fastapi

trace("2: backend.config")
import backend.config

trace("3: backend.db.repo")
import backend.db.repo

trace("4: backend.api.auth")
import backend.api.auth

trace("5: backend.vectorstore.store")
import backend.vectorstore.store

trace("6: backend.crawler.scheduler")
import backend.crawler.scheduler

trace("7: backend.rag.pipeline")
import backend.rag.pipeline

trace("8: backend.api.admin")
import backend.api.admin

trace("9: backend.api.main")
import backend.api.main

trace("ALL SUCCESSFUL!")
