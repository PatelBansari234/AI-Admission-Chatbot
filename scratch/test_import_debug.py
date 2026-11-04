import sys
import os
import traceback

print("Step 1: Setting up PATH and sys.path")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ["PATH"] = os.environ.get("PATH", "") + r";C:\PostgreSQL17\pgsql\bin"

def try_import(name, func):
    print(f"Importing {name}...", flush=True)
    try:
        func()
        print(f"Successfully imported {name}", flush=True)
    except Exception as e:
        print(f"FAILED to import {name}: {e}", flush=True)
        traceback.print_exc()

try_import("backend.crawler.crawler", lambda: __import__("backend.crawler.crawler"))
try_import("backend.rag.ingest", lambda: __import__("backend.rag.ingest"))
try_import("backend.vectorstore.store", lambda: __import__("backend.vectorstore.store"))
try_import("backend.crawler.scheduler", lambda: __import__("backend.crawler.scheduler"))

try_import("backend.api.auth", lambda: __import__("backend.api.auth"))
try_import("backend.vectorstore.store", lambda: __import__("backend.vectorstore.store"))
try_import("backend.rag.pipeline", lambda: __import__("backend.rag.pipeline"))
try_import("backend.api.admin", lambda: __import__("backend.api.admin"))
try_import("backend.api.main", lambda: __import__("backend.api.main"))
