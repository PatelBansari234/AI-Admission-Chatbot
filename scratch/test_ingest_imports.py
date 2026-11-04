import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PATH"] = os.environ.get("PATH", "") + r";C:\PostgreSQL17\pgsql\bin"


def test(name, code):
    t0 = time.time()
    print(f"Testing {name}...", end="", flush=True)
    code()
    print(f" OK ({time.time() - t0:.2f}s)", flush=True)

test("backend.config", lambda: __import__("backend.config"))
test("backend.db.repo", lambda: __import__("backend.db.repo"))
test("backend.embeddings.embedder", lambda: __import__("backend.embeddings.embedder"))
test("langchain_text_splitters", lambda: __import__("langchain_text_splitters"))
test("backend.preprocessing.chunker", lambda: __import__("backend.preprocessing.chunker"))
test("backend.preprocessing.html_loader", lambda: __import__("backend.preprocessing.html_loader"))
test("pdfplumber", lambda: __import__("pdfplumber"))
test("backend.preprocessing.pdf_loader", lambda: __import__("backend.preprocessing.pdf_loader"))
test("backend.preprocessing.table_extractor", lambda: __import__("backend.preprocessing.table_extractor"))
test("backend.vectorstore", lambda: __import__("backend.vectorstore"))
test("backend.rag.ingest", lambda: __import__("backend.rag.ingest"))
