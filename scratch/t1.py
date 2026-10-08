import sys
import os
sys.path.insert(0, os.getcwd())

print("HELLO STEP 1")
from backend.rag import counsellor, pipeline, query_rewrite
print("HELLO STEP 2")
from backend.llm import qwen
print("HELLO STEP 3")
