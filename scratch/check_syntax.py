import sys
import os
import traceback

sys.path.insert(0, os.getcwd())

try:
    print("Importing config...")
    from backend import config
    print("Importing qwen...")
    from backend.llm import qwen
    print("Importing counsellor...")
    from backend.rag import counsellor
    print("Importing pipeline...")
    from backend.rag import pipeline
    print("SUCCESS: All imports loaded cleanly!")
except Exception as e:
    traceback.print_exc()
