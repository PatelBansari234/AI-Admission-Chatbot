import sys
import os

sys.path.insert(0, os.getcwd())

from backend.rag import pipeline

try:
    print("Calling pipeline.answer('What is the hostel fee at SXCA?')...")
    res = pipeline.answer("What is the hostel fee at SXCA?")
    print("Result:", res)
except Exception:
    import traceback
    traceback.print_exc()
