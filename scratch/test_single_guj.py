import sys
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from backend.rag import pipeline

pipeline.cache.clear()
res = pipeline.answer("bca ma admission levu che process su che?")
print("ANSWER:\n", res["answer"])
