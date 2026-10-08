import sys
import os

sys.path.insert(0, os.getcwd())

from backend.rag import counsellor, query_rewrite, pipeline
from backend.rag import retriever

cands = retriever._candidates("what is the hostel fee at sxca?")
pairs = [["what is the hostel fee at sxca?", c["text"]] for c in cands]
scores = retriever._rerank_predict(pairs)
for c, s in zip(cands, scores):
    c["score"] = float(s)
cands.sort(key=lambda x: x["score"], reverse=True)

print("Rank 1 Source:", cands[0]["metadata"].get("source"))
print("Rank 1 Full Text:\n", cands[0]["text"].encode('ascii', errors='backslashreplace').decode('ascii'))
