import sys
import os

sys.path.insert(0, os.getcwd())

from backend.rag import counsellor, query_rewrite, pipeline
from backend.rag import retriever
from backend.vectorstore import store

q = "What is the hostel fee at SXCA?"
norm_q = retriever.normalize_query(q)

all_chunks = store.get_all_chunks(college_id=1)
for i, c in enumerate(all_chunks):
    if "Term Fees: \u20b910,000" in c["text"] or "Term Fees: ₹10,000" in c["text"]:
        score = retriever._rerank_predict([[norm_q, c["text"]]])
        print(f"ACTUAL Men's Fee Chunk (len={len(c['text'])}): rerank score = {float(score[0]):.4f}")
        print("Text snippet:", c["text"])

for i, c in enumerate(all_chunks):
    if "For one year stay in the hostel" in c["text"]:
        score = retriever._rerank_predict([[norm_q, c["text"]]])
        print(f"ACTUAL Ladies Fee Chunk (len={len(c['text'])}): rerank score = {float(score[0]):.4f}")
        print("Text snippet preview:", c["text"][:300])
