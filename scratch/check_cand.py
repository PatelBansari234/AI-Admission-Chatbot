import sys
import os

sys.path.insert(0, os.getcwd())

from backend.rag import counsellor, query_rewrite, pipeline
from backend.rag import retriever

q = "What is the hostel fee at SXCA?"
norm_q = retriever.normalize_query(q)
candidates = retriever._candidates(norm_q)

pairs = [[norm_q, c["text"]] for c in candidates]
scores = retriever._rerank_predict(pairs)

for c, s in zip(candidates, scores):
    c["rerank_score"] = float(s)

candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

print("--- Top 10 candidates for 'What is the hostel fee at SXCA?' ---")
for i, c in enumerate(candidates[:10]):
    src = c["metadata"].get("source")
    has_hostel = "hostel" in c["text"].lower()
    has_fee = any(tok in c["text"].lower() for tok in ["fee", "fees", "rent", "₹", "rs"])
    print(f"Rank {i+1}: score={c['rerank_score']:.4f} | has_hostel={has_hostel} | has_fee={has_fee} | src={src}")
