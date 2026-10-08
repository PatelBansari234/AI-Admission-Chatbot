import sys
import os

sys.path.insert(0, os.getcwd())

from backend.rag import counsellor, query_rewrite, pipeline
from backend.rag import retriever

q = "What is the hostel fee at SXCA?"
understanding = pipeline._understand(q, None, [])
print("Understanding:", understanding)

search_query = understanding["query"]
passages = retriever.retrieve(search_query)
for i, p in enumerate(passages):
    src = p["metadata"].get("source")
    print(f"[{i+1}] Score: {p.get('boosted_score')} | Source: {src}")
    has_men_fee = "Term Fees:" in p["text"]
    has_ladies_fee = "Total Fee Structure" in p["text"]
    print(f"     has_men_fee={has_men_fee}, has_ladies_fee={has_ladies_fee}")
