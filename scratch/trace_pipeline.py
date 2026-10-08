import sys
import os

sys.path.insert(0, os.getcwd())

from backend.rag import counsellor, query_rewrite, pipeline
from backend.rag import retriever

q = "What is the hostel fee at SXCA?"
understanding = pipeline._understand(q, None, [])
print("Understanding intent:", understanding.get("intent"))
print("Understanding query:", understanding.get("query"))
print("Understanding course:", understanding.get("course"))

search_query = understanding["query"]
print("\nCalling retriever.retrieve with search_query:", search_query)
passages = retriever.retrieve(search_query)
print("Passages count:", len(passages))
for i, p in enumerate(passages):
    src = p["metadata"].get("source")
    score = p.get("boosted_score")
    print(f"[{i+1}] Score: {score} | Source: {src}")

is_valid, reason = pipeline.validate_evidence(q, passages, [], understanding)
print(f"\nvalidate_evidence with q='{q}': is_valid={is_valid}, reason={reason}")

is_valid2, reason2 = pipeline.validate_evidence(search_query, passages, [], understanding)
print(f"validate_evidence with search_query: is_valid={is_valid2}, reason={reason2}")
