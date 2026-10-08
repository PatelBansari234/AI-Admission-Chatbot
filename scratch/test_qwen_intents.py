import sys, os
sys.path.insert(0, os.getcwd())
from backend.rag import query_rewrite

queries = [
    "What is the history of St. Xavier's College Ahmedabad?",
    "What are the rules regarding student attendance at SXCA?",
    "Tell me about the IQAC cell at SXCA",
    "What are the research facilities at SXCA?",
]

for q in queries:
    u = query_rewrite.understand(q, [], None)
    print(f"'{q}' -> intent: {u.get('intent')}, course: {u.get('course')}, needs_structured_lookup: {u.get('needs_structured_lookup')}")
