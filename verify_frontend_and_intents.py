import os, sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())

from backend.rag.query_rewrite import understand
from backend.rag.pipeline import answer

print("=== 1. TESTING INTENT PARSING ===")
queries = [
    "who is pravida raja",
    "there is swimming pool in the clg ?",
    "what is today date",
    "list of ug cource",
    "14 sep ae raja che ?",
    "there is holiday on Janmashtami",
    "what is the eligibility for physics?",
    "what are the fees for physics?",
    "give me link to apply MSc AI",
    "details about Python programming",
]

for q in queries:
    u = understand(q, [], None)
    print(f"Query: '{q}' -> Intent: {u.get('intent')}")

print("\n=== 2. TESTING PIPELINE ANSWERS ===")
test_runs = [
    "who is pravida raja",
    "there is swimming pool in the clg ?",
    "what is today date",
    "list of ug cource",
]

for q in test_runs:
    res = answer(q, [], None)
    ans = res["answer"]
    preview = ans.split("\n")[0] if ans else ""
    print(f"\n[Q]: {q}\n[A Preview]: {preview}\n[Grounded]: {res.get('grounded')}")

print("\nALL CHECKS COMPLETED SUCCESSFULLY!")
