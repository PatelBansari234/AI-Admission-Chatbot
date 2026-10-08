import sys
import os

sys.path.insert(0, os.getcwd())

from backend.rag import counsellor, query_rewrite, pipeline

questions = [
    "What is the capital of France?",
    "Who founded Microsoft?",
    "Tell me about aerospace engineering at SXCA",
    "What are the research labs in astrophysics at SXCA?",
    "How to make pasta at home?"
]

for q in questions:
    res = pipeline.answer(q)
    print(f"\nQ: {q}")
    print(f"Answer: {res['answer']}")
    print(f"Grounded: {res.get('grounded')}")
    print(f"Sources count: {len(res.get('sources', []))}")
