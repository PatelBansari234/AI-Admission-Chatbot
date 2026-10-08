import sys
import os
import re

sys.path.insert(0, os.getcwd())

from backend.rag import counsellor, query_rewrite, pipeline
from backend.rag import retriever

STOP_WORDS = {
    "what", "is", "are", "the", "at", "in", "of", "for", "to", "a", "an", "and", "or",
    "by", "on", "sxca", "college", "st", "xavier", "xaviers", "ahmedabad", "can", "i",
    "you", "me", "tell", "about", "how", "much", "does", "do", "please", "there", "any",
    "with", "from", "be", "have", "has", "had", "this", "that", "it", "its"
}

def extract_distinctive_terms(question: str) -> list[str]:
    raw_tokens = retriever._tokenize(question)
    distinctive = []
    for t in raw_tokens:
        if t not in STOP_WORDS and len(t) >= 3:
            distinctive.append(t)
    return distinctive

def check_alignment(distinctive_terms: list[str], text: str) -> float:
    if not distinctive_terms:
        return 0.0
    text_tokens = set(retriever._tokenize(text))
    matches = 0
    for term in distinctive_terms:
        # direct match or singular/plural stem match
        base = term.rstrip('s')
        if any(tok == term or (len(base) >= 3 and tok.rstrip('s') == base) for tok in text_tokens):
            matches += 1
    return matches / len(distinctive_terms)

questions = [
    "What is the hostel fee at SXCA?",
    "What is the history of St. Xavier's College Ahmedabad?",
    "What is Nisarg Vyas's email?",
    "What is the swimming pool fee at SXCA?",
    "How to make pizza in Ahmedabad?"
]

for q in questions:
    terms = extract_distinctive_terms(q)
    print(f"\n==========================================")
    print(f"Query: {q}")
    print(f"Distinctive terms: {terms}")
    norm_q = retriever.normalize_query(q)
    cands = retriever._candidates(norm_q)
    print(f"Total candidates: {len(cands)}")
    pairs = [[norm_q, c["text"]] for c in cands]
    scores = retriever._rerank_predict(pairs)
    
    for c, s in zip(cands, scores):
        c["rerank_score"] = float(s)
        c["alignment"] = check_alignment(terms, c["text"])
        
    cands.sort(key=lambda c: c["rerank_score"], reverse=True)
    top_score = cands[0]["rerank_score"] if cands else 0.0
    print(f"Top rerank score: {top_score:.4f} (Source: {cands[0]['metadata'].get('source') if cands else 'None'})")
    
    print("\nTop 5 by rerank score:")
    for c in cands[:5]:
        src = c["metadata"].get("source")
        print(f"  Score: {c['rerank_score']:.4f} | Alignment: {c['alignment']*100:.0f}% | Source: {src} | Snippet: {repr(c['text'][:80])}")
        
    aligned_cands = [c for c in cands if c["alignment"] >= 1.0 and c["rerank_score"] >= 0.02]
    print(f"\nCandidates with 100% distinctive term alignment and score >= 0.02: {len(aligned_cands)}")
    for c in aligned_cands:
        src = c["metadata"].get("source")
        print(f"  -> Score: {c['rerank_score']:.4f} | Source: {src} | Snippet: {repr(c['text'][:80])}")
