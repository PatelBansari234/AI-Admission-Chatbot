import sys
import os

sys.path.insert(0, os.getcwd())

from backend.rag import counsellor, query_rewrite, pipeline
from backend.rag import retriever
from backend.rag.prompt import build_messages
from backend.llm import qwen

q = "What is the hostel fee at SXCA?"

# Test our proposed retrieve logic
distinctive_tokens = retriever._tokenize(q)
stop = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "the", "a", "an", "and", "or", "but", "if", "then",
    "this", "that", "these", "those", "for", "with", "about", "to", "from",
    "in", "out", "on", "at", "by", "sxca", "college", "st", "xavier", "xaviers",
    "xavier's", "ahmedabad", "tell", "give", "info", "information", "please", "me",
    "can", "could", "would", "should", "there", "any", "much", "many"
}
distinctive_tokens = [t for t in distinctive_tokens if len(t) >= 3 and t not in stop]
print(f"Distinctive query tokens: {distinctive_tokens}")

cands = retriever._candidates(retriever.normalize_query(q))
pairs = [[retriever.normalize_query(q), c["text"]] for c in cands]
scores = retriever._rerank_predict(pairs)

for c, s in zip(cands, scores):
    c["boosted_score"] = float(s) * retriever.source_priority.boost_for(c["metadata"])

cands.sort(key=lambda c: c["boosted_score"], reverse=True)
top_score = cands[0]["boosted_score"]
margin_cutoff = top_score * retriever.RERANK_RELATIVE_MARGIN

retained = []
for i, c in enumerate(cands):
    score = c["boosted_score"]
    if score < retriever.RERANK_FLOOR:
        continue
    if i == 0:
        retained.append(c)
        continue
    # Check alignment
    p_tokens = set(retriever._tokenize(c["text"]))
    matched = 0
    for dt in distinctive_tokens:
        base = dt.rstrip('s')
        if any(pt == dt or (len(base) >= 3 and pt.rstrip('s') == base) for pt in p_tokens):
            matched += 1
    has_strong_alignment = (matched == len(distinctive_tokens)) if len(distinctive_tokens) <= 2 else (matched >= len(distinctive_tokens) * 0.75)
    
    if has_strong_alignment:
        retained.append(c)
    elif score >= margin_cutoff:
        retained.append(c)

passages = retained[:5]
print(f"Retained passages count: {len(passages)}")
for i, p in enumerate(passages):
    src = p["metadata"].get("source")
    print(f"[{i+1}] Score: {p['boosted_score']:.4f} | Source: {src}")

understanding = {"intent": "general", "course": None, "qualification": None}
is_valid, reason = pipeline.validate_evidence(q, passages, [], understanding)
print(f"validate_evidence: is_valid={is_valid}, reason={reason}")

if is_valid:
    print("Generating answer with Qwen...")
    messages = build_messages(q, passages, [], [])
    ans = qwen.generate(messages)
    print("\n--- QWEN GENERATED ANSWER ---")
    print(ans.encode('ascii', errors='backslashreplace').decode('ascii'))
