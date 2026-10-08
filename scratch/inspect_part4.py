import sys
import os
sys.path.insert(0, os.getcwd())

from backend.rag import query_rewrite, counsellor, pipeline, retriever

questions = [
    "What is the hostel fee at SXCA?",
    "Who is John Doe at SXCA?",
    "Who is Nisarg?",
    "What is Nisarg Vyas's personal mobile phone number?",
]

for q in questions:
    print(f"\n====================\nQ: {q}")
    u = query_rewrite.understand(q, [], None)
    print("Intent:", u.get("intent"))
    print("Course:", u.get("course"))
    print("Rewritten query:", u.get("query"))
    
    if u.get("intent") == "faculty":
        res_fac = counsellor.resolve_faculty(q)
        print("Faculty resolution status:", res_fac.get("status"))
        if res_fac.get("status") == "AMBIGUOUS":
            print("Candidates:", [c["canonical_name"] for c in res_fac.get("candidates", [])])
        elif res_fac.get("status") == "RESOLVED":
            print("Resolved person:", res_fac.get("person", {}).get("canonical_name"))
    elif u.get("intent") == "hostel":
        print("Hostel intent handled by counsellor.handle_hostel")
        h_text = counsellor.handle_hostel(q)
        print("Hostel handler output:\n", h_text)
    else:
        passages = retriever.retrieve(u.get("query"))
        print(f"Retrieved passages: {len(passages)}")
        for p in passages:
            print(f"  Score: {p.get('boosted_score'):.3f} | Source: {p['metadata'].get('source')} | Text: {p['text'][:90]}...")
