import sys, os, time, uuid
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

from backend.rag import pipeline

queries = [
    ("Greeting ('hii')", "hii"),
    ("Course Direct ('bca')", "bca"),
    ("Fee Inquiry ('bca fee')", "bca fee"),
    ("Fee Inquiry BDA ('bda fee')", "bda fee"),
    ("Eligibility ('bca eligibility')", "bca eligibility"),
    ("Documents ('what documents')", "what documents"),
    ("Calendar ('cai 2 date')", "cai 2 date"),
    ("Curriculum ('bca subjects')", "bca subjects"),
    ("Hostel ('what about hostel')", "what about hostel"),
    ("Faculty ('who is david')", "who is david"),
    ("Contact ('college phone number')", "college phone number"),
    ("Office Timings ('office timings')", "office timings"),
    ("Language Switch ('mane Gujarati ma j answer aapo')", "mane Gujarati ma j answer aapo"),
    ("Sports ('sports facilities')", "sports facilities"),
    ("Events ('what is culfest')", "what is culfest"),
    ("Faculty Typo ('who is nisar vyash')", "who is nisar vyash"),
    ("Faculty Short ('fr.david')", "fr.david"),
    ("Leadership ('who is president of this clg')", "who is president of this clg"),
    ("Qualification Discovery ('mene bcom kiya hai konsa course hai')", "mene bcom kiya hai konsa course hai"),
    ("Negation Rejection ('bcom nahi karna hai')", "bcom nahi karna hai"),
]

def benchmark():
    print("==========================================================")
    print("SXCA ADMISSION ASSISTANT — LATENCY BENCHMARK")
    print("==========================================================\n")
    
    # Warmup
    cid_warmup = str(uuid.uuid4())
    _ = pipeline.answer("hii", [], conversation_id=cid_warmup)
    
    results = []
    for label, q in queries:
        cid = str(uuid.uuid4())
        start = time.perf_counter()
        res = pipeline.answer(q, [], conversation_id=cid)
        elapsed_ms = (time.perf_counter() - start) * 1000
        has_sources = len(res.get("sources", []))
        results.append((label, elapsed_ms, has_sources))
        print(f"{label:35} : {elapsed_ms:7.1f} ms  (sources: {has_sources})")

    avg_ms = sum(r[1] for r in results) / len(results)
    print("\n----------------------------------------------------------")
    print(f"AVERAGE LATENCY: {avg_ms:.1f} ms")
    print("==========================================================")

if __name__ == "__main__":
    benchmark()
