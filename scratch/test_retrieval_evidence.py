import sys
import os
import traceback

sys.path.insert(0, os.getcwd())

from backend.rag import counsellor, query_rewrite, pipeline
from backend.llm import qwen
from backend.rag import cache

EXACT_NO_INFO = "Sorry, I don't have information about that in the available SXCA materials."

def run_tracked(question, history=None):
    cache.clear()
    calls = []
    original_generate = qwen.generate
    original_stream = qwen.stream

    def tracked_generate(*args, **kwargs):
        calls.append("generate")
        return original_generate(*args, **kwargs)

    def tracked_stream(*args, **kwargs):
        calls.append("stream")
        return original_stream(*args, **kwargs)

    qwen.generate = tracked_generate
    qwen.stream = tracked_stream
    try:
        ans = pipeline.answer(question, history=history)
    finally:
        qwen.generate = original_generate
        qwen.stream = original_stream

    return ans, len(calls) > 0

def test_retrieval_evidence():
    print("=" * 60)
    print("RUNNING RETRIEVAL & EVIDENCE PRESERVATION TESTS")
    print("=" * 60)

    # TEST 1: "What is the hostel fee at SXCA?"
    print("\n--- TEST 1: Valid structured query with lower initial rerank score ---")
    q1 = "What is the hostel fee at SXCA?"
    res1, qwen_called1 = run_tracked(q1)
    ans1 = res1.get("answer", "")
    print(f"Question: {q1}")
    print(f"Answer preview: {ans1[:180]}...")
    print(f"Qwen called: {qwen_called1}")
    print(f"Grounded: {res1.get('grounded')}")
    print(f"Sources: {[s.get('source') for s in res1.get('sources', [])]}")

    assert qwen_called1, "Expected Qwen to be called when valid evidence is retrieved!"
    assert res1.get("grounded") is True, "Expected grounded=True"
    assert any("admissions-accommodation" in s.get("source", "") for s in res1.get("sources", [])), "Expected admissions-accommodation in sources"
    # Ensure real numbers from source appear, and no fabricated values
    assert any(term in ans1 for term in ["10,000", "20,000", "41,050", "3,000", "77,925"]), "Expected official hostel fee amounts"
    assert "15,000 – 25,000" not in ans1 and "15,000 to 25,000" not in ans1, "Fabicated rent estimate must not appear"
    print("PASS: TEST 1 verified (Hostel fee evidence retained, gate accepted, Qwen answered with factual data)")

    # TEST 2: Unrelated question
    print("\n--- TEST 2: Unrelated question ---")
    q2 = "Tell me about aerospace engineering at SXCA"
    res2, qwen_called2 = run_tracked(q2)

    print(f"Question: {q2}")
    print(f"Answer: {res2.get('answer')}")
    print(f"Qwen called: {qwen_called2}")
    print(f"Grounded: {res2.get('grounded')}")

    assert res2.get("answer") == EXACT_NO_INFO, f"Expected exact no-info string, got {res2.get('answer')}"
    assert not qwen_called2, "Qwen must NOT be called for ungrounded query"
    print("PASS: TEST 2 verified (Unrelated query -> NO_INFORMATION, Qwen not called)")

    # TEST 3: "Who is Nisarg?"
    print("\n--- TEST 3: Ambiguous entity ---")
    q3 = "Who is Nisarg?"
    res3, qwen_called3 = run_tracked(q3)
    print(f"Question: {q3}")
    print(f"Answer: {res3.get('answer')}")
    print(f"Status: {res3.get('status')}")
    print(f"Qwen called: {qwen_called3}")

    assert res3.get("status") == "AMBIGUOUS"
    assert "multiple people" in res3["answer"].lower() or "specify the full name" in res3["answer"].lower()
    assert "Nisarg Vyas" in res3["answer"] and "Nisarg Patel" in res3["answer"]
    assert not qwen_called3, "Qwen must not be called for ambiguous entity"
    print("PASS: TEST 3 verified (Ambiguous entity -> clarification with candidates list)")

    # TEST 4: "What is Nisarg Vyas's email?"
    print("\n--- TEST 4: Resolved entity email ---")
    q4 = "What is Nisarg Vyas's email?"
    res4, qwen_called4 = run_tracked(q4)
    print(f"Question: {q4}")
    print(f"Answer: {res4.get('answer')}")
    print(f"Qwen called: {qwen_called4}")

    assert "nisarg.vyas@sxca.edu.in" in res4.get("answer", "")
    assert not qwen_called4, "Qwen not called (resolved faculty fast path)"
    print("PASS: TEST 4 verified (Resolved faculty email returned accurately)")

    print("\n" + "=" * 60)
    print("ALL 4 RETRIEVAL & EVIDENCE TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    test_retrieval_evidence()
