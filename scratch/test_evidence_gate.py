import sys
import os
import traceback

sys.path.insert(0, os.getcwd())

from backend.rag import counsellor, pipeline, query_rewrite
from backend.llm import qwen
from backend.rag import cache

EXACT_NO_INFO = "Sorry, I don't have information about that in the available SXCA materials."

def run_test(test_name, question, history=None):
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

def test_evidence_gate():
    print("=" * 60)
    print("RUNNING EVIDENCE GATE & ANTI-HALLUCINATION SUITE")
    print("=" * 60)

    # TEST 1: No evidence / completely ungrounded query (e.g., "Tell me about aerospace engineering at SXCA")
    print("\n--- TEST 1: No evidence / completely ungrounded query ---")
    q1 = "Tell me about aerospace engineering at SXCA"
    res1, qwen_called1 = run_test("TEST 1", q1)
    print(f"Question: {q1}")
    print(f"Answer: {res1['answer']}")
    print(f"Qwen called: {qwen_called1}")
    print(f"Grounded: {res1.get('grounded')}")
    assert res1["answer"] == EXACT_NO_INFO, f"Expected exact no-info string, got: {res1['answer']}"
    assert not qwen_called1, "Qwen must NOT be called for ungrounded query, but was called!"
    assert res1["grounded"] is False, "Expected grounded=False"
    print("PASS: TEST 1 verified (No evidence -> Exact no-info string, Qwen called = False)")


    # TEST 2: Completely unknown entity (e.g., "Who is John Doe at SXCA?")
    print("\n--- TEST 2: Completely unknown entity ---")
    q2 = "Who is John Doe at SXCA?"
    res2, qwen_called2 = run_test("TEST 2", q2)
    print(f"Question: {q2}")
    print(f"Answer: {res2['answer']}")
    print(f"Qwen called: {qwen_called2}")
    print(f"Status: {res2.get('status')}")
    assert res2["answer"] == EXACT_NO_INFO, f"Expected exact no-info string, got: {res2['answer']}"
    assert not qwen_called2, "Qwen must NOT be called for unknown entity, but was called!"
    assert res2.get("status") == "NOT_FOUND" or res2.get("grounded") is False
    print("PASS: TEST 2 verified (Unknown entity -> Exact no-info string, Qwen called = False)")

    # TEST 3: Ambiguous entity ("Who is Nisarg?")
    print("\n--- TEST 3: Ambiguous entity ---")
    q3 = "Who is Nisarg?"
    res3, qwen_called3 = run_test("TEST 3", q3)
    print(f"Question: {q3}")
    print(f"Answer: {res3['answer']}")
    print(f"Qwen called: {qwen_called3}")
    print(f"Status: {res3.get('status')}")
    assert res3.get("status") == "AMBIGUOUS"
    assert "multiple people" in res3["answer"].lower() or "specify the full name" in res3["answer"].lower()
    assert "Nisarg Vyas" in res3["answer"] and "Nisarg Patel" in res3["answer"]
    assert "Coordinator of AI and ML" not in res3["answer"]
    assert not qwen_called3, "Qwen must NOT be called for ambiguous entity, but was called!"
    print("PASS: TEST 3 verified (Ambiguous entity -> Clarification with candidates, Qwen called = False)")

    # TEST 4: Resolved entity but missing requested attribute
    print("\n--- TEST 4: Resolved entity but missing requested attribute ---")
    q4a = "What is Nisarg Vyas's personal mobile phone number?"
    res4a, qwen_called4a = run_test("TEST 4A", q4a)
    print(f"Question: {q4a}")
    print(f"Answer: {res4a['answer']}")
    print(f"Qwen called: {qwen_called4a}")
    assert res4a["answer"] == EXACT_NO_INFO, f"Expected exact no-info string, got: {res4a['answer']}"
    assert not qwen_called4a, "Qwen must NOT be called for missing phone attribute!"
    print("PASS: TEST 4 verified (Missing attribute -> No-info string, Qwen called = False, no hallucination)")

    # TEST 5: Valid grounded query
    print("\n--- TEST 5: Valid grounded query ---")
    q5 = "What is the history of St. Xavier's College Ahmedabad?"
    res5, qwen_called5 = run_test("TEST 5", q5)
    print(f"Question: {q5}")
    print(f"Answer preview: {res5['answer'][:180]}...")
    print(f"Qwen called: {qwen_called5}")
    print(f"Grounded: {res5.get('grounded')}")
    print(f"Sources count: {len(res5.get('sources', []))}")
    assert qwen_called5, "Qwen MUST be called for valid grounded query!"
    assert res5.get("grounded") is True, "Expected grounded=True"
    assert len(res5.get("sources", [])) > 0, "Expected sources to be provided"
    assert any(term in res5["answer"].lower() for term in ["xavier", "1955", "college", "arts", "campus"])
    print("PASS: TEST 5 verified (Valid grounded query -> Correct answer from approved source, Qwen called = True)")

    print("\n" + "=" * 60)
    print("ALL 5 EVIDENCE GATE TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    test_evidence_gate()
