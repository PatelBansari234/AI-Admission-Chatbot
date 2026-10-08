import sys
import os
sys.path.insert(0, os.getcwd())

from backend.rag import counsellor, query_rewrite, pipeline

def test_followup_resolution():
    print("=== TEST 1: Resolved entity + pronoun teaching inquiry ===")
    # Turn 1: "Who is Nisarg Vyas?"
    res1_t1 = pipeline.answer("Who is Nisarg Vyas?")
    print("Turn 1 Status:", res1_t1.get("status"))
    assert res1_t1.get("status") == "RESOLVED"
    assert "Nisarg Vyas" in res1_t1["answer"]

    # Turn 2: "What does he teach?"
    history1 = [
        {"role": "user", "content": "Who is Nisarg Vyas?"},
        {"role": "assistant", "content": res1_t1["answer"]}
    ]
    res1_t2 = pipeline.answer("What does he teach?", history=history1)
    print("Turn 2 Answer:\n", res1_t2["answer"])
    assert res1_t2.get("status") == "RESOLVED"
    assert "Nisarg Vyas" in res1_t2["answer"]
    assert any(term in res1_t2["answer"] for term in ["teaches", "programmes", "Data Science", "Artificial Intelligence", "BCA"])
    print("PASS: TEST 1 verified ('he' resolved to Nisarg Vyas teaching)!\n")

    print("=== TEST 2: Ambiguous entity + pronoun teaching inquiry ===")
    # Turn 1: "Who is Nisarg?"
    res2_t1 = pipeline.answer("Who is Nisarg?")
    print("Turn 1 Status:", res2_t1.get("status"))
    assert res2_t1.get("status") == "AMBIGUOUS"

    # Turn 2: "What does he teach?"
    history2 = [
        {"role": "user", "content": "Who is Nisarg?"},
        {"role": "assistant", "content": res2_t1["answer"]}
    ]
    res2_t2 = pipeline.answer("What does he teach?", history=history2)
    print("Turn 2 Answer:\n", res2_t2["answer"])
    assert res2_t2.get("status") == "AMBIGUOUS"
    # MUST NOT select Nisarg Vyas or Coordinator of AI and ML
    assert "Coordinator of AI and ML" not in res2_t2["answer"]
    assert "specify which" in res2_t2["answer"].lower() or "multiple people" in res2_t2["answer"].lower()
    print("PASS: TEST 2 verified (ambiguity preserved, Nisarg Vyas not selected)!\n")

    print("=== TEST 3: Resolved entity + email inquiry ===")
    # Turn 1: "Who is Nisarg Vyas?"
    # Turn 2: "What is his email?"
    history3 = [
        {"role": "user", "content": "Who is Nisarg Vyas?"},
        {"role": "assistant", "content": res1_t1["answer"]}
    ]
    res3_t2 = pipeline.answer("What is his email?", history=history3)
    print("Turn 2 Answer:\n", res3_t2["answer"])
    assert res3_t2.get("status") == "RESOLVED"
    assert "Nisarg Vyas" in res3_t2["answer"]
    assert "nisarg.vyas@sxca.edu.in" in res3_t2["answer"]
    print("PASS: TEST 3 verified ('his' resolved to Nisarg Vyas email)!\n")

    print("=== TEST 4: Resolved entity + department inquiry ===")
    # Turn 1: "Who is Nisarg Vyas?"
    # Turn 2: "What department is he from?"
    history4 = [
        {"role": "user", "content": "Who is Nisarg Vyas?"},
        {"role": "assistant", "content": res1_t1["answer"]}
    ]
    res4_t2 = pipeline.answer("What department is he from?", history=history4)
    print("Turn 2 Answer:\n", res4_t2["answer"])
    assert res4_t2.get("status") == "RESOLVED"
    assert "Nisarg Vyas" in res4_t2["answer"]
    assert "Department of Data Science" in res4_t2["answer"] or "Computer Sciences" in res4_t2["answer"]
    print("PASS: TEST 4 verified ('he' resolved to Nisarg Vyas department)!\n")

    print("=== TEST 5: Ambiguous entity + 'Tell me about him' ===")
    # Turn 1: "Who is Nisarg?"
    # Turn 2: "Tell me about him."
    history5 = [
        {"role": "user", "content": "Who is Nisarg?"},
        {"role": "assistant", "content": res2_t1["answer"]}
    ]
    res5_t2 = pipeline.answer("Tell me about him.", history=history5)
    print("Turn 2 Answer:\n", res5_t2["answer"])
    assert res5_t2.get("status") == "AMBIGUOUS"
    assert "Coordinator of AI and ML" not in res5_t2["answer"]
    assert "specify which" in res5_t2["answer"].lower() or "multiple people" in res5_t2["answer"].lower()
    print("PASS: TEST 5 verified (clarification requested because first entity was ambiguous)!\n")

    print("=== TEST 6: Entity switch - 'Tell me about another faculty member' ===")
    # Turn 1: "Who is Nisarg Vyas?"
    # Turn 2: "Tell me about another faculty member."
    history6 = [
        {"role": "user", "content": "Who is Nisarg Vyas?"},
        {"role": "assistant", "content": res1_t1["answer"]}
    ]
    res6_t2 = pipeline.answer("Tell me about another faculty member.", history=history6)
    print("Turn 2 Answer:\n", res6_t2["answer"])
    # Must NOT continue Nisarg Vyas
    assert "Assistant Professor & Coordinator of AI and ML" not in res6_t2["answer"]
    assert "faculty" in res6_t2["answer"].lower() or "department" in res6_t2["answer"].lower()
    print("PASS: TEST 6 verified (cleared Nisarg Vyas on request for another faculty member)!\n")

    print("ALL 6 FOLLOW-UP RESOLUTION TESTS PASSED!")

if __name__ == "__main__":
    test_followup_resolution()
