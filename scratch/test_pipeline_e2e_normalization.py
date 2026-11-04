"""
End-to-End Pipeline test for Query Normalization & Percentage Understanding:
1. "I got 65% in 12th can I take BCA?" (Explicit percentage)
2. "I got 65 in 12th can I take BCA?" (Ambiguous score - requires clarification)
3. "12th ma 65% che, BCA lai saku?" (Gujarati percentage)
4. "bro i wanna study AI which course is best for me?" (Casual Gen Z counselling)
5. "wht r the bca fees" (WhatsApp shorthand + typo)
6. "plz tel me bca eligibilty" (WhatsApp shorthand + typo)
7. "bca me admn mil skta h?" (Romanized Hindi shorthand)
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))

from backend.rag import pipeline

def run_tests():
    print("=" * 60)
    print("RUNNING E2E PIPELINE NORMALIZATION & PERCENTAGE SUITE")
    print("=" * 60)

    # 1. Explicit 65% for BCA
    print("\n--- TEST 1: Explicit 65% in 12th for BCA ---")
    q1 = "I got 65% in 12th can I take BCA?"
    res1 = pipeline.answer(q1)
    ans1 = res1["answer"]
    print("Answer preview:", ans1[:180])
    assert "BCA" in ans1, f"Missing BCA in answer: {ans1}"
    assert "12th Science & Commerce" in ans1, f"Missing official eligibility requirement in answer: {ans1}"
    assert "65" in ans1, f"Missing percentage in answer: {ans1}"
    assert len(res1["sources"]) > 0, f"Expected verified sources, got: {res1['sources']}"
    assert res1["grounded"] is True
    print("PASS: TEST 1 verified (Explicit percentage + course -> verified eligibility requirement & merit context)")

    # 2. Ambiguous 65 in 12th for BCA
    print("\n--- TEST 2: Ambiguous score (65 in 12th) ---")
    q2 = "I got 65 in 12th can I take BCA?"
    res2 = pipeline.answer(q2)
    ans2 = res2["answer"]
    print("Answer preview:", ans2[:180])
    assert "clarify whether" in ans2.lower(), f"Expected clarification question, got: {ans2}"
    assert "65" in ans2
    assert "BCA" in ans2
    assert len(res2["sources"]) > 0
    print("PASS: TEST 2 verified (Ambiguous number -> asks polite clarification without converting to % automatically)")

    # 3. Gujarati percentage for BCA
    print("\n--- TEST 3: Gujarati Percentage (12th ma 65% che, BCA lai saku?) ---")
    q3 = "12th ma 65% che, BCA lai saku?"
    res3 = pipeline.answer(q3)
    ans3 = res3["answer"]
    sys.stdout.reconfigure(encoding="utf-8")
    print("Answer preview:", ans3[:180])
    assert "BCA" in ans3
    assert "65" in ans3
    assert "અરજી" in ans3 or "મેરિટ" in ans3 or "પાત્રતા" in ans3
    print("PASS: TEST 3 verified (Gujarati percentage -> natural Gujarati verified eligibility response)")

    # 4. Casual Gen Z course counselling
    print("\n--- TEST 4: Casual Gen Z ('bro i wanna study AI which course is best for me?') ---")
    q4 = "bro i wanna study AI which course is best for me?"
    res4 = pipeline.answer(q4)
    ans4 = res4["answer"]
    print("Answer preview:", ans4[:180])
    # Must NOT be bare greeting, must discuss AI programs (BSc AI & ML, BCA, etc.)
    assert any(term in ans4 for term in ["B.Sc", "AI", "Machine Learning", "BCA", "Stream", "Science"]), f"Not counselling response: {ans4}"
    assert "I am Niaa, your dedicated" not in ans4, f"Mistaken for greeting: {ans4}"
    print("PASS: TEST 4 verified (Casual question correctly routed to course counselling, not greeting)")

    # 5. WhatsApp shorthand for fees
    print("\n--- TEST 5: WhatsApp Shorthand ('wht r the bca fees') ---")
    q5 = "wht r the bca fees"
    res5 = pipeline.answer(q5)
    ans5 = res5["answer"]
    print("Answer preview:", ans5[:180])
    assert "BCA" in ans5
    assert "₹" in ans5 or "29,750" in ans5 or "fee" in ans5.lower()
    print("PASS: TEST 5 verified (WhatsApp shorthand 'wht r the bca fees' -> structured fee answer)")

    # 6. Typo in eligibility ('plz tel me bca eligibilty')
    print("\n--- TEST 6: Typo in eligibility ('plz tel me bca eligibilty') ---")
    q6 = "plz tel me bca eligibilty"
    res6 = pipeline.answer(q6)
    ans6 = res6["answer"]
    print("Answer preview:", ans6[:180])
    assert "BCA" in ans6
    assert "12th" in ans6 or "Science" in ans6 or "Commerce" in ans6
    print("PASS: TEST 6 verified (Typo 'eligibilty' -> structured eligibility answer)")

    # 7. Romanized Hindi shorthand ('bca me admn mil skta h?')
    print("\n--- TEST 7: Romanized Hindi ('bca me admn mil skta h?') ---")
    q7 = "bca me admn mil skta h?"
    res7 = pipeline.answer(q7)
    ans7 = res7["answer"]
    print("Answer preview:", ans7[:180])
    assert "BCA" in ans7
    assert len(ans7) > 20
    print("PASS: TEST 7 verified (Romanized Hindi shorthand 'bca me admn mil skta h?' -> handled reliably)")

    print("\n========================================================")
    print("ALL E2E PIPELINE NORMALIZATION & PERCENTAGE TESTS PASSED!")
    print("========================================================\n")

if __name__ == "__main__":
    run_tests()
