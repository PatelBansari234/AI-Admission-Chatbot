"""
Unit tests for query_normalizer.py verifying:
1. Percentage variants (65%, 65 percent, 65 persent, 65 prcnt, 65 marks out of 100)
2. Ambiguous numbers ("I got 65 in 12th")
3. Eligibility ("I got 65% in 12th can I take BCA?")
4. Typing mistakes ("plz tel me bca eligibilty", "what r the admn requirments?", "bca feees")
5. WhatsApp shorthand ("wht r the bca fees", "idk which course to take", "u tell me which is best fr AI")
6. Casual but substantive queries ("bro i wanna study AI which course is best for me?")
7. Gujarati and Hindi (native and Romanized)
8. Safety of normalization (protected entities, negation, faculty names)
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))

from backend.rag import query_normalizer

def test_percentages():
    print("\n--- TEST: Percentage Variants & Ambiguous Numbers ---")

    # Explicit percentage forms
    p1 = query_normalizer.extract_percentage_and_score("65%")
    assert p1["percentage"] == 65.0 and not p1["is_ambiguous"], f"Failed on 65%: {p1}"

    p2 = query_normalizer.extract_percentage_and_score("65 percent")
    assert p2["percentage"] == 65.0 and not p2["is_ambiguous"], f"Failed on 65 percent: {p2}"

    p3 = query_normalizer.extract_percentage_and_score("65 persent")
    assert p3["percentage"] == 65.0 and not p3["is_ambiguous"], f"Failed on 65 persent: {p3}"

    p4 = query_normalizer.extract_percentage_and_score("65 prcnt")
    assert p4["percentage"] == 65.0 and not p4["is_ambiguous"], f"Failed on 65 prcnt: {p4}"

    p5 = query_normalizer.extract_percentage_and_score("I scored 65 marks out of 100")
    assert p5["percentage"] == 65.0 and not p5["is_ambiguous"] and p5["marks_out_of_100"], f"Failed on 65 marks out of 100: {p5}"

    p6 = query_normalizer.extract_percentage_and_score("12th ma 65% che")
    assert p6["percentage"] == 65.0, f"Failed on 12th ma 65% che: {p6}"

    print("  [PASS] All explicit percentage variants extracted accurately as 65.0%.")

    # Ambiguous forms - MUST NOT become 65% automatically
    a1 = query_normalizer.extract_percentage_and_score("I got 65 in 12th")
    assert a1["percentage"] is None and a1["is_ambiguous"] and a1["ambiguous_score"] == 65.0, f"Failed on ambiguous score: {a1}"

    a2 = query_normalizer.extract_percentage_and_score("12th ma 65 che")
    assert a2["percentage"] is None and a2["is_ambiguous"] and a2["ambiguous_score"] == 65.0, f"Failed on ambiguous score: {a2}"

    print("  [PASS] Ambiguous scores flagged as is_ambiguous=True without forced conversion to percentage.")


def test_shorthand_and_typos():
    print("\n--- TEST: WhatsApp Shorthand & Domain Typos ---")

    cases = [
        ("plz tel me bca eligibilty", "please tell me bca eligibility"),
        ("what r the admn requirments?", "what are the admission requirement?"),
        ("bca feees", "bca fees"),
        ("wht r the bca fees", "what are the bca fees"),
        ("idk which course to take", "I don't know which course to take"),
        ("u tell me which is best fr AI", "you tell me which is best for AI"),
        ("bro i wanna study AI which course is best for me?", "I want to study AI which course is best for me?"),
        ("bca me admn mil skta h?", "bca me admission mil sakta hai?"),
        ("clg feeees", "college fees"),
    ]

    for raw, expected in cases:
        norm, _ = query_normalizer.normalize_query_text(raw)
        assert norm.lower() == expected.lower(), f"Failed on '{raw}': got '{norm}', expected '{expected}'"
        print(f"  [PASS] '{raw}' -> '{norm}'")


def test_safety_and_preservation():
    print("\n--- TEST: Safety & Protected Entities ---")

    # Protected faculty names must never be altered
    faculty_queries = [
        "Who is Nisarg Vyas?",
        "Tell me about Nisarg Patel",
        "Who is Dr. Pravida Raja?",
        "Contact Dr. Sebastian V.A.",
    ]
    for fq in faculty_queries:
        norm, _ = query_normalizer.normalize_query_text(fq)
        assert "nisarg vyas" in norm.lower() or "nisarg patel" in norm.lower() or "pravida raja" in norm.lower() or "sebastian" in norm.lower(), f"Corrupted faculty: {norm}"
        print(f"  [PASS] Preserved faculty: '{fq}' -> '{norm}'")

    # Negation words must not be altered
    neg_queries = [
        "I do not want biology",
        "Is hostel free? no",
        "college ma swimming pool nathi",
    ]
    for nq in neg_queries:
        norm, _ = query_normalizer.normalize_query_text(nq)
        assert "not" in norm.lower() or "no" in norm.lower() or "nathi" in norm.lower(), f"Corrupted negation: {norm}"
        print(f"  [PASS] Preserved negation: '{nq}' -> '{norm}'")

    # Paper codes and numbers must not be altered
    code_queries = [
        "What is PAI-1803 syllabus?",
        "Tell me about CAI-2 exam date",
    ]
    for cq in code_queries:
        norm, _ = query_normalizer.normalize_query_text(cq)
        assert "pai-1803" in norm.lower() or "cai-2" in norm.lower(), f"Corrupted code: {norm}"
        print(f"  [PASS] Preserved code: '{cq}' -> '{norm}'")


def test_eligibility_response():
    print("\n--- TEST: Percentage-based Eligibility Evaluation ---")

    # Case 1: Explicit 65% in 12th for BCA
    q1 = "I got 65% in 12th can I take BCA?"
    parsed1 = query_normalizer.parse_and_normalize(q1)
    assert parsed1.extracted_percentage == 65.0
    assert parsed1.extracted_course == "BCA"
    res1 = query_normalizer.build_percentage_eligibility_response(parsed1)
    assert "12th Science & Commerce" in res1["answer"]
    assert "65" in res1["answer"]
    assert "eligible to apply" in res1["answer"]
    assert len(res1["sources"]) > 0
    print(f"  [PASS] Explicit BCA eligibility check correctly returns official requirements & merit context.")

    # Case 2: Ambiguous score "I got 65 in 12th" for BCA
    q2 = "I got 65 in 12th can I take BCA?"
    parsed2 = query_normalizer.parse_and_normalize(q2)
    assert parsed2.is_ambiguous_score
    assert parsed2.ambiguous_score_val == 65.0
    res2 = query_normalizer.build_percentage_eligibility_response(parsed2)
    assert res2.get("status") == "CLARIFICATION_REQUIRED"
    assert "clarify whether" in res2["answer"]
    assert "65" in res2["answer"]
    print(f"  [PASS] Ambiguous score correctly asks clarifying question without assuming percentage.")


if __name__ == "__main__":
    test_percentages()
    test_shorthand_and_typos()
    test_safety_and_preservation()
    test_eligibility_response()
    print("\n========================================================")
    print("ALL QUERY NORMALIZER UNIT TESTS PASSED!")
    print("========================================================\n")
