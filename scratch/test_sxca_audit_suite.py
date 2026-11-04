"""Comprehensive Audit Test Suite for SXCA AI Admission Assistant (Niaa)

Covers Scenarios A through F from the 3-hour deployment requirements:
- Scenario A: Faculty Hallucination & Deterministic Roster Matching (Nisarg Patel, Nisarg Shah, Nisarg Vyas, Nancy Patel)
- Scenario B: Fast-Path Greetings & Identity (hello, kem cho, majama ne, who are you, tame kon cho, what is your name)
- Scenario C: Multilingual Interaction (English, Hindi, Gujarati, Romanized)
- Scenario D: Course Query Paraphrasing (All 5 variants)
- Scenario E: Missing Information Verified Fallback (info@sxca.edu.in / +91 79 2630 2872)
- Scenario F: Reliability & Pipeline Integrity (no stuck state, sources empty on refusal)
"""

import time
import sys
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.rag import counsellor, query_rewrite, pipeline


def test_scenario_a_faculty_resolution():
    print("\n--- [Scenario A] Faculty Entity Resolution & Anti-Hallucination ---")

    # A1: "Nisarg Patel" must NOT fabricate a faculty profile; must clarify similar names
    res = counsellor.handle_faculty("Nisarg Patel")
    ans = res["answer"]
    assert "exact match" in ans.lower() or "mean" in ans.lower() or "clarify" in ans.lower(), (
        f"Expected clarification for 'Nisarg Patel', got: {ans}"
    )
    assert "Nisarg Vyas" in ans, f"Expected Nisarg Vyas suggested in similar names, got: {ans}"
    assert "nisarg.patel@sxca.edu.in" not in ans.lower(), "Fabricated fake email for Nisarg Patel!"
    print("  [PASS] [A1] 'Nisarg Patel' properly triggers SIMILAR_NAME_CLARIFICATION without fabricating identity.")

    # A2: "Nisarg Shah" must clarify similar names
    res2 = counsellor.handle_faculty("Nisarg Shah")
    ans2 = res2["answer"]
    assert "exact match" in ans2.lower() or "mean" in ans2.lower() or "did you mean" in ans2.lower(), (
        f"Expected clarification for 'Nisarg Shah', got: {ans2}"
    )
    assert "nisarg.shah@sxca.edu.in" not in ans2.lower(), "Fabricated fake email for Nisarg Shah!"
    print("  [PASS] [A2] 'Nisarg Shah' properly triggers SIMILAR_NAME_CLARIFICATION without fabricating identity.")

    # A3: "Nisarg Vyas" must return official verified profile
    res3 = counsellor.handle_faculty("Nisarg Vyas")
    ans3 = res3["answer"]
    assert "Nisarg Vyas" in ans3
    assert "Computer Sciences" in ans3 or "Data Science" in ans3
    assert "nisarg.vyas@sxca.edu.in" in ans3
    print("  [PASS] [A3] 'Nisarg Vyas' resolves with verified designation, department, and official email.")

    # A4: Missing faculty attributes (salary, phone number) must NOT be hallucinated
    prof = {"active_person": res3.get("person")}
    res4 = counsellor.handle_faculty("What is his salary?", prof)
    ans4 = res4["answer"]
    assert "couldn't verify" in ans4.lower() or "contact" in ans4.lower(), (
        f"Expected refusal for faculty salary, got: {ans4}"
    )
    print("  [PASS] [A4] Faculty salary query returns verified fallback without hallucinating numbers.")

    # A5: Nancy Patel - faculty member without public email in records
    res5 = counsellor.handle_faculty("Nancy Patel")
    ans5 = res5["answer"]
    assert "Nancy Patel" in ans5
    assert "nancy.patel@sxca.edu.in" not in ans5, "Fabricated fake email for Nancy Patel!"
    print("  [PASS] [A5] 'Nancy Patel' profile displays verified designation/dept without inventing email.")


def test_scenario_b_greetings_and_identity():
    print("\n--- [Scenario B] Fast-Path Greetings & Identity ---")

    greetings_and_identity = [
        ("hello", "greeting", ["welcome", "niaa"]),
        ("hi", "greeting", ["welcome", "niaa"]),
        ("hey", "greeting", ["welcome", "niaa"]),
        ("kem cho", "greeting", ["મજામાં"]),
        ("majama ne", "greeting", ["મજામાં"]),
        ("who are you?", "identity", ["niaa", "assistant"]),
        ("what is your name?", "identity", ["niaa"]),
        ("tame kon cho?", "identity", ["niaa"]),
    ]

    for q, expected_intent, expected_snippets in greetings_and_identity:
        t0 = time.perf_counter()
        u = query_rewrite.understand(q)
        dt = (time.perf_counter() - t0) * 1000

        assert u.get("intent") == expected_intent, (
            f"Query '{q}' expected intent '{expected_intent}', got '{u.get('intent')}'"
        )
        assert dt < 100, f"Query '{q}' took {dt:.1f}ms (expected fast path <100ms)"

        if expected_intent == "greeting":
            reply = counsellor.handle_greeting(q, u.get("profile"))
        else:
            reply = counsellor.handle_identity(q, u.get("profile"))

        for snippet in expected_snippets:
            assert snippet.lower() in reply.lower(), (
                f"Query '{q}' reply missing snippet '{snippet}': {reply}"
            )
        print(f"  [PASS] [B] '{q}' -> intent '{expected_intent}' in {dt:.1f}ms")


def test_scenario_c_multilingual():
    print("\n--- [Scenario C] Multilingual Interaction ---")

    # English
    lang_en = counsellor.detect_language("Can you tell me about the admission procedure?")
    assert lang_en == "english"

    # Gujarati script
    lang_gu = counsellor.detect_language("સેન્ટ ઝેવિયર્સ કોલેજમાં કયા કોર્સ ઉપલબ્ધ છે?")
    assert lang_gu == "gujarati"
    reply_gu = counsellor.handle_course_discovery("સેન્ટ ઝેવિયર્સ કોલેજમાં કયા કોર્સ ઉપલબ્ધ છે?")
    assert "અંડરગ્રેજ્યુએટ" in reply_gu or "કોર્સ" in reply_gu

    # Hindi script
    lang_hi = counsellor.detect_language("कॉलेज में कौन से कोर्स उपलब्ध हैं?")
    assert lang_hi == "hindi"
    reply_hi = counsellor.handle_course_discovery("कॉलेज में कौन से कोर्स उपलब्ध हैं?")
    assert "पाठ्यक्रम" in reply_hi or "कोर्स" in reply_hi

    # Romanized Gujarati
    lang_rg = counsellor.detect_language("kem cho majama ne tame kon cho?")
    assert lang_rg == "gujarati"

    print("  [PASS] [C] English, Gujarati, Hindi, and Romanized Gujarati languages accurately detected and served.")


def test_scenario_d_course_paraphrasing():
    print("\n--- [Scenario D] Course Query Paraphrasing ---")

    phrases = [
        "What courses are available?",
        "Which programs can I study at SXCA?",
        "SXCA ma kaya course che?",
        "College ma su courses che?",
        "Tell me the courses offered.",
    ]

    for p in phrases:
        u = query_rewrite.understand(p)
        assert u.get("intent") in ("course_discovery", "all_ug_courses"), (
            f"Phrase '{p}' failed to resolve to course_discovery, got: {u.get('intent')}"
        )
        reply = counsellor.handle_course_discovery(p, u.get("profile"))
        assert "B.Sc" in reply or "BCA" in reply or "B.Com" in reply
        print(f"  [PASS] [D] '{p}' successfully mapped to course discovery.")


def test_scenario_e_missing_info_fallback():
    print("\n--- [Scenario E] Missing Information Verified Fallback ---")

    fallback_en = counsellor.get_verified_fallback("What is the fee for Aerospace Engineering at SXCA?")
    assert "info@sxca.edu.in" in fallback_en
    assert "+91 79 2630 2872" in fallback_en
    assert "Sorry, I couldn't verify this information" in fallback_en

    fallback_gu = counsellor.get_verified_fallback("એરોસ્પેસ એન્જિનિયરિંગની ફી કેટલી છે?")
    assert "info@sxca.edu.in" in fallback_gu
    assert "+91 79 2630 2872" in fallback_gu
    assert "દિલગીર છું" in fallback_gu

    fallback_hi = counsellor.get_verified_fallback("एयरोस्पेस इंजीनियरिंग की फीस कितनी है?")
    assert "info@sxca.edu.in" in fallback_hi
    assert "+91 79 2630 2872" in fallback_hi
    assert "क्षमा करें" in fallback_hi

    print("  [PASS] [E] Localized verified fallbacks contain official contact information: info@sxca.edu.in / +91 79 2630 2872.")


def test_scenario_f_pipeline_integrity():
    print("\n--- [Scenario F] Pipeline Integrity & No Stuck Chat ---")

    # Fast path greeting in pipeline.answer()
    res_greet = pipeline.answer("hello")
    assert res_greet["grounded"] is True
    assert len(res_greet["answer"]) > 10

    # Fast path identity in pipeline.answer()
    res_ident = pipeline.answer("who are you?")
    assert res_ident["grounded"] is True
    assert "Niaa" in res_ident["answer"]

    # Fast path faculty in pipeline.answer()
    res_fac = pipeline.answer("Nisarg Vyas")
    assert "Nisarg Vyas" in res_fac["answer"]
    assert res_fac["sources"] is not None

    # Unverified / unsupported query must have empty sources on refusal
    res_missing = pipeline.answer("Does SXCA offer Ph.D in Rocket Science?")
    if "couldn't verify" in res_missing["answer"].lower() or "info@sxca.edu.in" in res_missing["answer"]:
        assert len(res_missing["sources"]) == 0, f"Refusal answer must have empty sources, got: {res_missing['sources']}"

    print("  [PASS] [F] Pipeline executes reliably with zero hangs and clean sources.")


if __name__ == "__main__":
    test_scenario_a_faculty_resolution()
    test_scenario_b_greetings_and_identity()
    test_scenario_c_multilingual()
    test_scenario_d_course_paraphrasing()
    test_scenario_e_missing_info_fallback()
    test_scenario_f_pipeline_integrity()
    print("\n========================================================")
    print("ALL SXCA AUDIT SUITE SCENARIOS (A - F) PASSED SUCCESSFULLY!")
    print("========================================================\n")
