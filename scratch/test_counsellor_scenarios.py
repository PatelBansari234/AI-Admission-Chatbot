import sys, os
import uuid
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

from backend.rag import pipeline, counsellor
from backend.db import repo

def run_tests():
    print("==========================================================")
    print("STARTING TEST SUITE: REAL ADMISSION COUNSELLOR BEHAVIOR")
    print("==========================================================\n")

    passed = 0
    total = 0

    # ---------------------------------------------------------
    # TEST 1: Bare Greeting
    # ---------------------------------------------------------
    total += 1
    cid_1 = str(uuid.uuid4())
    res1 = pipeline.answer("hii", [], conversation_id=cid_1)
    ans1 = res1["answer"]
    print(f"[TEST 1] 'hii'")
    print(f"Answer: {ans1}")
    if "Welcome to St. Xavier's College" in ans1 and "Which stream" not in ans1 and "program, stream" not in ans1:
        print("PASS: Greeting is natural and does NOT ask for stream.\n")
        passed += 1
    else:
        print("FAIL: Greeting asked for stream or was robotic.\n")

    # ---------------------------------------------------------
    # TEST 2: Direct Course Mention ("I want BCA")
    # ---------------------------------------------------------
    total += 1
    cid_bca = str(uuid.uuid4())
    res2 = pipeline.answer("I want BCA", [], conversation_id=cid_bca)
    ctx2 = repo.get_conversation_context(cid_bca) or {}
    prof2 = ctx2.get("profile", {})
    ans2 = res2["answer"]
    print(f"[TEST 2] 'I want BCA'")
    print(f"Active Course: {prof2.get('active_course')}, Mode: {prof2.get('mode')}")
    print(f"Answer: {ans2}")
    if prof2.get("active_course") == "BCA" and "BCA" in ans2 and "eligibility" in ans2.lower() and "fees" in ans2.lower():
        print("PASS: Direct course inquiry sets BCA active and offers relevant topics.\n")
        passed += 1
    else:
        print("FAIL: Did not set BCA active course properly.\n")

    # ---------------------------------------------------------
    # TEST 3: BCA Eligibility ("BCA eligibility?")
    # ---------------------------------------------------------
    total += 1
    res3 = pipeline.answer("BCA eligibility?", [], conversation_id=cid_bca)
    ans3 = res3["answer"]
    print(f"[TEST 3] 'BCA eligibility?'")
    print(f"Answer: {ans3}")
    if "BCA" in ans3 and ("12th" in ans3 or "pass" in ans3):
        print("PASS: Verified BCA eligibility returned.\n")
        passed += 1
    else:
        print("FAIL: BCA eligibility incorrect.\n")

    # ---------------------------------------------------------
    # TEST 4: Follow-up fees ("fees?") after BCA context
    # ---------------------------------------------------------
    total += 1
    res4 = pipeline.answer("fees?", [], conversation_id=cid_bca)
    ans4 = res4["answer"]
    print(f"[TEST 4] 'fees?' after BCA")
    print(f"Answer: {ans4}")
    if "BCA" in ans4 and "29,750" in ans4:
        print("PASS: Resolved 'fees?' strictly to BCA fee (₹29,750).\n")
        passed += 1
    else:
        print("FAIL: 'fees?' did not resolve to BCA fee.\n")

    # ---------------------------------------------------------
    # TEST 5: Follow-up documents ("what documents are required?")
    # ---------------------------------------------------------
    total += 1
    res5 = pipeline.answer("what documents are required?", [], conversation_id=cid_bca)
    ans5 = res5["answer"]
    print(f"[TEST 5] 'what documents are required?'")
    print(f"Answer: {ans5}")
    if "Marksheet" in ans5 or "School Leaving Certificate" in ans5:
        print("PASS: Relevant admission documents returned.\n")
        passed += 1
    else:
        print("FAIL: Documents missing.\n")

    # ---------------------------------------------------------
    # TEST 6: Student background ("I completed 12th Science")
    # ---------------------------------------------------------
    total += 1
    cid_counsel = str(uuid.uuid4())
    res6 = pipeline.answer("I completed 12th Science", [], conversation_id=cid_counsel)
    ctx6 = repo.get_conversation_context(cid_counsel) or {}
    prof6 = ctx6.get("profile", {})
    ans6 = res6["answer"]
    print(f"[TEST 6] 'I completed 12th Science'")
    print(f"Profile: education_level={prof6.get('education_level')}, stream={prof6.get('stream')}")
    print(f"Answer: {ans6}")
    if prof6.get("education_level") == "12th" and prof6.get("stream") == "Science" and "Which stream" not in ans6:
        print("PASS: Profile correctly recorded 12th Science and asked ONE area-interest question without asking for stream again.\n")
        passed += 1
    else:
        print("FAIL: Failed to record 12th Science.\n")

    # ---------------------------------------------------------
    # TEST 7: Needs course guidance ("I don't know which course to choose")
    # ---------------------------------------------------------
    total += 1
    cid_confused = str(uuid.uuid4())
    res7 = pipeline.answer("I don't know which course to choose", [], conversation_id=cid_confused)
    ans7 = res7["answer"]
    print(f"[TEST 7] 'I don't know which course to choose'")
    print(f"Answer: {ans7}")
    if "narrow it down" in ans7.lower() and ("bachelor" in ans7.lower() or "stream" in ans7.lower() or "science" in ans7.lower()):
        print("PASS: Entered counselling mode and asked ONE helpful question.\n")
        passed += 1
    else:
        print("FAIL: Counselling flow did not trigger as expected.\n")

    # ---------------------------------------------------------
    # TEST 8: Computer interest ("I like computers") after 12th Science
    # ---------------------------------------------------------
    total += 1
    res8 = pipeline.answer("I like computers", [], conversation_id=cid_counsel)
    ctx8 = repo.get_conversation_context(cid_counsel) or {}
    prof8 = ctx8.get("profile", {})
    ans8 = res8["answer"]
    print(f"[TEST 8] 'I like computers' after 12th Science")
    print(f"Profile: interest={prof8.get('interest_area')}")
    print(f"Answer: {ans8}")
    if ("BCA" in ans8 or "Artificial Intelligence" in ans8) and "MSc-Microbiology" not in ans8 and "MSc" not in ans8:
        print("PASS: Offered Computer/IT UG courses without mixing PG courses.\n")
        passed += 1
    else:
        print("FAIL: Mixed PG courses or failed Computer/IT counselling.\n")

    # ---------------------------------------------------------
    # TEST 9: Career Goal ("I want to become an AI engineer")
    # ---------------------------------------------------------
    total += 1
    res9 = pipeline.answer("I want to become an AI engineer", [], conversation_id=cid_counsel)
    ctx9 = repo.get_conversation_context(cid_counsel) or {}
    prof9 = ctx9.get("profile", {})
    ans9 = res9["answer"]
    print(f"[TEST 9] 'I want to become an AI engineer'")
    print(f"Profile: career_goal={prof9.get('career_goal')}")
    print(f"Answer: {ans9}")
    if ("Artificial Intelligence" in ans9 or "BCA" in ans9) and "guarantee" not in ans9.lower() and "will become" not in ans9.lower():
        print("PASS: Highlighted verified AI/ML offerings without false guarantees.\n")
        passed += 1
    else:
        print("FAIL: AI career goal guidance failed.\n")

    # ---------------------------------------------------------
    # TEST 10: Course discovery ("What courses do you have?") with known background
    # ---------------------------------------------------------
    total += 1
    res10 = pipeline.answer("What courses do you have?", [], conversation_id=cid_counsel)
    ans10 = res10["answer"]
    print(f"[TEST 10] 'What courses do you have?' (Context: 12th Science)")
    print(f"Answer: {ans10}")
    if "Science" in ans10 and "BCA" in ans10 and "MSc-Microbiology" not in ans10:
        print("PASS: Showed 12th Science relevant courses contextually.\n")
        passed += 1
    else:
        print("FAIL: Failed contextual course discovery.\n")

    # ---------------------------------------------------------
    # TEST 11: Explicit UG request ("UG courses")
    # ---------------------------------------------------------
    total += 1
    cid_fresh = str(uuid.uuid4())
    res11 = pipeline.answer("UG courses", [], conversation_id=cid_fresh)
    ans11 = res11["answer"]
    print(f"[TEST 11] 'UG courses'")
    print(f"Answer excerpt: {ans11[:200]}...")
    if "Undergraduate" in ans11 and ("BCA" in ans11 or "B.C.A." in ans11) and "M.Sc." not in ans11 and "MSc" not in ans11:
        print("PASS: Listed only UG courses without PG mixing.\n")
        passed += 1
    else:
        print("FAIL: Mixed PG courses into UG list.\n")

    # ---------------------------------------------------------
    # TEST 12: Context Switch to PG ("MSc AI")
    # ---------------------------------------------------------
    total += 1
    res12 = pipeline.answer("MSc AI", [], conversation_id=cid_fresh)
    ctx12 = repo.get_conversation_context(cid_fresh) or {}
    prof12 = ctx12.get("profile", {})
    ans12 = res12["answer"]
    print(f"[TEST 12] 'MSc AI'")
    print(f"Active Course: {prof12.get('active_course')}, Level: {prof12.get('course_level')}")
    print(f"Answer: {ans12}")
    if prof12.get("active_course") == "MSc-Artificial Intelligence" and prof12.get("course_level") == "PG":
        print("PASS: Active course switched to MSc-Artificial Intelligence (PG).\n")
        passed += 1
    else:
        print("FAIL: Failed to switch to MSc AI.\n")

    # ---------------------------------------------------------
    # TEST 13: MSc AI Eligibility ("eligibility?")
    # ---------------------------------------------------------
    total += 1
    res13 = pipeline.answer("eligibility?", [], conversation_id=cid_fresh)
    ans13 = res13["answer"]
    print(f"[TEST 13] 'eligibility?' after MSc AI")
    print(f"Answer: {ans13}")
    if "MSc-Artificial Intelligence" in ans13 and ("BSc" in ans13 or "BCA" in ans13):
        print("PASS: Resolved to MSc AI eligibility.\n")
        passed += 1
    else:
        print("FAIL: MSc AI eligibility failed.\n")

    # ---------------------------------------------------------
    # TEST 14: MSc AI Fees ("fees?")
    # ---------------------------------------------------------
    total += 1
    res14 = pipeline.answer("fees?", [], conversation_id=cid_fresh)
    ans14 = res14["answer"]
    print(f"[TEST 14] 'fees?' after MSc AI")
    print(f"Answer: {ans14}")
    if "MSc-Artificial Intelligence" in ans14 and "50,000" in ans14:
        print("PASS: Resolved to MSc AI fee (₹50,000).\n")
        passed += 1
    else:
        print("FAIL: MSc AI fee failed.\n")

    # ---------------------------------------------------------
    # TEST 15: Hostel query ("what about hostel?")
    # ---------------------------------------------------------
    total += 1
    res15 = pipeline.answer("what about hostel?", [], conversation_id=cid_fresh)
    ans15 = res15["answer"]
    ctx15 = repo.get_conversation_context(cid_fresh) or {}
    prof15 = ctx15.get("profile", {})
    print(f"[TEST 15] 'what about hostel?'")
    print(f"Answer: {ans15}")
    print(f"Active Course preserved: {prof15.get('active_course')}")
    if "hostel" in ans15.lower() and prof15.get("active_course") == "MSc-Artificial Intelligence":
        print("PASS: Answered hostel information and preserved active course.\n")
        passed += 1
    else:
        print("FAIL: Hostel info failed or lost active course.\n")

    # ---------------------------------------------------------
    # TEST 16: Course switch ("actually tell me about B.Com")
    # ---------------------------------------------------------
    total += 1
    res16 = pipeline.answer("actually tell me about B.Com", [], conversation_id=cid_fresh)
    ctx16 = repo.get_conversation_context(cid_fresh) or {}
    prof16 = ctx16.get("profile", {})
    ans16 = res16["answer"]
    print(f"[TEST 16] 'actually tell me about B.Com'")
    print(f"Active Course: {prof16.get('active_course')}")
    print(f"Answer: {ans16}")
    if "BCom" in (prof16.get("active_course") or "") and "B.Com" in ans16:
        print("PASS: Active course cleanly switched to B.Com.\n")
        passed += 1
    else:
        print("FAIL: Failed to switch active course to B.Com.\n")

    # ---------------------------------------------------------
    # TEST 17: Fees after B.Com ("fees?") - MUST NOT BE BCA!
    # ---------------------------------------------------------
    total += 1
    res17 = pipeline.answer("fees?", [], conversation_id=cid_fresh)
    ans17 = res17["answer"]
    print(f"[TEST 17] 'fees?' after B.Com switch")
    print(f"Answer: {ans17}")
    if "BCom" in ans17 and "30,750" in ans17 and "BCA" not in ans17:
        print("PASS: Strictly answered B.Com fees (₹30,750) without leaking BCA fees.\n")
        passed += 1
    else:
        print("FAIL: Leaked previous course fees or wrong B.Com fee.\n")

    # ---------------------------------------------------------
    # TEST 18: Campus Complaint ("The computer in lab 204 is not working.")
    # ---------------------------------------------------------
    total += 1
    cid_complaint = str(uuid.uuid4())
    res18 = pipeline.answer("The computer in lab 204 is not working.", [], conversation_id=cid_complaint)
    ans18 = res18["answer"]
    print(f"[TEST 18] 'The computer in lab 204 is not working.'")
    print(f"Answer: {ans18}")
    if "Lab 204" in ans18 and "admission" not in ans18.lower() and "course" not in ans18.lower():
        print("PASS: Identified complaint for Lab 204 with campus IT support and zero admission mix-up.\n")
        passed += 1
    else:
        print("FAIL: Complaint mixed with admission text or failed to detect location.\n")

    # ---------------------------------------------------------
    # TEST 19: Fees asked without active course
    # ---------------------------------------------------------
    total += 1
    cid_nocourse = str(uuid.uuid4())
    res19 = pipeline.answer("fees?", [], conversation_id=cid_nocourse)
    ans19 = res19["answer"]
    print(f"[TEST 19] 'fees?' without active course")
    print(f"Answer: {ans19}")
    if "Which course" in ans19:
        print("PASS: Prompted which course's fee to check.\n")
        passed += 1
    else:
        print("FAIL: Did not prompt for course name.\n")

    print("==========================================================")
    print(f"TEST RESULTS: {passed} / {total} PASSED ({(passed/total)*100:.1f}%)")
    print("==========================================================")
    return passed == total

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
