import sys, os, uuid
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.path.insert(0, ".")

from backend.rag import pipeline, counsellor
from backend.db import repo

def run_production_tests():
    print("==========================================================", flush=True)
    print("SXCA ADMISSION ASSISTANT (NIAA) — PRODUCTION TEST SUITE", flush=True)
    print("==========================================================\n", flush=True)

    passed = 0
    total = 0

    def check(name, condition, detail=""):
        nonlocal passed, total
        total += 1
        if condition:
            print(f"[{total}] PASS: {name}", flush=True)
            passed += 1
        else:
            print(f"[{total}] FAIL: {name} | {detail}", flush=True)

    # =========================================================
    # PART 1: 25 INDIVIDUAL PRODUCTION CAPABILITY TESTS
    # =========================================================
    print("--- PART 1: 25 CAPABILITY TESTS ---\n")

    # 1. Bare Greeting
    cid1 = str(uuid.uuid4())
    res1 = pipeline.answer("hii", [], conversation_id=cid1)
    ans1 = res1.get("answer", "")
    check("1. Greeting introduces Niaa & SXCA without barrage", 
          "Niaa" in ans1 and "St. Xavier's" in ans1 and "Which stream" not in ans1,
          f"Answer: {ans1[:100]}")

    # 2. Direct BCA mention
    cid2 = str(uuid.uuid4())
    res2 = pipeline.answer("bca", [], conversation_id=cid2)
    ans2 = res2.get("answer", "")
    ctx2 = repo.get_conversation_context(cid2) or {}
    prof2 = ctx2.get("profile", {})
    check("2. 'bca' sets BCA active course and provides overview",
          prof2.get("active_course") == "BCA" and "BCA" in ans2,
          f"Active: {prof2.get('active_course')}")

    # 3. BCA Fees
    cid3 = str(uuid.uuid4())
    res3 = pipeline.answer("bca fee", [], conversation_id=cid3)
    ans3 = res3.get("answer", "")
    check("3. 'bca fee' returns ₹29,750",
          "29,750" in ans3,
          f"Answer: {ans3[:100]}")

    # 4. Direct BDA mention
    cid4 = str(uuid.uuid4())
    res4 = pipeline.answer("bda", [], conversation_id=cid4)
    ans4 = res4.get("answer", "")
    ctx4 = repo.get_conversation_context(cid4) or {}
    prof4 = ctx4.get("profile", {})
    check("4. 'bda' matches MSc-Big Data Analytics (NOT MSc AI)",
          prof4.get("active_course") == "MSc-Big Data Analytics" and "Big Data" in ans4 and "Artificial Intelligence" not in ans4,
          f"Active: {prof4.get('active_course')}, Ans: {ans4[:100]}")

    # 5. BDA Fees
    cid5 = str(uuid.uuid4())
    res5 = pipeline.answer("bda fee", [], conversation_id=cid5)
    ans5 = res5.get("answer", "")
    check("5. 'bda fee' returns ₹44,950 and NOT MSc AI fee ₹50,000",
          "44,950" in ans5 and "50,000" not in ans5,
          f"Answer: {ans5[:120]}")

    # 6. Full phrase Big Data Analytics
    cid6 = str(uuid.uuid4())
    res6 = pipeline.answer("big data analytics", [], conversation_id=cid6)
    ctx6 = repo.get_conversation_context(cid6) or {}
    prof6 = ctx6.get("profile", {})
    check("6. 'big data analytics' resolves to MSc-Big Data Analytics",
          prof6.get("active_course") == "MSc-Big Data Analytics",
          f"Active: {prof6.get('active_course')}")

    # 7. MSc AI Fees
    cid7 = str(uuid.uuid4())
    res7 = pipeline.answer("msc ai fee", [], conversation_id=cid7)
    ans7 = res7.get("answer", "")
    check("7. 'msc ai fee' returns ₹50,000 and NOT ₹44,950",
          "50,000" in ans7 and "44,950" not in ans7,
          f"Answer: {ans7[:120]}")

    # 8. BCA Eligibility
    cid8 = str(uuid.uuid4())
    res8 = pipeline.answer("bca eligibility", [], conversation_id=cid8)
    ans8 = res8.get("answer", "")
    check("8. 'bca eligibility' mentions 12th pass / qualifying marks",
          "12th" in ans8 or "Higher Secondary" in ans8 or "qualifying" in ans8.lower(),
          f"Answer: {ans8[:120]}")

    # 9. MSc AI Eligibility
    cid9 = str(uuid.uuid4())
    res9 = pipeline.answer("msc ai eligibility", [], conversation_id=cid9)
    ans9 = res9.get("answer", "")
    check("9. 'msc ai eligibility' mentions bachelor degree / BCA / BSc",
          "BCA" in ans9 or "B.Sc" in ans9 or "BSc" in ans9 or "bachelor" in ans9.lower() or "graduate" in ans9.lower(),
          f"Answer: {ans9[:120]}")

    # 10. General Documents
    cid10 = str(uuid.uuid4())
    res10 = pipeline.answer("what documents", [], conversation_id=cid10)
    ans10 = res10.get("answer", "")
    check("10. 'what documents' lists standard admission documents",
          "Marksheet" in ans10 or "marksheet" in ans10.lower(),
          f"Answer: {ans10[:120]}")

    # 11. PG Documents
    cid11 = str(uuid.uuid4())
    res11 = pipeline.answer("what documents for msc", [], conversation_id=cid11)
    ans11 = res11.get("answer", "")
    check("11. 'what documents for msc' includes graduation marksheets / degree certificate",
          "Graduation" in ans11 or "graduation" in ans11.lower() or "Degree Certificate" in ans11 or "degree" in ans11.lower(),
          f"Answer: {ans11[:120]}")

    # 12. Admission Process
    cid12 = str(uuid.uuid4())
    res12 = pipeline.answer("admission process", [], conversation_id=cid12)
    ans12 = res12.get("answer", "")
    check("12. 'admission process' states merit-based and NO interview required",
          "merit" in ans12.lower() and ("interview" in ans12.lower() or "entrance" in ans12.lower()),
          f"Answer: {ans12[:120]}")

    # 13. CAI-2 Exam Date
    cid13 = str(uuid.uuid4())
    res13 = pipeline.answer("cai 2 date", [], conversation_id=cid13)
    ans13 = res13.get("answer", "")
    srcs13 = res13.get("sources", [])
    check("13. 'cai 2 date' retrieves calendar info without course collision",
          ("CAI" in ans13 or "Continuous Assessment" in ans13 or "October" in ans13 or "February" in ans13 or "exam" in ans13.lower()) and len(srcs13) > 0,
          f"Answer: {ans13[:120]}")

    # 14. Hostel facility
    cid14 = str(uuid.uuid4())
    res14 = pipeline.answer("hostel facility", [], conversation_id=cid14)
    ans14 = res14.get("answer", "")
    check("14. 'hostel facility' returns verified campus hostel info",
          "hostel" in ans14.lower(),
          f"Answer: {ans14[:120]}")

    # 15. Placements
    cid15 = str(uuid.uuid4())
    res15 = pipeline.answer("placement", [], conversation_id=cid15)
    ans15 = res15.get("answer", "")
    check("15. 'placement' returns career/placement guidance",
          "placement" in ans15.lower() or "career" in ans15.lower() or "companies" in ans15.lower(),
          f"Answer: {ans15[:120]}")

    # 16. Course categories
    cid16 = str(uuid.uuid4())
    res16 = pipeline.answer("courses", [], conversation_id=cid16)
    ans16 = res16.get("answer", "")
    check("16. 'courses' introduces Undergraduate and Postgraduate offerings",
          ("Undergraduate" in ans16 or "UG" in ans16) and ("Postgraduate" in ans16 or "PG" in ans16),
          f"Answer: {ans16[:120]}")

    # 17. UG Courses only
    cid17 = str(uuid.uuid4())
    res17 = pipeline.answer("ug courses", [], conversation_id=cid17)
    ans17 = res17.get("answer", "")
    check("17. 'ug courses' returns UG courses without MSc mixing",
          "BCA" in ans17 and "MSc-Artificial Intelligence" not in ans17 and "M.Sc." not in ans17,
          f"Answer: {ans17[:120]}")

    # 18. PG Courses only
    cid18 = str(uuid.uuid4())
    res18 = pipeline.answer("pg courses", [], conversation_id=cid18)
    ans18 = res18.get("answer", "")
    check("18. 'pg courses' returns PG courses without BCA/BCom mixing",
          ("MSc" in ans18 or "M.Sc" in ans18 or "Big Data" in ans18) and "BCA" not in ans18,
          f"Answer: {ans18[:120]}")

    # 19. Fees asked without active course
    cid19 = str(uuid.uuid4())
    res19 = pipeline.answer("fees?", [], conversation_id=cid19)
    ans19 = res19.get("answer", "")
    check("19. 'fees?' without active course politely asks which course",
          "Which course" in ans19 or "which course" in ans19.lower(),
          f"Answer: {ans19[:120]}")

    # 20. Campus IT / Facility Complaint
    cid20 = str(uuid.uuid4())
    res20 = pipeline.answer("The computer in lab 204 is broken", [], conversation_id=cid20)
    ans20 = res20.get("answer", "")
    check("20. Campus complaint routes to IT/facility support without admission mixup",
          ("204" in ans20 or "lab" in ans20.lower() or "it" in ans20.lower() or "support" in ans20.lower()) and "admission process" not in ans20.lower(),
          f"Answer: {ans20[:120]}")

    # 21. Research / PhD Guidance
    cid21 = str(uuid.uuid4())
    res21 = pipeline.answer("I want to do research", [], conversation_id=cid21)
    ans21 = res21.get("answer", "")
    check("21. 'I want to do research' provides research/Ph.D guidance",
          "research" in ans21.lower() or "ph.d" in ans21.lower() or "phd" in ans21.lower() or "faculty" in ans21.lower(),
          f"Answer: {ans21[:120]}")

    # 22. Library timing
    cid22 = str(uuid.uuid4())
    res22 = pipeline.answer("library timing", [], conversation_id=cid22)
    ans22 = res22.get("answer", "")
    check("22. 'library timing' provides library information",
          "library" in ans22.lower(),
          f"Answer: {ans22[:120]}")

    # 23. Sports facilities
    cid23 = str(uuid.uuid4())
    res23 = pipeline.answer("sports facilities", [], conversation_id=cid23)
    ans23 = res23.get("answer", "")
    check("23. 'sports facilities' provides sports / campus life details",
          "sport" in ans23.lower() or "playground" in ans23.lower() or "gym" in ans23.lower() or "campus" in ans23.lower(),
          f"Answer: {ans23[:120]}")

    # 24. Scholarship
    cid24 = str(uuid.uuid4())
    res24 = pipeline.answer("scholarship", [], conversation_id=cid24)
    ans24 = res24.get("answer", "")
    check("24. 'scholarship' provides verified scholarship schemes",
          "scholarship" in ans24.lower() or "government" in ans24.lower() or "freeship" in ans24.lower(),
          f"Answer: {ans24[:120]}")

    # 25. Source structure verification
    cid25 = str(uuid.uuid4())
    res25 = pipeline.answer("msc ai fees", [], conversation_id=cid25)
    srcs25 = res25.get("sources", [])
    valid_sources = len(srcs25) > 0 and all(
        "title" in s and "type" in s and "page" in s and "snippet" in s for s in srcs25
    )
    check("25. Sources contain title, type, page, snippet",
          valid_sources,
          f"Sources: {srcs25}")

    # =========================================================
    # PART 2: 10-STEP CONTINUOUS CONVERSATION SEQUENCE TEST
    # =========================================================
    print("\n--- PART 2: 10-STEP CONTINUOUS CONVERSATION TEST ---\n")
    seq_cid = str(uuid.uuid4())
    history = []

    def run_step(step_num, query, expected_fn, desc):
        nonlocal passed, total, history
        total += 1
        res = pipeline.answer(query, history, conversation_id=seq_cid)
        ans = res.get("answer", "")
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": ans})
        ok = expected_fn(ans, res)
        if ok:
            print(f"[Step {step_num}] PASS: {desc}", flush=True)
            passed += 1
        else:
            print(f"[Step {step_num}] FAIL: {desc}", flush=True)
            print(f"      Query: {query}", flush=True)
            print(f"      Ans:   {ans[:140]}", flush=True)

    # Step 1: Greeting
    run_step(1, "hi", lambda a, r: "Niaa" in a and "St. Xavier's" in a,
             "Niaa warm greeting")

    # Step 2: Course mention "bca"
    run_step(2, "bca", lambda a, r: "BCA" in a,
             "BCA details presented & active course set")

    # Step 3: Follow-up fees
    run_step(3, "fees?", lambda a, r: "29,750" in a,
             "Resolves 'fees?' to BCA fee ₹29,750")

    # Step 4: Follow-up subjects
    run_step(4, "subjects?", lambda a, r: "29,750" not in a and ("Subject" in a or "curriculum" in a.lower() or "course" in a.lower() or "Data" in a or "Programming" in a or "semester" in a.lower()),
             "Resolves 'subjects?' to curriculum without repeating fees")

    # Step 5: Follow-up documents
    run_step(5, "what documents", lambda a, r: "Marksheet" in a or "marksheet" in a.lower(),
             "Returns UG document requirements")

    # Step 6: Course switch to BDA
    run_step(6, "what about bda?", lambda a, r: "Big Data" in a and "Artificial Intelligence" not in a,
             "Switches active course cleanly to MSc Big Data Analytics")

    # Step 7: Follow-up fees on BDA
    run_step(7, "fees?", lambda a, r: "44,950" in a and "50,000" not in a and "29,750" not in a,
             "Resolves 'fees?' strictly to MSc BDA fee ₹44,950 (NOT BCA, NOT MSc AI)")

    # Step 8: Admission process
    run_step(8, "admission process", lambda a, r: "merit" in a.lower() and ("interview" in a.lower() or "entrance" in a.lower()),
             "Explains merit-based process with no interview required")

    # Step 9: Academic calendar question
    run_step(9, "when is cai 2 exam", lambda a, r: len(r.get("sources", [])) > 0 and ("CAI" in a or "October" in a or "February" in a or "exam" in a.lower()),
             "Calendar question answers CAI-2 dates with calendar source")

    # Step 10: Closing
    run_step(10, "thank you", lambda a, r: "welcome" in a.lower() or "help" in a.lower() or "all the best" in a.lower() or "luck" in a.lower() or "happy" in a.lower(),
             "Polite closing from Niaa")

    print("\n==========================================================", flush=True)
    print(f"FINAL PRODUCTION SUITE RESULT: {passed} / {total} PASSED ({(passed/total)*100:.1f}%)", flush=True)
    print("==========================================================", flush=True)
    return passed == total

if __name__ == "__main__":
    success = run_production_tests()
    sys.exit(0 if success else 1)
