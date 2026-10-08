"""Production Submission Verification Test Suite for Niaa - SXCA Admission Assistant.
Verifies all 40 core criteria:
1-5: Greeting, Course Selection, Fees, Eligibility, Documents.
6-10: Application Process, Application Link, B.Eco Completed Qualification, PG after BA Eco, Commerce Stream Inquiry.
11-15: Course Negation, Topic Negation (Fees), Topic Negation (Eligibility), Typo Faculty (Nisarg Vyas), Dr. David.
16-20: College President/Leadership, Principal, Rector, HOD Data Science, CIA-2 Exam Date.
21-25: CIA-1 Exam Date, Diwali Vacation, Helpline Phone, Office Timings, Campus Address.
26-30: Hostel Facilities, Library, Placement Cell, Scholarships, Student Clubs.
31-35: Events & Culfest, Academic Departments, Science Background (Gujarati), B.Com Fee (Roman Gujarati), BCA Admission (Roman Gujarati).
36-40: Language Preference (Gujarati), Language Preference (English), Campus IT Complaint (Lab 204), Naked 'fees?' without course, Multi-turn Context Switching without leakage.
"""

import sys
import unittest
import uuid

sys.stdout.reconfigure(encoding="utf-8")

from backend.rag import counsellor, query_rewrite, pipeline
from backend.db import repo


class TestFinalSubmission(unittest.TestCase):
    def setUp(self):
        pipeline.cache.clear()

    # 1. Bare Greeting
    def test_01_greeting(self):
        res = pipeline.answer("hii")
        ans = res["answer"]
        self.assertIn("Niaa", ans)
        self.assertIn("St. Xavier's", ans)
        self.assertNotIn("Which stream", ans)
        self.assertTrue(res["grounded"])

    # 2. Direct Course Mention
    def test_02_direct_course_bca(self):
        res = pipeline.answer("I want BCA")
        ans = res["answer"]
        self.assertIn("BCA", ans)
        self.assertTrue("eligibility" in ans.lower() or "fee" in ans.lower())

    # 3. BCA Fees
    def test_03_bca_fees(self):
        res = pipeline.answer("bca fee")
        ans = res["answer"]
        self.assertIn("29,750", ans)
        self.assertIn("Self-Financed", ans)

    # 4. BCA Eligibility
    def test_04_bca_eligibility(self):
        res = pipeline.answer("bca eligibility")
        ans = res["answer"]
        self.assertIn("BCA", ans)
        self.assertTrue("12th" in ans or "pass" in ans)

    # 5. Documents Required
    def test_05_documents(self):
        res = pipeline.answer("what documents are required for admission?")
        ans = res["answer"]
        self.assertTrue("Marksheet" in ans or "Leaving Certificate" in ans)

    # 6. Admission Process
    def test_06_admission_process(self):
        res = pipeline.answer("how to apply for admission?")
        ans = res["answer"]
        self.assertTrue("merit" in ans.lower() or "portal" in ans.lower())
        self.assertIn("admissions.sxca.edu.in", ans)

    # 7. Application Link
    def test_07_application_link(self):
        res = pipeline.answer("admission portal link")
        ans = res["answer"]
        self.assertIn("https://admissions.sxca.edu.in/SXCA/", ans)

    # 8. B.Eco Completed Qualification (MUST NEVER BE B.COM)
    def test_08_b_eco_completed_qualification(self):
        q = "mene bachelor mai b.eco liya hai master mai konse course hai"
        und = query_rewrite.understand(q)
        self.assertEqual(und["intent"], "eligible_pg_courses")
        self.assertIsNone(und.get("course"))  # Active course must NOT be set to B.Com or BA-Eco
        self.assertEqual(und.get("qualification"), "BA-Economics")

        res = pipeline.answer(q)
        ans = res["answer"]
        # Must list M.Sc. Big Data Analytics, M.Sc. AI, M.A. English
        self.assertIn("Big Data Analytics", ans)
        self.assertIn("Artificial Intelligence", ans)
        self.assertIn("M.A. English", ans)
        # Must never call it B.Com or Commerce
        self.assertNotIn("Bachelor of Commerce", ans)
        self.assertNotIn("B.Com (General)", ans)
        self.assertNotIn("फांसी", ans)

    # 9. PG after BA Economics
    def test_09_pg_after_ba_economics(self):
        res = pipeline.answer("b.eco graduate hu master me kya kar sakte hai")
        ans = res["answer"]
        self.assertIn("Big Data Analytics", ans)
        self.assertIn("Artificial Intelligence", ans)
        self.assertNotIn("Bachelor of Commerce", ans)

    # 10. Commerce Stream Inquiry without guessing degree
    def test_10_commerce_stream_inquiry(self):
        q = "mene commerce kiya hai course name"
        und = query_rewrite.understand(q)
        self.assertEqual(und["intent"], "commerce_stream_inquiry")

        res = pipeline.answer(q)
        ans = res["answer"]
        self.assertIn("Commerce", ans)
        self.assertTrue("12th" in ans and ("Bachelor" in ans or "Graduation" in ans or "B.Com" in ans))
        self.assertIn("B.Com", ans)
        self.assertIn("BCA", ans)

    # 11. Course Negation & Alternatives
    def test_11_course_negation(self):
        profile = {"active_course": "BCom (General)", "stream": "Commerce"}
        und = query_rewrite.understand("bcom nahi karna hai", context={"profile": profile})
        self.assertEqual(und["intent"], "negation")

        res = pipeline.answer("bcom nahi karna hai", history=[
            {"role": "user", "content": "tell me about bcom"},
            {"role": "assistant", "content": "B.Com is an undergraduate programme"}
        ])
        ans = res["answer"]
        self.assertTrue("don't want" in ans.lower() or "noted" in ans.lower() or "understood" in ans.lower() or "સમજી" in ans)
        self.assertTrue("BCA" in ans or "Business Administration" in ans)

    # 12. Topic Negation (Fees)
    def test_12_topic_negation_fees(self):
        profile = {"active_course": "BCA"}
        res = counsellor.handle_negation("fee", profile)
        self.assertNotIn("Fee details", res)
        self.assertIn("Subjects & curriculum", res)

    # 13. Topic Negation (Eligibility)
    def test_13_topic_negation_eligibility(self):
        profile = {"active_course": "BCA"}
        res = counsellor.handle_negation("eligibility", profile)
        self.assertNotIn("Eligibility criteria", res)
        self.assertIn("Fee details", res)

    # 14. Faculty Typo: Nisarg Vyas
    def test_14_faculty_nisarg_vyas_typo(self):
        res = pipeline.answer("who is nisar vyash")
        ans = res["answer"]
        self.assertIn("Nisarg Vyas", ans)
        self.assertTrue("Data Science" in ans or "Computer" in ans)
        self.assertNotIn("I don't have those specific details", ans)

    # 15. Faculty: Fr. Dr. David
    def test_15_faculty_dr_david(self):
        res = pipeline.answer("fr.david")
        ans = res["answer"]
        self.assertIn("David", ans)
        self.assertTrue("Director" in ans or "Computer Sciences" in ans)
        self.assertNotIn("I don't have those specific details", ans)

    # 16. College President / Jesuit Leadership
    def test_16_college_leadership(self):
        res = pipeline.answer("who is president of this clg")
        ans = res["answer"]
        self.assertTrue("Pitchai" in ans or "Sebastian" in ans or "Rector" in ans or "Director" in ans)
        self.assertNotIn("I don't have those specific details", ans)

    # 17. Principal
    def test_17_principal(self):
        res = pipeline.answer("who is principal")
        self.assertIn("Dr. Sebastian V.A.", res["answer"])

    # 18. Rector
    def test_18_rector(self):
        res = pipeline.answer("who is rector")
        self.assertIn("Fr. Antony Pitchai SJ", res["answer"])

    # 19. HOD Data Science
    def test_19_hod_data_science(self):
        res = pipeline.answer("who is the hod of data science?")
        self.assertIn("Dr. Pravida Raja", res["answer"])

    # 20. CIA-2 Exam Schedule
    def test_20_cia2_exam_schedule(self):
        res = pipeline.answer("cai 2 date")
        ans = res["answer"]
        self.assertTrue("October" in ans or "CIA-2" in ans)

    # 21. CIA-1 Exam Schedule
    def test_21_cia1_exam_schedule(self):
        res = pipeline.answer("cai 1 date")
        ans = res["answer"]
        self.assertTrue("August" in ans or "CIA-1" in ans)

    # 22. Diwali Vacation
    def test_22_diwali_vacation(self):
        res = pipeline.answer("when is diwali vacation?")
        ans = res["answer"]
        self.assertIn("Diwali", ans)
        self.assertTrue("November" in ans or "October" in ans)

    # 23. College Phone Number
    def test_23_phone_number(self):
        res = pipeline.answer("college phone number")
        self.assertIn("2630 2872", res["answer"])

    # 24. Office Timings
    def test_24_office_timings(self):
        res = pipeline.answer("office timings")
        self.assertIn("10:00 AM", res["answer"])
        self.assertIn("4:00 PM", res["answer"])

    # 25. College Address
    def test_25_campus_address(self):
        res = pipeline.answer("where is the college located?")
        ans = res["answer"]
        self.assertIn("Navrangpura", ans)
        self.assertIn("380009", ans)

    # 26. Hostel Facilities
    def test_26_hostel(self):
        res = pipeline.answer("what about hostel?")
        ans = res["answer"]
        self.assertIn("Men's Hostel", ans)
        self.assertIn("15,000", ans)

    # 27. Library Facilities
    def test_27_library(self):
        res = pipeline.answer("library facilities")
        ans = res["answer"]
        self.assertIn("Library", ans)
        self.assertTrue("books" in ans.lower() or "journal" in ans.lower() or "catalog" in ans.lower())

    # 28. Placement Cell
    def test_28_placement(self):
        res = pipeline.answer("placement details")
        ans = res["answer"]
        self.assertIn("Career", ans)
        self.assertIn("TCS", ans)

    # 29. Scholarships
    def test_29_scholarships(self):
        res = pipeline.answer("scholarships available")
        ans = res["answer"]
        self.assertTrue("Digital Gujarat" in ans or "Scholarship" in ans)

    # 30. Student Clubs
    def test_30_clubs(self):
        res = pipeline.answer("what clubs are there?")
        ans = res["answer"]
        self.assertTrue("Tarumitra" in ans or "Club" in ans)

    # 31. Events & Culfest
    def test_31_events(self):
        res = pipeline.answer("what events happen in college?")
        ans = res["answer"]
        self.assertTrue("Culfest" in ans or "Sports" in ans)

    # 32. Academic Departments
    def test_32_departments(self):
        res = pipeline.answer("what departments are there?")
        ans = res["answer"]
        self.assertIn("Computer Sciences & Data Science", ans)
        self.assertIn("Physics", ans)

    # 33. Science Background Course Discovery (Gujarati)
    def test_33_gujarati_science_courses(self):
        res = pipeline.answer("science background che, kaya course available che?")
        ans = res["answer"]
        self.assertTrue("સેન્ટ ઝેવિયર્સ" in ans or "કોલેજ" in ans or "કોર્સ" in ans)
        self.assertTrue("BCA" in ans or "B.Sc." in ans)

    # 34. B.Com Fee (Roman Gujarati)
    def test_34_roman_gujarati_bcom_fee(self):
        res = pipeline.answer("bcom ni fee ketli che?")
        ans = res["answer"]
        self.assertTrue("સેન્ટ ઝેવિયર્સ" in ans or "રૂપિયા" in ans or "3,565" in ans or "19,950" in ans)

    # 35. BCA Admission (Roman Gujarati)
    def test_35_roman_gujarati_bca_process(self):
        res = pipeline.answer("bca ma admission levu che process su che?")
        ans = res["answer"]
        self.assertTrue("BCA" in ans)
        self.assertTrue("પ્રવેશ" in ans or "મેરિટ" in ans or "પોર્ટલ" in ans or "અરજી" in ans)

    # 36. Explicit Language Preference (Gujarati)
    def test_36_language_preference_gujarati(self):
        cid = str(uuid.uuid4())
        res = pipeline.answer("mane Gujarati ma j answer aapo", [], conversation_id=cid)
        ans = res["answer"]
        self.assertTrue("ગુજરાતી" in ans or "હું" in ans or "આપીશ" in ans)
        ctx = repo.get_conversation_context(cid) or {}
        prof = ctx.get("profile", {})
        self.assertEqual(prof.get("language_preference"), "gujarati")

    # 37. Explicit Language Preference (English)
    def test_37_language_preference_english(self):
        cid = str(uuid.uuid4())
        res = pipeline.answer("answer in english", [], conversation_id=cid)
        ans = res["answer"]
        self.assertTrue("English" in ans or "assist" in ans.lower() or "help" in ans.lower())
        ctx = repo.get_conversation_context(cid) or {}
        prof = ctx.get("profile", {})
        self.assertEqual(prof.get("language_preference"), "english")

    # 38. Campus IT Complaint (Lab 204)
    def test_38_campus_complaint_lab204(self):
        res = pipeline.answer("The computer in lab 204 is not working.")
        ans = res["answer"]
        self.assertIn("Lab 204", ans)
        self.assertNotIn("admission", ans.lower())
        self.assertNotIn("eligibility", ans.lower())

    # 39. Naked 'fees?' without active course
    def test_39_naked_fees_without_course(self):
        res = pipeline.answer("fees?")
        ans = res["answer"]
        self.assertIn("Which course", ans)

    # 40. Multi-Turn Context Switching without leakage (BCA -> fees -> B.Com -> fees)
    def test_40_multiturn_context_switch(self):
        cid = str(uuid.uuid4())
        # Turn 1: BCA
        res1 = pipeline.answer("tell me about BCA", [], conversation_id=cid)
        self.assertIn("BCA", res1["answer"])

        # Turn 2: fees? (should be BCA fee ₹29,750)
        res2 = pipeline.answer("fees?", [
            {"role": "user", "content": "tell me about BCA"},
            {"role": "assistant", "content": res1["answer"]}
        ], conversation_id=cid)
        self.assertIn("29,750", res2["answer"])

        # Turn 3: switch to B.Com
        res3 = pipeline.answer("actually tell me about B.Com", [
            {"role": "user", "content": "tell me about BCA"},
            {"role": "assistant", "content": res1["answer"]},
            {"role": "user", "content": "fees?"},
            {"role": "assistant", "content": res2["answer"]},
        ], conversation_id=cid)
        self.assertIn("B.Com", res3["answer"])

        # Turn 4: fees? (MUST BE B.COM FEE, NEVER BCA ₹29,750!)
        res4 = pipeline.answer("fees?", [
            {"role": "user", "content": "tell me about BCA"},
            {"role": "assistant", "content": res1["answer"]},
            {"role": "user", "content": "fees?"},
            {"role": "assistant", "content": res2["answer"]},
            {"role": "user", "content": "actually tell me about B.Com"},
            {"role": "assistant", "content": res3["answer"]},
        ], conversation_id=cid)
        ans4 = res4["answer"]
        self.assertTrue("3,565" in ans4 or "19,950" in ans4 or "30,750" in ans4)
        self.assertNotIn("29,750", ans4)  # Zero BCA leakage!


if __name__ == "__main__":
    unittest.main()
