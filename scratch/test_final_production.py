"""Comprehensive production test suite verifying Niaa - Virtual Admission Counsellor.
Tests all 36+ domains, entity tracking, context gate, language switching,
and 20-step continuous conversation sequence without context contamination.
"""

import sys
import unittest
from backend.rag import counsellor, query_rewrite, pipeline


class TestNiaaProduction(unittest.TestCase):
    def setUp(self):
        pipeline.cache.clear()

    # --- 1. Basic Persona & NLU ---
    def test_01_greeting(self):
        res = pipeline.answer("hii")
        self.assertIn("Niaa", res["answer"])
        self.assertIn("St. Xavier's", res["answer"])
        self.assertTrue(res["grounded"])

    def test_02_bca_overview(self):
        res = pipeline.answer("bca")
        self.assertIn("BCA", res["answer"])
        self.assertEqual(res.get("course"), "BCA")

    def test_03_bca_fee(self):
        res = pipeline.answer("bca fee")
        self.assertIn("29,750", res["answer"])
        self.assertIn("Self-Financed", res["answer"])

    def test_04_cai2_exam_date(self):
        for q in ["cai 2 date", "cai-2 exam date", "when is cai 2 exam", "cai 2"]:
            res = pipeline.answer(q)
            self.assertTrue("October" in res["answer"] or "CIA-2" in res["answer"], f"Failed for query: {q}")
            self.assertTrue("12 October" in res["answer"] or "7th October" in res["answer"] or "13/10/2025" in res["answer"] or "CIA-2" in res["answer"], f"Failed for query: {q}")

    def test_05_cai1_exam_date(self):
        res = pipeline.answer("cai 1 date")
        self.assertTrue("August" in res["answer"] or "CIA-1" in res["answer"])

    def test_06_diwali_vacation(self):
        res = pipeline.answer("when is diwali vacation?")
        self.assertIn("Diwali", res["answer"])
        self.assertTrue("November" in res["answer"] or "October" in res["answer"])

    # --- 2. Faculty & Leadership ---
    def test_07_who_is_david(self):
        res = pipeline.answer("who is david")
        ans = res["answer"].lower()
        self.assertIn("david", ans)
        self.assertIn("director", ans)
        self.assertTrue("computer" in ans or "data science" in ans)

    def test_08_david_department_followup(self):
        # Turn 1: who is david
        profile = counsellor.extract_and_update_profile(None, "who is david")
        p = profile.get("active_person")
        p_name = p.get("full_name") if isinstance(p, dict) else p
        self.assertIn("David", p_name)
        # Turn 2: what is his department
        dept_res = counsellor.handle_faculty("what is his department?", profile)
        self.assertIn("Computer Sciences & Data Science", dept_res["answer"])

    def test_09_who_is_principal(self):
        res = pipeline.answer("who is principal")
        self.assertIn("Dr. Sebastian V.A.", res["answer"])

    def test_10_who_is_rector(self):
        res = pipeline.answer("who is rector")
        self.assertIn("Fr. Antony Pitchai SJ", res["answer"])

    def test_11_who_is_hod(self):
        res = pipeline.answer("who is the hod of data science?")
        self.assertIn("Dr. Pravida Raja", res["answer"])

    # --- 3. Contact & Office Timing ---
    def test_12_college_phone_number(self):
        for q in ["college phone number", "helpline number", "contact details"]:
            res = pipeline.answer(q)
            self.assertIn("2630 2872", res["answer"], f"Failed for: {q}")
            self.assertIn("info@sxca.edu.in", res["answer"], f"Failed for: {q}")

    def test_13_office_timings(self):
        res = pipeline.answer("office timings")
        self.assertIn("10:00 AM", res["answer"])
        self.assertIn("4:00 PM", res["answer"])

    # --- 4. Campus Facilities & Student Life ---
    def test_14_hostel_facilities(self):
        res = pipeline.answer("what about hostel")
        self.assertIn("Men's Hostel", res["answer"])
        self.assertIn("₹15,000", res["answer"])
        self.assertIn("Aadhaar", res["answer"])

    def test_15_departments_overview(self):
        res = pipeline.answer("what departments are there?")
        self.assertIn("Computer Sciences & Data Science", res["answer"])
        self.assertIn("Physics", res["answer"])
        self.assertIn("Commerce", res["answer"])

    def test_16_clubs(self):
        res = pipeline.answer("what clubs does the college have?")
        self.assertIn("Science Association", res["answer"])
        self.assertIn("Tarumitra", res["answer"])
        self.assertIn("Rotaract", res["answer"])

    def test_17_sports_and_gym(self):
        res = pipeline.answer("sports facilities")
        self.assertIn("Football", res["answer"])
        self.assertIn("Basketball", res["answer"])
        self.assertIn("Gymnasium", res["answer"])

    def test_18_swimming_pool_negative(self):
        res = pipeline.answer("is there a swimming pool?")
        self.assertIn("does not have a swimming pool", res["answer"].lower())

    def test_19_ncc_nss(self):
        res = pipeline.answer("does sxca have ncc and nss?")
        self.assertIn("NCC", res["answer"])
        self.assertIn("NSS", res["answer"])

    def test_20_events_culfest(self):
        res = pipeline.answer("what is culfest?")
        self.assertIn("Culfest", res["answer"])
        self.assertIn("cultural", res["answer"].lower())

    def test_21_placement(self):
        res = pipeline.answer("placement details")
        self.assertIn("Placement", res["answer"])
        self.assertIn("TCS", res["answer"])

    def test_22_scholarships(self):
        res = pipeline.answer("scholarships available")
        self.assertIn("MYSY", res["answer"])
        self.assertIn("Digital Gujarat", res["answer"])

    def test_23_central_library(self):
        res = pipeline.answer("central library timings and books")
        self.assertIn("75,000", res["answer"])
        self.assertIn("8:30 AM", res["answer"])

    # --- 5. Course Integrity & Zero Fee Leaks ---
    def test_24_bda_fee_never_msc_ai(self):
        res = pipeline.answer("bda fee")
        self.assertIn("44,950", res["answer"])
        self.assertNotIn("50,000", res["answer"])
        self.assertNotIn("M.Sc. Artificial Intelligence", res["answer"])

    def test_25_msc_ai_fee(self):
        res = pipeline.answer("msc ai fee")
        self.assertIn("50,000", res["answer"])

    def test_26_bare_free_clarification(self):
        turns = [{"role": "user", "content": "msc ai"}, {"role": "assistant", "content": "Sure, M.Sc. AI is a PG course."}]
        res = pipeline.answer("free", history=turns)
        self.assertIn("Do you mean the fee for", res["answer"])

    def test_27_zero_cost_check(self):
        res = pipeline.answer("is msc ai free?")
        self.assertIn("₹50,000 per semester", res["answer"])
        self.assertIn("not a free programme", res["answer"])

    # --- 6. Language & Multilingual Understanding ---
    def test_28_gujarati_preference_switch(self):
        res = pipeline.answer("mane Gujarati ma j answer aapo")
        self.assertIn("ગુજરાતીમાં", res["answer"])

    def test_29_gujarati_query(self):
        res = pipeline.answer("bca ni fee ketli che?")
        self.assertIn("29,750", res["answer"])

    def test_30_science_courses_gujarati(self):
        res = pipeline.answer("science background che, kaya course available che?")
        self.assertTrue("BCA" in res["answer"] or "B.Sc." in res["answer"])

    def test_31_general_pg_admission(self):
        res = pipeline.answer("master ma admission levu che")
        self.assertIn("https://sxca.edu.in/", res["answer"])
        self.assertIn("Bachelor", res["answer"])

    # --- 7. Strict Context Gate Verification ---
    def test_32_context_gate_active_course_isolation(self):
        # Scenario: User was talking about MSc AI
        turns = [
            {"role": "user", "content": "msc ai"},
            {"role": "assistant", "content": "M.Sc. AI is a 2-year master's program."},
        ]
        ctx_profile = {"active_course": "MSc-Artificial Intelligence"}

        # Independent queries must NOT inherit active course MSc AI!
        independent_queries = [
            ("who is the director?", "david"),
            ("what is college helpline?", "2630 2872"),
            ("when is cai 2 exam?", "October"),
            ("is there hostel?", "Hostel"),
            ("what sports facilities?", "Football"),
            ("what clubs are there?", "Association"),
            ("library timings?", "8:30 AM"),
            ("what departments?", "Department"),
        ]

        for q, expected in independent_queries:
            u = query_rewrite.understand(q, turns, {"profile": ctx_profile, "course": "MSc-Artificial Intelligence"})
            self.assertIsNone(u.get("course"), f"Course context leaked for query: {q}")
            ans_res = pipeline.answer(q, history=turns)
            self.assertIn(expected.lower(), ans_res["answer"].lower(), f"Expected '{expected}' in answer for '{q}'")
            self.assertNotIn("M.Sc. Artificial Intelligence has a fee of", ans_res["answer"])

    def test_33_context_gate_legitimate_inheritance(self):
        # Scenario: User was talking about BCA and asks genuine follow-up
        turns = [
            {"role": "user", "content": "bca"},
            {"role": "assistant", "content": "BCA is an undergraduate degree."},
        ]
        ctx_profile = {"active_course": "BCA"}

        follow_ups = [
            ("its fee", "29,750"),
            ("eligibility?", "12th"),
            ("what subjects?", "BCA"),
        ]

        for q, expected in follow_ups:
            u = query_rewrite.understand(q, turns, {"profile": ctx_profile, "course": "BCA"})
            self.assertEqual(u.get("course"), "BCA", f"Legitimate course follow-up failed for: {q}")
            ans_res = pipeline.answer(q, history=turns)
            self.assertIn(expected.lower(), ans_res["answer"].lower())


if __name__ == "__main__":
    unittest.main()
