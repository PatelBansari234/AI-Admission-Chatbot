import sys
import os
sys.path.insert(0, os.getcwd())

from backend.rag import counsellor, query_rewrite, pipeline

def test_faculty_resolution():
    print("=== TEST A: 'Who is Nisarg?' (Ambiguous Name) ===")
    res_a = counsellor.resolve_faculty("Who is Nisarg?")
    print("counsellor.resolve_faculty status:", res_a["status"])
    print("Candidates found:", [c["canonical_name"] for c in res_a.get("candidates", [])])
    assert res_a["status"] == "AMBIGUOUS", f"Expected AMBIGUOUS, got {res_a['status']}"
    assert len(res_a["candidates"]) > 1, f"Expected multiple candidates, got {len(res_a['candidates'])}"
    
    # End-to-end pipeline answer check
    pipe_a = pipeline.answer("Who is Nisarg?")
    print("Pipeline answer preview:\n", pipe_a["answer"])
    assert "multiple people" in pipe_a["answer"].lower() or "specify the full name" in pipe_a["answer"].lower()
    assert "Nisarg Vyas" in pipe_a["answer"]
    assert "Nisarg Patel" in pipe_a["answer"]
    # Must NOT select or answer about Nisarg Vyas directly
    assert "Coordinator of AI and ML" not in pipe_a["answer"]
    print("PASS: TEST A verified (AMBIGUOUS with candidates list, no person selected)!\n")

    print("=== TEST B: 'Who is Nisarg Vyas?' (Full Name) ===")
    res_b = counsellor.resolve_faculty("Who is Nisarg Vyas?")
    print("counsellor.resolve_faculty status:", res_b["status"])
    assert res_b["status"] == "RESOLVED", f"Expected RESOLVED, got {res_b['status']}"
    assert res_b["person"]["canonical_name"] == "Nisarg Vyas"

    pipe_b = pipeline.answer("Who is Nisarg Vyas?")
    print("Pipeline answer preview:\n", pipe_b["answer"][:160], "...")
    assert "Mr. Nisarg Vyas" in pipe_b["answer"]
    assert "Coordinator" in pipe_b["answer"]
    assert "Department of Data Science" in pipe_b["answer"]
    print("PASS: TEST B verified (RESOLVED -> Nisarg Vyas)!\n")

    print("=== TEST C: 'nisar vyash' (Typo Normalization) ===")
    # Check typo normalization preserved
    norm_c = query_rewrite.normalize_student_text("nisar vyash")
    print("Normalized text:", norm_c)
    assert "nisarg vyas" in norm_c

    res_c = counsellor.resolve_faculty("nisar vyash")
    print("counsellor.resolve_faculty status:", res_c["status"])
    assert res_c["status"] == "RESOLVED", f"Expected RESOLVED, got {res_c['status']}"
    assert res_c["person"]["canonical_name"] == "Nisarg Vyas"

    pipe_c = pipeline.answer("who is nisar vyash")
    print("Pipeline answer preview:\n", pipe_c["answer"][:160], "...")
    assert "Mr. Nisarg Vyas" in pipe_c["answer"]
    assert "Coordinator" in pipe_c["answer"]
    print("PASS: TEST C verified (Typo normalization -> Nisarg Vyas)!\n")

    print("=== TEST D: Unknown faculty name ===")
    res_d = counsellor.resolve_faculty("Who is Professor Rajesh Sharma?")
    print("counsellor.resolve_faculty status:", res_d["status"])
    assert res_d["status"] == "NOT_FOUND", f"Expected NOT_FOUND, got {res_d['status']}"

    pipe_d = pipeline.answer("Who is Professor Rajesh Sharma?")
    print("Pipeline answer preview:\n", pipe_d["answer"])
    assert "couldn't find" in pipe_d["answer"].lower() or "not available" in pipe_d["answer"].lower() or "office" in pipe_d["answer"].lower() or "available sxca materials" in pipe_d["answer"].lower() or "sorry" in pipe_d["answer"].lower()
    print("PASS: TEST D verified (NOT_FOUND)!\n")

    print("ALL 4 FACULTY RESOLUTION TESTS PASSED!")

if __name__ == "__main__":
    test_faculty_resolution()
