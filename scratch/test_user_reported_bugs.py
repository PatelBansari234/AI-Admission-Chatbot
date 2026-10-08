import sys
import os
sys.path.insert(0, os.getcwd())

from backend.rag import query_rewrite, counsellor, pipeline

def test_user_cases():
    print("=== Testing Case 1: Hostel Fee / Free ===")
    # Case 1: "what is free of hostel" with prior MSc AI in history
    history1 = [
        {"role": "user", "content": "What is the fee of MSc AI?"},
        {"role": "assistant", "content": "M.Sc. Artificial Intelligence fee is ₹50,000 per semester."}
    ]
    res = pipeline.answer("what is free of hostel", history=history1)
    print("Q: what is free of hostel")
    print("Ans:", res["answer"][:150], "...")
    assert "M.Sc. Artificial Intelligence" not in res["answer"], f"Error: active course leaked into hostel response: {res['answer']}"
    assert "Hostel accommodation at St. Xavier's College is **not free**" in res["answer"] or "hostel" in res["answer"].lower(), "Error: Hostel answer missing"
    print("PASS: Case 1 correctly answered with hostel details and not MSc AI fee!")

    print("\n=== Testing Case 2a: 12 in arts ===")
    # Case 2a: "i complite 12 in arts"
    res2a = pipeline.answer("i complite 12 in arts")
    print("Q: i complite 12 in arts")
    print("Ans:", res2a["answer"][:150], "...")
    assert "Since you're exploring options with a" not in res2a["answer"], f"Error: Fell through to generic fallback: {res2a['answer']}"
    assert "B.A." in res2a["answer"] or "Arts" in res2a["answer"], "Error: Arts programs missing"
    print("PASS: Case 2a returned Arts programs!")

    print("\n=== Testing Case 2b: 12th in commerce available courses ===")
    # Case 2b: "i complite 12th in commerce what are the cource available"
    res2b = pipeline.answer("i complite 12th in commerce what are the cource available")
    print("Q: i complite 12th in commerce what are the cource available")
    print("Ans:", res2b["answer"][:150], "...")
    assert "Since you're exploring options with a" not in res2b["answer"], f"Error: Fell through to generic fallback: {res2b['answer']}"
    assert "B.Com" in res2b["answer"] or "Commerce" in res2b["answer"], "Error: Commerce courses missing"
    print("PASS: Case 2b returned Commerce programs!")

    print("\n=== Testing Case 3: B.Eco graduate asking for master courses ===")
    # Case 3: "i complite the b.eco what are cource i take in master"
    res3 = pipeline.answer("i complite the b.eco what are cource i take in master")
    print("Q: i complite the b.eco what are cource i take in master")
    print("Ans:", res3["answer"][:150], "...")
    assert "M.Sc. Big Data Analytics" in res3["answer"] or "M.Sc. Artificial Intelligence" in res3["answer"], f"Error: BA Economics master options missing: {res3['answer']}"
    assert "undergraduate (UG) programme" not in res3["answer"], "Error: Mistakenly answered as UG inquiry!"
    print("PASS: Case 3 correctly gave PG master courses for BA Economics graduates!")

    print("\n=== Testing Case 4a: I don't want B.Eco ===")
    # Case 4a: "i dont want b.eco"
    history4 = [
        {"role": "user", "content": "What about BA Economics?"},
        {"role": "assistant", "content": "BA Economics is a 3-year UG program."}
    ]
    res4a = pipeline.answer("i dont want b.eco", history=history4)
    print("Q: i dont want b.eco")
    print("Ans:", res4a["answer"][:150], "...")
    assert "B.A. Programs (Economics" not in res4a["answer"], f"Error: Recommended Economics despite rejection: {res4a['answer']}"
    print("PASS: Case 4a did not recommend Economics in alternative B.A. programs!")

    print("\n=== Testing Case 4b: in master not ug ===")
    # Case 4b: "in master not ug"
    turns = [
        {"role": "user", "content": "i complite the b.eco what are cource i take in master"},
        {"role": "assistant", "content": "Sure! Here are PG options..."},
        {"role": "user", "content": "i dont want b.eco"},
        {"role": "assistant", "content": "Got it, no B.A. Economics..."},
    ]
    res4b = pipeline.answer("in master not ug", history=turns)
    print("Q: in master not ug")
    print("Ans:", res4b["answer"][:150], "...")
    assert "M.Sc. Big Data Analytics" in res4b["answer"] or "Postgraduate" in res4b["answer"] or "M.Sc." in res4b["answer"], f"Error: Master programs missing: {res4b['answer']}"
    assert "Mathematics: Bachelor's degree (any stream" not in res4b["answer"], f"Error: Hallucination present: {res4b['answer']}"
    print("PASS: Case 4b returned authoritative Master options without hallucination!")

    print("\nALL 4 USER CASES VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    test_user_cases()
