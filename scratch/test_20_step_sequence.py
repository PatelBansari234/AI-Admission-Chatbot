"""20-Step Continuous Realistic Dialogue Sequence Test for Niaa.
Simulates a real student having a 20-turn continuous conversation with state preservation,
clean context switching, language switching, and zero contamination.
"""

import sys
sys.path.insert(0, ".")
import uuid
from backend.rag import pipeline
from backend.db import repo

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def run_20_step_dialogue():
    conversation_id = str(uuid.uuid4())
    repo.get_or_create_conversation(conversation_id)

    turns = [
        # Turn 1
        ("hii", ["Niaa", "St. Xavier's"]),
        # Turn 2
        ("bca", ["BCA", "undergraduate"]),
        # Turn 3
        ("bca fee", ["29,750", "Self-Financed"]),
        # Turn 4
        ("cai 2 date", ["October", "CIA-2"]),
        # Turn 5
        ("who is david", ["David", "Director"]),
        # Turn 6
        ("what is his department", ["Computer Sciences & Data Science"]),
        # Turn 7
        ("who is principal", ["Sebastian"]),
        # Turn 8
        ("college phone number", ["2630 2872", "info@sxca.edu.in"]),
        # Turn 9
        ("what about hostel", ["Hostel", "₹15,000"]),
        # Turn 10
        ("what documents", ["Marksheet", "Certificate"]),
        # Turn 11
        ("bda fee", ["44,950"]),
        # Turn 12
        ("msc ai", ["Artificial Intelligence", "postgraduate"]),
        # Turn 13
        ("free", ["Do you mean the fee for"]),
        # Turn 14
        ("yes", ["50,000"]),
        # Turn 15
        ("mane Gujarati ma j answer aapo", ["ગુજરાતીમાં"]),
        # Turn 16
        ("science background che, kaya course available che?", ["BCA", "B.Sc."]),
        # Turn 17
        ("sports facilities", ["Gym"]),
        # Turn 18
        ("what is culfest", ["Culfest"]),
        # Turn 19
        ("placement details", ["TCS"]),
        # Turn 20
        ("thank you", ["આભાર"]),
    ]

    print(f"\n=======================================================")
    print(f"Starting 20-Turn Continuous Dialogue Test [{conversation_id}]")
    print(f"=======================================================\n")

    history = []
    for step_num, (user_msg, expected_keywords) in enumerate(turns, 1):
        # We pass history array or conversation_id - both work seamlessly!
        res = pipeline.answer(user_msg, history=history, conversation_id=conversation_id)
        ans = res["answer"]
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": ans, "sources": res.get("sources", [])})

        print(f"Step {step_num:02d}: Student: \"{user_msg}\"")
        first_line = ans.strip().split("\n")[0]
        print(f"         Niaa:    {first_line[:90]}...")

        # Assertions
        for kw in expected_keywords:
            if kw.lower() not in ans.lower():
                raise AssertionError(f"Step {step_num} FAILED: Expected keyword '{kw}' not in answer:\n{ans}")

        # Extra checks on sensitive steps
        if step_num == 4:
            # Ensure CAI 2 didn't repeat BCA fees
            if "29,750" in ans:
                raise AssertionError(f"Step 4 context leak: CAI 2 answer contained BCA fee 29,750!")
        if step_num == 11:
            # Ensure BDA fee didn't return MSc AI fee 50,000
            if "50,000" in ans:
                raise AssertionError(f"Step 11 context leak: BDA fee contained MSc AI fee 50,000!")

        print(f"         ✓ PASSED (grounded={res.get('grounded', False)}, sources={len(res.get('sources', []))})")

    print(f"\n=======================================================")
    print(f"🎉 SUCCESS: All 20 turns completed flawlessly!")
    print(f"=======================================================\n")

if __name__ == "__main__":
    run_20_step_dialogue()
