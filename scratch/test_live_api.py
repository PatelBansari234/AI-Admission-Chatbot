import requests
import json
import uuid

BASE_URL = "http://127.0.0.1:8000"

def test_live_chat():
    conv_id = str(uuid.uuid4())
    history = []

    # Turn 1: i complite the b.eco what are cource i take in master
    payload1 = {
        "question": "i complite the b.eco what are cource i take in master",
        "conversation_id": conv_id,
        "history": history
    }
    r1 = requests.post(f"{BASE_URL}/api/chat", json=payload1, timeout=10)
    d1 = r1.json()
    history.append({"role": "user", "content": payload1["question"]})
    history.append({"role": "assistant", "content": d1["answer"]})
    print("\n--- Multi-turn 1: 'i complite the b.eco what are cource i take in master' ---")
    print("Ans:", d1["answer"][:160], "...")
    assert "Big Data Analytics" in d1["answer"]

    # Turn 2: i dont want b.eco
    payload2 = {
        "question": "i dont want b.eco",
        "conversation_id": conv_id,
        "history": history
    }
    r2 = requests.post(f"{BASE_URL}/api/chat", json=payload2, timeout=10)
    d2 = r2.json()
    history.append({"role": "user", "content": payload2["question"]})
    history.append({"role": "assistant", "content": d2["answer"]})
    print("\n--- Multi-turn 2: 'i dont want b.eco' ---")
    print("Ans:", d2["answer"][:160], "...")
    assert "B.A. Programs (Economics" not in d2["answer"]

    # Turn 3: in master not ug
    payload3 = {
        "question": "in master not ug",
        "conversation_id": conv_id,
        "history": history
    }
    r3 = requests.post(f"{BASE_URL}/api/chat", json=payload3, timeout=10)
    d3 = r3.json()
    print("\n--- Multi-turn 3: 'in master not ug' ---")
    print("Ans:", d3["answer"][:160], "...")
    assert "Big Data Analytics" in d3["answer"] or "Postgraduate" in d3["answer"] or "Artificial Intelligence" in d3["answer"]
    assert "Mathematics: Bachelor's degree (any stream" not in d3["answer"]

    print("\nALL MULTI-TURN TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_live_chat()
