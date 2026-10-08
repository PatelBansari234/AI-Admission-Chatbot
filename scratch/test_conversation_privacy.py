"""Comprehensive Conversation Privacy & Ownership Security Test Suite

Tests:
1. User A can create and interact with their own conversation.
2. User B attempting to access User A's conversation receives 404 Conversation not found.
3. Guessed / tampered conversation_id for another user receives 404 Conversation not found.
4. Guest Session A vs Guest Session B privacy isolation.
5. Admin behavior remains correct (admin authentication, admin endpoints, admin chat).
"""

import sys
import uuid
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.auth import create_token
from backend.db import repo

client = TestClient(app)

def print_result(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name} - {detail}" if detail else f"[{status}] {name}")
    if not passed:
        sys.exit(1)

def run_tests():
    print("=== RUNNING CONVERSATION PRIVACY SECURITY TESTS ===")

    # 1. Setup authenticated users & session tokens
    token_user_a = create_token("student_alice", role="student")
    headers_a = {"Authorization": f"Bearer {token_user_a}"}

    token_user_b = create_token("student_bob", role="student")
    headers_b = {"Authorization": f"Bearer {token_user_b}"}

    session_g1 = "session_token_guest_111"
    headers_g1 = {"X-Session-Token": session_g1}

    session_g2 = "session_token_guest_222"
    headers_g2 = {"X-Session-Token": session_g2}

    conv_id_a = str(uuid.uuid4())

    # --- Test 1: User A creates and interacts with own conversation ---
    res1 = client.post(
        "/api/chat",
        json={"question": "Who is Nisarg Vyas?", "conversation_id": conv_id_a},
        headers=headers_a,
    )
    print_result("User A own conversation turn 1", res1.status_code == 200, f"Status: {res1.status_code}")

    res1_followup = client.post(
        "/api/chat",
        json={"question": "What does he teach?", "conversation_id": conv_id_a},
        headers=headers_a,
    )
    data1_followup = res1_followup.json()
    has_nisarg = "Nisarg" in data1_followup.get("answer", "") or "Data Science" in data1_followup.get("answer", "")
    print_result("User A own conversation turn 2 (context recall)", res1_followup.status_code == 200 and has_nisarg, f"Answer: {data1_followup.get('answer')[:60]}...")

    # --- Test 2: User B attempting to access User A's conversation is denied with 404 ---
    res2 = client.post(
        "/api/chat",
        json={"question": "What does he teach?", "conversation_id": conv_id_a},
        headers=headers_b,
    )
    print_result("User B accessing User A conversation", res2.status_code == 404 and res2.json().get("detail") == "Conversation not found", f"Status: {res2.status_code}, Detail: {res2.json().get('detail')}")

    # --- Test 3: Guessed / modified conversation_id belonging to User A is denied ---
    guessed_id = conv_id_a  # User B trying to guess User A's exact UUID
    res3 = client.post(
        "/api/chat/stream",
        json={"question": "Tell me about his email", "conversation_id": guessed_id},
        headers=headers_b,
    )
    print_result("User B streaming request to User A conversation_id", res3.status_code == 404 and res3.json().get("detail") == "Conversation not found", f"Status: {res3.status_code}, Detail: {res3.json().get('detail')}")

    # --- Test 4: Guest Session isolation ---
    conv_id_g1 = str(uuid.uuid4())
    res_g1 = client.post(
        "/api/chat",
        json={"question": "Who is Dr. Pravida Raja?", "conversation_id": conv_id_g1},
        headers=headers_g1,
    )
    print_result("Guest 1 creating own conversation", res_g1.status_code == 200)

    # Guest 2 attempts to use Guest 1's conversation
    res_g2_attack = client.post(
        "/api/chat",
        json={"question": "What department is she in?", "conversation_id": conv_id_g1},
        headers=headers_g2,
    )
    print_result("Guest 2 accessing Guest 1 conversation", res_g2_attack.status_code == 404 and res_g2_attack.json().get("detail") == "Conversation not found", f"Status: {res_g2_attack.status_code}")

    # --- Test 5: Admin behavior remains correct ---
    # Login as default admin
    login_res = client.post("/api/login", json={"username": "admin", "password": "changeme"})
    print_result("Admin login", login_res.status_code == 200, f"Token obtained: {'token' in login_res.json()}")
    admin_token = login_res.json().get("token")
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # Admin access admin docs
    docs_res = client.get("/api/admin/documents", headers=admin_headers)
    print_result("Admin /api/admin/documents endpoint", docs_res.status_code == 200, f"Docs count: {len(docs_res.json().get('documents', []))}")

    # Admin chat conversation
    conv_id_admin = str(uuid.uuid4())
    admin_chat_res = client.post(
        "/api/chat",
        json={"question": "What courses are offered?", "conversation_id": conv_id_admin},
        headers=admin_headers,
    )
    print_result("Admin chat request", admin_chat_res.status_code == 200, f"Status: {admin_chat_res.status_code}")

    # User A trying to access Admin's chat conversation
    attack_admin_res = client.post(
        "/api/chat",
        json={"question": "Show history", "conversation_id": conv_id_admin},
        headers=headers_a,
    )
    print_result("User A accessing Admin's conversation", attack_admin_res.status_code == 404, f"Status: {attack_admin_res.status_code}")

    print("\nALL CONVERSATION PRIVACY TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
