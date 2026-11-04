"""End-to-End Test Suite for Niaa Minimal Admin Panel & Chatbot Non-Regression.

Tests:
1. Admin Authentication & Role Protection (login, token validation, 401 unauthorized handling)
2. Overview Dashboard metrics
3. System Health Diagnostics (safe component probes, no leaked credentials)
4. PDF Knowledge Management:
   - List PDFs
   - Upload & Index new PDF (validates magic bytes, embeddings created, chunks stored)
   - Delete PDF safely (verifies chunks removed from Chroma without touching other docs)
5. Unanswered-Question Review (list fallback questions, toggle reviewed state)
6. Existing Chatbot Regression (greeting fast-path, faculty grounding, percentage understanding)
"""

import io
import json
import os
import sys
import urllib.request
import urllib.error

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_URL = "http://127.0.0.1:8000"


def request(method, path, data=None, token=None, files=None):
    url = f"{BASE_URL}{path}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    if files:
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        body = bytearray()
        for field_name, (filename, file_bytes, mime_type) in files.items():
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode())
            body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode())
            body.extend(file_bytes)
            body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode())
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        req = urllib.request.Request(url, data=bytes(body), headers=headers, method=method)
    elif data is not None:
        headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers, method=method)
    else:
        req = urllib.request.Request(url, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req) as resp:
            content = resp.read().decode("utf-8")
            return resp.status, json.loads(content) if content else {}
    except urllib.error.HTTPError as e:
        err_content = e.read().decode("utf-8")
        try:
            return e.code, json.loads(err_content)
        except Exception:
            return e.code, {"detail": err_content}


def test_admin_auth():
    print("\n--- [Test 1] Admin Authentication & Access Control ---")
    # 1. Invalid login
    status, res = request("POST", "/api/login", {"username": "admin", "password": "wrongpassword"})
    assert status == 401, f"Expected 401 for bad password, got {status}: {res}"
    print("  [PASS] Invalid password rejected with 401.")

    # 2. Valid login
    status, res = request("POST", "/api/login", {"username": "admin", "password": "admin123"})
    assert status == 200, f"Expected 200 for valid login, got {status}: {res}"
    token = res.get("token")
    assert token, "Token missing from login response"
    print("  [PASS] Valid admin credentials accepted, JWT token issued.")

    # 3. Unauthenticated access to admin endpoint
    status, res = request("GET", "/api/admin/overview")
    assert status == 401, f"Expected 401 for unauthenticated request, got {status}"
    print("  [PASS] Unauthenticated access properly rejected with 401.")

    # 4. Authenticated admin access
    status, res = request("GET", "/api/admin/me", token=token)
    assert status == 200 and res.get("role") == "admin", f"Failed admin verification: {res}"
    print("  [PASS] Admin role verified via /api/admin/me.")

    return token


def test_overview_metrics(token):
    print("\n--- [Test 2] Overview Dashboard Metrics ---")
    status, res = request("GET", "/api/admin/overview", token=token)
    assert status == 200, f"Overview endpoint failed: {res}"

    assert "total_questions" in res and res["total_questions"] >= 0
    assert "unanswered_questions" in res and res["unanswered_questions"] >= 0
    assert "indexed_documents" in res and res["indexed_documents"] >= 0
    assert "pdf_sources_count" in res and res["pdf_sources_count"] >= 0
    assert "searchable_chunks" in res and res["searchable_chunks"] > 0
    assert res["system_status"] in ("Healthy", "Degraded", "Unavailable")

    print(f"  [PASS] Overview metrics verified:")
    print(f"         Total Questions: {res['total_questions']}")
    print(f"         Unanswered Questions: {res['unanswered_questions']} ({res['unreviewed_count']} pending review)")
    print(f"         Indexed PDF Sources: {res['pdf_sources_count']}")
    print(f"         Searchable Chunks in ChromaDB: {res['searchable_chunks']}")
    print(f"         System Status: {res['system_status']}")


def test_system_health(token):
    print("\n--- [Test 3] System Health Diagnostics ---")
    status, res = request("GET", "/api/admin/system-health", token=token)
    assert status == 200, f"System health failed: {res}"

    # Check that credentials / secrets are NOT exposed
    res_str = json.dumps(res).lower()
    assert "password" not in res_str
    assert "secret" not in res_str
    assert "admission_bot:" not in res_str

    # Component checks
    assert res["api"]["status"] == "Healthy"
    assert res["database"]["status"] == "Healthy"
    assert res["vector_db"]["status"] == "Healthy"
    assert res["model"]["status"] in ("Healthy", "Degraded")

    print(f"  [PASS] Component health verified:")
    print(f"         API: {res['api']['status']} (v{res['api']['version']})")
    print(f"         Database: {res['database']['status']} ({res['database']['engine']}, probe latency: {res['database']['latency_ms']} ms)")
    print(f"         Vector DB: {res['vector_db']['status']} ({res['vector_db']['collection']}, {res['vector_db']['chunks']} chunks)")
    print(f"         Model: {res['model']['status']} ({res['model']['model_name']}, CUDA GPU: {res['model']['cuda_available']})")


def test_pdf_knowledge_lifecycle(token):
    print("\n--- [Test 4] PDF Knowledge Management (Upload & Safe Deletion) ---")
    # 1. List existing PDFs
    status, res = request("GET", "/api/admin/pdf-knowledge", token=token)
    assert status == 200
    initial_count = len(res["pdfs"])
    initial_chunks = res["index"]["chunks"]
    print(f"  [INFO] Initial PDF count: {initial_count}, Initial chunks in ChromaDB: {initial_chunks}")

    # 2. Reject non-PDF file upload
    status, err = request("POST", "/api/admin/pdf/upload", token=token, files={
        "file": ("malicious.txt", b"plain text", "text/plain")
    })
    assert status == 400, f"Expected 400 for non-pdf, got {status}: {err}"
    print("  [PASS] Non-PDF upload correctly rejected.")

    # 3. Create a valid single-page test PDF with real text
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader("data/raw/pdfs/SXC-Handbook-2025-26.pdf")
    writer = PdfWriter()
    writer.add_page(reader.pages[3])
    buf = io.BytesIO()
    writer.write(buf)
    pdf_content = buf.getvalue()

    test_filename = "SXCA-Admin-Verification-Notice-2026.pdf"

    # 4. Upload and index PDF
    status, upload_res = request("POST", "/api/admin/pdf/upload", token=token, files={
        "file": (test_filename, pdf_content, "application/pdf")
    })
    assert status == 200 and upload_res.get("success"), f"Upload failed: {upload_res}"
    assert upload_res["chunks"] > 0, "No chunks created for uploaded PDF"
    print(f"  [PASS] Uploaded '{test_filename}' successfully. Chunks indexed: {upload_res['chunks']}.")

    # 5. Verify PDF appears in indexed list
    status, res_after = request("GET", "/api/admin/pdf-knowledge", token=token)
    assert status == 200
    sources = [p["source"] for p in res_after["pdfs"]]
    assert test_filename in sources, f"Uploaded PDF '{test_filename}' not found in list: {sources}"
    print(f"  [PASS] Verified '{test_filename}' is present in knowledge base.")

    # 6. Safely Delete PDF
    status, del_res = request("DELETE", f"/api/admin/pdf/{test_filename}", token=token)
    assert status == 200 and del_res.get("success"), f"Deletion failed: {del_res}"
    print(f"  [PASS] Safely deleted '{test_filename}': {del_res['message']}")

    # 7. Verify chunks were cleaned from ChromaDB
    status, res_final = request("GET", "/api/admin/pdf-knowledge", token=token)
    final_sources = [p["source"] for p in res_final["pdfs"]]
    assert test_filename not in final_sources, f"Deleted PDF '{test_filename}' still present!"
    assert res_final["index"]["chunks"] == initial_chunks, (
        f"Expected chunks to return to initial count {initial_chunks}, got {res_final['index']['chunks']}"
    )
    print(f"  [PASS] Verified ChromaDB chunk count restored to {initial_chunks} with zero orphaned embeddings.")


def test_unanswered_question_review(token):
    print("\n--- [Test 5] Unanswered-Question Review & Tracking ---")
    status, res = request("GET", "/api/admin/unanswered-questions?limit=50", token=token)
    assert status == 200
    questions = res.get("unanswered_questions", [])
    assert len(questions) > 0, "No unanswered questions found in database"

    first_item = questions[0]
    chat_id = first_item["id"]
    initial_status = first_item.get("reviewed", False)
    target_status = not initial_status

    print(f"  [INFO] Question #{chat_id}: '{first_item['question']}' (current reviewed: {initial_status})")

    # Toggle review status
    status, rev_res = request(
        "POST",
        f"/api/admin/unanswered-questions/{chat_id}/review",
        token=token,
        data={"reviewed": target_status, "notes": "Audited by QA Engineer"},
    )
    assert status == 200 and rev_res.get("success"), f"Review toggle failed: {rev_res}"
    print(f"  [PASS] Successfully updated review status for #{chat_id} to {target_status}.")

    # Confirm persistence
    status, verify_res = request("GET", "/api/admin/unanswered-questions?limit=50", token=token)
    matched = [q for q in verify_res["unanswered_questions"] if q["id"] == chat_id][0]
    assert matched["reviewed"] == target_status, "Review status was not persisted!"
    assert matched["reviewed_by"] == "admin"
    print(f"  [PASS] Verified review state persisted in lightweight store without altering DB schema.")


def test_student_chatbot_regression():
    print("\n--- [Test 6] Student Chatbot Endpoints & Grounding Non-Regression ---")

    # 1. Greeting fast path
    status, res = request("POST", "/api/chat", {"question": "hello"})
    assert status == 200 and res["grounded"] is True
    print(f"  [PASS] Greeting: '{res['answer'][:40]}...' (grounded: {res['grounded']})")

    # 2. Anti-hallucination faculty test ("Nisarg Patel" must clarify, never invent email)
    status, res_np = request("POST", "/api/chat", {"question": "Nisarg Patel"})
    assert status == 200
    assert "nisarg.patel@sxca.edu.in" not in res_np["answer"].lower(), "Fabricated email for Nisarg Patel!"
    assert "Nisarg Vyas" in res_np["answer"] or "mean" in res_np["answer"].lower(), "Clarification not triggered!"
    print("  [PASS] Faculty anti-hallucination intact for 'Nisarg Patel'.")

    # 3. Verified faculty profile ("Nisarg Vyas")
    status, res_nv = request("POST", "/api/chat", {"question": "Nisarg Vyas"})
    assert status == 200
    assert "Nisarg Vyas" in res_nv["answer"]
    assert "nisarg.vyas@sxca.edu.in" in res_nv["answer"]
    print("  [PASS] Verified faculty profile intact for 'Nisarg Vyas'.")

    # 4. Normalized percentage test ("65% in 12th for BCA")
    status, res_pct = request("POST", "/api/chat", {"question": "I got 65% in 12th can I apply for BCA?"})
    assert status == 200
    assert "65%" in res_pct["answer"]
    assert "BCA" in res_pct["answer"]
    print("  [PASS] Percentage understanding intact for student admission query.")


def main():
    print("==============================================================")
    print("RUNNING E2E VERIFICATION FOR NIA ADMIN PANEL & STUDENT CHATBOT")
    print("==============================================================")
    token = test_admin_auth()
    test_overview_metrics(token)
    test_system_health(token)
    test_pdf_knowledge_lifecycle(token)
    test_unanswered_question_review(token)
    test_student_chatbot_regression()
    print("\n==============================================================")
    print("ALL ADMIN PANEL AND REGRESSION TESTS PASSED SUCCESSFULLY! (6/6)")
    print("==============================================================\n")


if __name__ == "__main__":
    main()
