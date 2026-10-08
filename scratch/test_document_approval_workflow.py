"""Test Suite: Document Approval Workflow & Status Lifecycle

Tests:
1. Uploading a document sets status to PENDING.
2. PENDING document chunks are NOT indexed in vector store and CANNOT be retrieved by RAG.
3. Non-admin user cannot approve, reject, or archive documents (returns 401/403).
4. Admin approves document -> status changes to ACTIVE, chunks indexed in vector store, RAG can retrieve it.
5. Admin archives active document -> status changes to ARCHIVED, chunks removed from vector store, RAG cannot retrieve it.
6. Admin approves archived/pending document -> status ACTIVE, RAG can retrieve it.
7. Admin rejects document -> status changes to REJECTED, chunks removed from vector store, RAG cannot retrieve it.
"""

import sys
import io
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.auth import create_token
from backend.rag import retriever, pipeline
from backend.db import repo

client = TestClient(app)

def print_result(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name} - {detail}" if detail else f"[{status}] {name}")
    if not passed:
        sys.exit(1)

def run_tests():
    print("============================================================")
    print("RUNNING DOCUMENT APPROVAL WORKFLOW TESTS")
    print("============================================================")

    # 1. Setup admin and student authorization headers
    admin_token = create_token("admin", role="admin")
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    student_token = create_token("student_user", role="student")
    student_headers = {"Authorization": f"Bearer {student_token}"}

    # Unique test PDF content
    unique_keyword = "QuantumRobotics2026SXCA"
    unique_filename = "test_quantum_robotics_syllabus.pdf"
    file_bytes = f"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n{unique_keyword} Special Admission Guidelines 2026.".encode("utf-8")

    # Clean up prior test runs if existing
    existing = repo.get_document(unique_filename)
    if existing:
        repo.delete_document(unique_filename)

    # --- Test 1: Upload -> PENDING ---
    files = {"file": (unique_filename, io.BytesIO(file_bytes), "application/pdf")}
    upload_res = client.post("/api/admin/upload", files=files, headers=admin_headers)
    print_result("Admin Upload endpoint", upload_res.status_code == 200, f"Status: {upload_res.status_code}")
    
    upload_data = upload_res.json()
    doc_id = upload_data.get("id")
    status_1 = upload_data.get("status")
    print_result("Uploaded document status is PENDING", status_1 == "PENDING", f"Status: {status_1}, ID: {doc_id}")

    # --- Test 2: PENDING document CANNOT be retrieved by RAG ---
    hits_pending = retriever.retrieve(f"Tell me about {unique_keyword}")
    retrieved_pending = any(unique_keyword.lower() in h.get("text", "").lower() for h in hits_pending)
    print_result("Pending document NOT searchable by RAG retriever", not retrieved_pending, f"Hits: {len(hits_pending)}")

    pipe_pending = pipeline.answer(f"What is {unique_keyword}?")
    print_result("Pending document RAG pipeline answer is NO_INFO", unique_keyword not in pipe_pending["answer"], f"Answer: {pipe_pending['answer'][:60]}...")

    # --- Test 3: Non-admin CANNOT approve, reject, or archive ---
    approve_student = client.post(f"/api/admin/documents/{doc_id}/approve", headers=student_headers)
    print_result("Student approve blocked", approve_student.status_code in (401, 403), f"Status: {approve_student.status_code}")

    reject_student = client.post(f"/api/admin/documents/{doc_id}/reject", headers=student_headers)
    print_result("Student reject blocked", reject_student.status_code in (401, 403), f"Status: {reject_student.status_code}")

    archive_student = client.post(f"/api/admin/documents/{doc_id}/archive", headers=student_headers)
    print_result("Student archive blocked", archive_student.status_code in (401, 403), f"Status: {archive_student.status_code}")

    no_auth_approve = client.post(f"/api/admin/documents/{doc_id}/approve")
    print_result("Unauthenticated approve blocked", no_auth_approve.status_code in (401, 403), f"Status: {no_auth_approve.status_code}")

    # --- Test 4: Admin Approve -> ACTIVE ---
    approve_res = client.post(f"/api/admin/documents/{doc_id}/approve", headers=admin_headers)
    print_result("Admin Approve endpoint", approve_res.status_code == 200, f"Status: {approve_res.status_code}")

    doc_active = repo.get_document(doc_id)
    print_result("Approved document DB status is ACTIVE", doc_active and doc_active["status"] == "ACTIVE", f"DB Status: {doc_active.get('status') if doc_active else None}")

    # ACTIVE document CAN be retrieved by RAG
    hits_active = retriever.retrieve(f"Tell me about {unique_keyword}")
    retrieved_active = any(unique_keyword.lower() in h.get("text", "").lower() for h in hits_active)
    print_result("Active document searchable by RAG retriever", retrieved_active, f"Hits: {len(hits_active)}")

    # --- Test 5: Admin Archive -> ARCHIVED ---
    archive_res = client.post(f"/api/admin/documents/{doc_id}/archive", headers=admin_headers)
    print_result("Admin Archive endpoint", archive_res.status_code == 200, f"Status: {archive_res.status_code}")

    doc_archived = repo.get_document(doc_id)
    print_result("Archived document DB status is ARCHIVED", doc_archived and doc_archived["status"] == "ARCHIVED", f"DB Status: {doc_archived.get('status') if doc_archived else None}")

    # ARCHIVED document CANNOT be retrieved by RAG
    hits_archived = retriever.retrieve(f"Tell me about {unique_keyword}")
    retrieved_archived = any(unique_keyword.lower() in h.get("text", "").lower() for h in hits_archived)
    print_result("Archived document NOT searchable by RAG retriever", not retrieved_archived, f"Hits: {len(hits_archived)}")

    # --- Test 6: Re-approve -> ACTIVE ---
    reapprove_res = client.post(f"/api/admin/documents/{doc_id}/approve", headers=admin_headers)
    print_result("Admin Re-approve endpoint", reapprove_res.status_code == 200)
    
    hits_reactive = retriever.retrieve(f"Tell me about {unique_keyword}")
    retrieved_reactive = any(unique_keyword.lower() in h.get("text", "").lower() for h in hits_reactive)
    print_result("Re-approved document searchable by RAG retriever", retrieved_reactive)

    # --- Test 7: Admin Reject -> REJECTED ---
    reject_res = client.post(f"/api/admin/documents/{doc_id}/reject", headers=admin_headers)
    print_result("Admin Reject endpoint", reject_res.status_code == 200, f"Status: {reject_res.status_code}")

    doc_rejected = repo.get_document(doc_id)
    print_result("Rejected document DB status is REJECTED", doc_rejected and doc_rejected["status"] == "REJECTED", f"DB Status: {doc_rejected.get('status') if doc_rejected else None}")

    # REJECTED document CANNOT be retrieved by RAG
    hits_rejected = retriever.retrieve(f"Tell me about {unique_keyword}")
    retrieved_rejected = any(unique_keyword.lower() in h.get("text", "").lower() for h in hits_rejected)
    print_result("Rejected document NOT searchable by RAG retriever", not retrieved_rejected, f"Hits: {len(hits_rejected)}")

    # Cleanup test document after verification
    repo.delete_document(unique_filename)

    print("\nALL DOCUMENT APPROVAL WORKFLOW TESTS PASSED PERFECTLY!")

if __name__ == "__main__":
    run_tests()
