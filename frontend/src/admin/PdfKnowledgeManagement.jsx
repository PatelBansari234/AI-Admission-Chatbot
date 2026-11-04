import { useEffect, useRef, useState } from "react";
import { authFetch, getToken } from "../api.js";

function formatBytes(bytes) {
  if (!bytes || bytes === 0) return "—";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

function formatDate(dateStr) {
  if (!dateStr) return "—";
  try {
    const d = new Date(dateStr);
    return isNaN(d.getTime()) ? dateStr : d.toLocaleDateString() + " " + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return dateStr;
  }
}

export default function PdfKnowledgeManagement({ notify }) {
  const [pdfs, setPdfs] = useState([]);
  const [indexStats, setIndexStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [deleteTarget, setDeleteTarget] = useState(null); // Document object to delete
  const [deleting, setDeleting] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");

  const fileInputRef = useRef(null);

  async function loadPdfs() {
    setLoading(true);
    try {
      const data = await authFetch("/api/admin/pdf-knowledge");
      setPdfs(data.pdfs || []);
      setIndexStats(data.index || null);
    } catch (err) {
      if (notify) notify(err.message || "Failed to load PDF knowledge records", false);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadPdfs();
  }, []);

  function handleFileSelect(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      if (notify) notify("Invalid file type: Only official PDF documents (.pdf) are accepted.", false);
      e.target.value = "";
      return;
    }

    if (file.size > 60 * 1024 * 1024) {
      if (notify) notify("File is too large: Maximum allowed PDF size is 60 MB.", false);
      e.target.value = "";
      return;
    }

    executeUpload(file);
  }

  function executeUpload(file) {
    const formData = new FormData();
    formData.append("file", file);

    setUploading(true);
    setUploadProgress(20);
    setUploadStatus(`Uploading ${file.name}...`);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/admin/pdf/upload");
    xhr.setRequestHeader("Authorization", `Bearer ${getToken()}`);

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        const percent = Math.round((event.loaded / event.total) * 60);
        setUploadProgress(percent);
        if (percent >= 60) {
          setUploadStatus("Processing text, extracting tables & generating vector embeddings...");
        }
      }
    };

    xhr.onload = () => {
      setUploading(false);
      setUploadProgress(100);
      try {
        const res = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) {
          if (notify) notify(res.message || `Successfully indexed ${file.name}`);
          setUploadStatus(`Indexed ${res.chunks} chunks for ${res.filename}!`);
          loadPdfs();
        } else {
          if (notify) notify(res.detail || "PDF upload and indexing failed", false);
          setUploadStatus("");
        }
      } catch (err) {
        if (notify) notify("Server returned an unexpected response", false);
        setUploadStatus("");
      }
      if (fileInputRef.current) fileInputRef.current.value = "";
    };

    xhr.onerror = () => {
      setUploading(false);
      if (notify) notify("Network error while uploading PDF", false);
      setUploadStatus("");
      if (fileInputRef.current) fileInputRef.current.value = "";
    };

    xhr.send(formData);
  }

  async function confirmDelete() {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      const data = await authFetch(`/api/admin/pdf/${encodeURIComponent(deleteTarget.source)}`, {
        method: "DELETE",
      });
      if (notify) notify(data.message || `Successfully removed ${deleteTarget.source}`);
      setDeleteTarget(null);
      loadPdfs();
    } catch (err) {
      if (notify) notify(err.message || "Failed to safely remove document", false);
    } finally {
      setDeleting(false);
    }
  }

  const filteredPdfs = pdfs.filter((p) =>
    p.source.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div>
      {/* Upload PDF Section */}
      <div className="admin-card">
        <div className="admin-card-header">
          <h3 className="admin-card-title">📤 Add New Knowledge PDF</h3>
          {indexStats && (
            <span className="admin-badge">
              {indexStats.chunks?.toLocaleString()} Total Chunks in ChromaDB
            </span>
          )}
        </div>
        <div className="admin-card-body">
          <input
            type="file"
            ref={fileInputRef}
            accept=".pdf"
            onChange={handleFileSelect}
            style={{ display: "none" }}
            disabled={uploading}
          />

          <div
            className="admin-upload-zone"
            onClick={() => !uploading && fileInputRef.current?.click()}
          >
            <div className="admin-upload-icon">📁</div>
            <h4 style={{ margin: "0 0 6px", fontSize: 15, color: "#0f172a" }}>
              {uploading ? uploadStatus : "Click to select and upload an official SXCA PDF"}
            </h4>
            <p style={{ margin: 0, fontSize: 13, color: "#64748b" }}>
              Accepts <code>.pdf</code> files up to 60 MB. Extracts course tables, chunks text, and builds vector embeddings automatically.
            </p>

            {uploading && (
              <div style={{ marginTop: 16 }}>
                <div className="admin-progress-container">
                  <div
                    className="admin-progress-bar"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
                <small style={{ color: "#0b2545", fontWeight: 600 }}>
                  {uploadStatus} ({uploadProgress}%)
                </small>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Existing PDFs Table */}
      <div className="admin-card">
        <div className="admin-card-header">
          <h3 className="admin-card-title">📑 Indexed PDF Documents ({pdfs.length})</h3>
          <input
            type="text"
            className="admin-form-input"
            style={{ width: 240, padding: "6px 12px", fontSize: 13 }}
            placeholder="Search by filename..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <div className="admin-card-body" style={{ padding: 0 }}>
          <div className="admin-table-wrapper">
            <table className="admin-table">
              <thead>
                <tr>
                  <th>Filename / Source</th>
                  <th>Status</th>
                  <th>Chunks</th>
                  <th>Size</th>
                  <th>Last Updated</th>
                  <th>Uploaded By</th>
                  <th style={{ textAlign: "right" }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={7} style={{ textAlign: "center", padding: 30, color: "#64748b" }}>
                      Loading indexed PDFs...
                    </td>
                  </tr>
                ) : filteredPdfs.length === 0 ? (
                  <tr>
                    <td colSpan={7} style={{ textAlign: "center", padding: 30, color: "#64748b" }}>
                      {searchTerm ? "No documents match your search filter." : "No PDFs currently indexed."}
                    </td>
                  </tr>
                ) : (
                  filteredPdfs.map((pdf) => (
                    <tr key={pdf.source}>
                      <td>
                        <strong style={{ color: "#0b2545" }}>{pdf.source}</strong>
                        {!pdf.exists_on_disk && (
                          <span style={{ marginLeft: 8, color: "#b91c1c", fontSize: 11 }}>
                            (missing on disk)
                          </span>
                        )}
                      </td>
                      <td>
                        <span className={`admin-status-badge ${pdf.status === "ACTIVE" ? "admin-status-active" : "admin-status-pending"}`}>
                          {pdf.status}
                        </span>
                      </td>
                      <td>
                        <span className="admin-badge" style={{ background: "#f1f5f9", color: "#334155" }}>
                          {pdf.chunks} chunks
                        </span>
                      </td>
                      <td>{formatBytes(pdf.file_size_bytes)}</td>
                      <td style={{ fontSize: 12.5, color: "#64748b" }}>
                        {formatDate(pdf.updated_at || pdf.added_at)}
                      </td>
                      <td style={{ fontSize: 12.5, color: "#64748b" }}>{pdf.uploaded_by || "system"}</td>
                      <td style={{ textAlign: "right" }}>
                        <button
                          className="admin-action-btn admin-btn-danger"
                          onClick={() => setDeleteTarget(pdf)}
                          title="Safely remove this document and delete its embeddings"
                        >
                          🗑 Delete PDF
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Safe Deletion Confirmation Modal */}
      {deleteTarget && (
        <div className="admin-modal-overlay">
          <div className="admin-modal-card">
            <div className="admin-modal-header">
              <h4 className="admin-modal-title">⚠️ Confirm PDF Deletion</h4>
              <button
                className="admin-link-btn"
                onClick={() => setDeleteTarget(null)}
                disabled={deleting}
              >
                ✕
              </button>
            </div>
            <div className="admin-modal-body">
              <p>
                Are you sure you want to delete{" "}
                <strong style={{ color: "#0b2545" }}>{deleteTarget.source}</strong>?
              </p>
              <div style={{ background: "#fef2f2", border: "1px solid #fecaca", padding: 12, borderRadius: 8, fontSize: 13, color: "#991b1b" }}>
                <strong>Safe Deletion Guarantee:</strong>
                <ul style={{ margin: "6px 0 0", paddingLeft: 20 }}>
                  <li>All <strong>{deleteTarget.chunks}</strong> embeddings and chunks belonging to this PDF will be deleted from ChromaDB.</li>
                  <li>Other documents and web crawls will <strong>not</strong> be affected.</li>
                  <li>The file will be unlinked from disk and database records will be cleaned.</li>
                </ul>
              </div>
            </div>
            <div className="admin-modal-footer">
              <button
                className="admin-action-btn admin-btn-secondary"
                onClick={() => setDeleteTarget(null)}
                disabled={deleting}
              >
                Cancel
              </button>
              <button
                className="admin-action-btn admin-btn-danger"
                onClick={confirmDelete}
                disabled={deleting}
              >
                {deleting ? "Deleting chunks..." : "Yes, Safely Delete"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
